"""
detection/nutrition_region.py

Locates the FULL Nutrition Information / Facts section, treating it as a table.
Uses OCR line builder, line classifier, block detector, and table heuristics to
determine precise bounds. Operates completely independently of Ingredients.
"""

import numpy as np
import config
from detection.geometry import (
    union_rect,
    vertical_distance,
    horizontal_overlap_ratio,
    median_line_height,
    sort_reading_order,
    cluster_by_proximity,
    validate_region,
    rect_height,
    rect_width,
)
from detection.line_builder import reconstruct_lines
from detection.line_classifier import classify_lines
from detection.block_detector import detect_logical_blocks
from detection.ocr_detector import normalize_ocr_text, find_anchor_candidates
from detection.section_signals import classify_nutrition_row, classify_ingredient_line, detect_section_boundaries

def find_nutrition_anchor_candidates(lines, top_n=None):
    """
    Finds top-N candidates for Nutrition headings.
    """
    top_n = top_n if top_n is not None else config.NUM_ANCHOR_CANDIDATES_TO_TRY
    candidates = find_anchor_candidates(
        lines, config.NUTRITION_ANCHORS, threshold=config.FUZZY_ANCHOR_THRESHOLD
    )
    return candidates[:top_n]

def _expand_directional(seed_rect, band, ordered_lines, line_h, max_gap, band_tolerance, ingredient_vocab=None):
    """
    Greedily walks lines in one direction, absorbing nutrient table rows.
    """
    collected = []
    rejected = []
    current_rect = list(seed_rect)
    band_left, band_right = band
    stop_reason = "exhausted_candidate_lines"
    max_block_height = line_h * config.REGION_MAX_BLOCK_HEIGHT_LINE_FACTOR
    block_start_y = seed_rect[1]

    for ln in ordered_lines:
        if len(collected) >= config.REGION_EXPANSION_MAX_LINES:
            stop_reason = "max_lines_reached"
            break

        rect = ln["rect"]
        gap = vertical_distance(current_rect, rect)
        if gap > max_gap:
            stop_reason = "vertical_gap_exceeded"
            rejected.append({"text": ln["text"], "reason": stop_reason})
            break

        # Check total block height sanity
        prospective_height = max(rect[3], current_rect[3]) - min(block_start_y, rect[1])
        if prospective_height > max_block_height:
            stop_reason = "max_block_height_exceeded"
            rejected.append({"text": ln["text"], "reason": stop_reason})
            break

        # Check column alignment
        left_close = abs(rect[0] - band_left) <= band_tolerance
        overlap_ratio = horizontal_overlap_ratio((band_left, 0, band_right, 1), (rect[0], 0, rect[2], 1))
        overlaps_band = overlap_ratio >= config.REGION_BAND_OVERLAP_MIN_RATIO

        if not (left_close or overlaps_band):
            # If the line is a very strong nutrition row, be more lenient on geometry
            if ln.get("nutrition_score", 0.0) < 0.35:
                rejected.append({"text": ln["text"], "reason": "outside_table_band"})
                continue

        # Semantic disqualification: never absorb ingredients
        if ln.get("ingredient_score", 0.0) > 0.65 and not ln.get("nutrition_score", 0.0) > 0.4:
            stop_reason = "ingredient_line_detected"
            rejected.append({"text": ln["text"], "reason": stop_reason})
            break

        # Stop words / other sections
        is_stop, stop_anchor, stop_score, reason = detect_section_boundaries(
            ln["text"], config.SECTION_STOP_WORDS
        )
        if is_stop or ln.get("other_score", 0.0) > 0.75:
            stop_reason = f"stop_word:{stop_anchor or 'other_section'}({reason})"
            rejected.append({"text": ln["text"], "reason": stop_reason})
            break

        collected.append(ln)
        current_rect = union_rect([current_rect, rect])
        band_left = min(band_left, rect[0])
        band_right = max(band_right, rect[2])

    return collected, rejected, stop_reason, (band_left, band_right)

def expand_nutrition_region(anchor_line, all_lines, image_shape, ingredient_vocab=None):
    """
    Expands the nutrition region from the anchor heading.
    """
    anchor_rect = anchor_line["rect"]
    others = [ln for ln in all_lines if ln is not anchor_line]
    below = [ln for ln in others if ln["rect"][1] >= anchor_rect[1] - 1e-6]
    ordered = sort_reading_order(below)

    line_h = median_line_height([ln["rect"] for ln in all_lines]) or 20.0
    max_gap = line_h * config.REGION_EXPANSION_MAX_LINE_GAP_FACTOR
    band_tolerance = line_h * config.REGION_BAND_LEFT_TOLERANCE_FACTOR

    collected_rest, rejected, stop_reason, band = _expand_directional(
        anchor_rect, (anchor_rect[0], anchor_rect[2]), ordered, line_h, max_gap, band_tolerance,
        ingredient_vocab=ingredient_vocab,
    )

    collected = [anchor_line] + collected_rest
    bbox = union_rect([ln["rect"] for ln in collected])

    debug_info = {
        "stop_reason": stop_reason,
        "rejected_lines": rejected,
        "num_lines_collected": len(collected),
        "column_band": list(band),
    }
    return bbox, collected, debug_info

def _table_geometry_score(cluster_lines):
    """
    Evaluates how much a cluster of lines resembles a nutrition table.
    """
    if len(cluster_lines) < 2:
        return 0.0

    rects = [ln["rect"] for ln in cluster_lines]
    heights = [rect_height(r) for r in rects]
    widths = [rect_width(r) for r in rects]

    mean_h = float(np.mean(heights)) if heights else 1.0
    std_h = float(np.std(heights)) if heights else 0.0
    height_consistency = max(0.0, 1.0 - min(std_h / max(mean_h, 1e-6), 1.0))

    mean_w = float(np.mean(widths)) if widths else 1.0
    shortness = 1.0 if mean_w < 600 else max(0.0, 1.0 - (mean_w - 600) / 1000.0)

    bbox = union_rect(rects)
    bbox_area = max(1.0, (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
    items_area = sum(rect_width(r) * rect_height(r) for r in rects)
    compactness = min(1.0, items_area / bbox_area * 3.0)

    return float(0.4 * height_consistency + 0.3 * shortness + 0.3 * compactness)

def detect_nutrition_rows(lines, image_shape, ingredient_vocab=None):
    """
    Fallback table-geometry region builder if no nutrition facts header is found.
    """
    hits = []
    for ln in lines:
        classification = classify_nutrition_row(ln["text"])
        if classification["keyword_hits"]:
            hits.append(ln)

    if len(hits) < config.NUTRITION_TABLE_MIN_KEYWORDS:
        return None

    clusters = cluster_by_proximity(
        hits, max_gap_factor=config.NUTRITION_TABLE_CLUSTER_DIST_FACTOR
    )

    best_cluster = None
    best_score = 0.0
    best_terms = set()

    for cluster in clusters:
        distinct_terms = set()
        for ln in cluster:
            distinct_terms.update(classify_nutrition_row(ln["text"])["keyword_hits"])

        if len(distinct_terms) < config.NUTRITION_TABLE_MIN_KEYWORDS:
            continue

        table_score = _table_geometry_score(cluster)
        term_score = min(1.0, len(distinct_terms) / 8.0)
        combined = 0.55 * term_score + 0.45 * table_score

        if combined > best_score:
            best_score = combined
            best_cluster = cluster
            best_terms = distinct_terms

    if best_cluster is None:
        return None

    seed_line = sort_reading_order(best_cluster)[0]
    line_h = median_line_height([ln["rect"] for ln in lines]) or 20.0
    max_gap = line_h * config.REGION_EXPANSION_MAX_LINE_GAP_FACTOR
    band_tolerance = line_h * config.REGION_BAND_LEFT_TOLERANCE_FACTOR
    header_gap = line_h * config.NUTRITION_HEADER_FOOTER_GAP_FACTOR

    seed_rect = seed_line["rect"]
    others = [ln for ln in lines if ln is not seed_line]

    below = sort_reading_order([ln for ln in others if ln["rect"][1] >= seed_rect[1] - 1e-6])
    above = sort_reading_order([ln for ln in others if ln["rect"][1] < seed_rect[1] - 1e-6])[::-1]

    down_collected, down_rejected, down_stop, band_after_down = _expand_directional(
        seed_rect, (seed_rect[0], seed_rect[2]), below, line_h, max_gap, band_tolerance,
        ingredient_vocab=ingredient_vocab,
    )
    up_collected, up_rejected, up_stop, band_after_up = _expand_directional(
        seed_rect, (seed_rect[0], seed_rect[2]), above, line_h, header_gap, band_tolerance,
        ingredient_vocab=ingredient_vocab,
    )

    collected = list(reversed(up_collected)) + [seed_line] + down_collected
    bbox = union_rect([ln["rect"] for ln in collected])

    confidence = round(float(min(0.9, best_score)), 3)

    debug_info = {
        "stop_reason_down": down_stop,
        "stop_reason_up": up_stop,
        "rejected_lines": up_rejected + down_rejected,
        "num_lines_collected": len(collected),
        "column_band": [
            min(band_after_down[0], band_after_up[0]),
            max(band_after_down[1], band_after_up[1]),
        ],
    }

    return {
        "bbox": bbox,
        "confidence": confidence,
        "anchor": None,
        "matched_items": collected,
        "matched_terms": sorted(best_terms),
        "method": "table_geometry_fallback",
        "debug": debug_info,
    }

def score_nutrition_region(collected_lines, anchor_score, stop_reason, ingredient_vocab=None):
    """
    Computes confidence score for the nutrition facts table region.
    """
    if not collected_lines:
        return 0.0, []

    matched_terms = set()
    for ln in collected_lines:
        matched_terms.update(classify_nutrition_row(ln["text"])["keyword_hits"])

    avg_nut_score = float(np.mean([ln.get("nutrition_score", 0.0) for ln in collected_lines]))

    line_bonus = min(0.3, 0.04 * max(0, len(collected_lines) - 1))
    term_bonus = min(0.2, 0.03 * len(matched_terms))

    boundary_bonus = 0.1 if stop_reason and (stop_reason.startswith("stop_word") or stop_reason == "ingredient_line_detected") else 0.03

    contamination_penalty = 0.0
    for ln in collected_lines:
        if ln.get("ingredient_score", 0.0) > 0.6:
            contamination_penalty += 0.15
        if ln.get("other_score", 0.0) > 0.75:
            contamination_penalty += 0.1

    confidence = (
        (anchor_score / 100.0) * 0.4
        + avg_nut_score * 0.25
        + line_bonus
        + term_bonus
        + boundary_bonus
        - contamination_penalty
    )
    return round(float(max(0.0, min(0.98, confidence))), 3), sorted(matched_terms)

def _vocabulary_fallback_candidate_v2(blocks):
    """
    Fallback candidate from block detector: returns best NUTRITION_TABLE_BLOCK.
    """
    nut_blocks = [b for b in blocks if b["type"] == "NUTRITION_TABLE_BLOCK"]
    if not nut_blocks:
        return None

    scored = []
    for b in nut_blocks:
        lines = b["lines"]
        avg_nut = float(np.mean([ln.get("nutrition_score", 0.0) for ln in lines]))
        confidence = min(0.7, 0.3 + 0.4 * avg_nut)

        matched_terms = set()
        for ln in lines:
            matched_terms.update(classify_nutrition_row(ln["text"])["keyword_hits"])

        scored.append({
            "bbox": b["rect"],
            "confidence": round(float(confidence), 3),
            "anchor": None,
            "matched_items": lines,
            "matched_terms": sorted(matched_terms),
            "method": "table_geometry_fallback",
            "debug": {"num_lines_collected": len(lines), "stop_reason": "block_cluster"}
        })

    scored.sort(key=lambda x: x["confidence"], reverse=True)
    return scored[0] if scored else None

def detect_nutrition_region(items, image_shape, ingredient_vocab=None, debug=False):
    """
    Main entry point for Nutrition Facts Table region detection.
    """
    include_debug = debug or config.DEBUG_REGION_DETECTION

    # 1. Reconstruct logical lines
    lines = reconstruct_lines(items, image_shape)
    if not lines:
        return {
            "bbox": None, "confidence": 0.0, "anchor": None,
            "matched_items": [], "matched_terms": [], "method": "none",
        }

    # 2. Score lines semantically
    lines = classify_lines(lines, ingredient_vocab)

    # 3. Detect logical blocks
    blocks = detect_logical_blocks(lines, image_shape, ingredient_vocab)

    # 4. Heading-driven candidates
    anchor_candidates = find_nutrition_anchor_candidates(lines)
    scored_candidates = []

    for cand in anchor_candidates:
        anchor_line = cand["line"]
        anchor_text = cand["matched_anchor"]
        anchor_score = cand["score"]

        bbox, collected, debug_info = expand_nutrition_region(
            anchor_line, lines, image_shape, ingredient_vocab=ingredient_vocab
        )
        confidence, matched_terms = score_nutrition_region(
            collected, anchor_score, debug_info["stop_reason"], ingredient_vocab
        )
        debug_info["anchor_text"] = anchor_line["text"]
        debug_info["anchor_score"] = anchor_score

        scored_candidates.append({
            "bbox": bbox,
            "confidence": confidence,
            "anchor": anchor_text,
            "matched_items": collected,
            "matched_terms": matched_terms,
            "method": "anchor_expansion",
            "debug": debug_info,
        })

    # 5. Fallback via block detector
    block_fallback = _vocabulary_fallback_candidate_v2(blocks)
    if block_fallback is not None:
        scored_candidates.append(block_fallback)

    # 6. Fallback via keyword row proximity
    fallback = detect_nutrition_rows(lines, image_shape, ingredient_vocab=ingredient_vocab)
    if fallback is not None:
        scored_candidates.append(fallback)

    if not scored_candidates:
        return {
            "bbox": None,
            "confidence": 0.0,
            "anchor": None,
            "matched_items": [],
            "matched_terms": [],
            "method": "none",
        }

    # Pick the highest confidence candidate
    scored_candidates.sort(key=lambda c: c["confidence"], reverse=True)
    best = scored_candidates[0]

    # Validate region coordinates
    best_bbox = validate_region(best["bbox"], image_shape)
    best["bbox"] = best_bbox

    if include_debug:
        best["debug"]["all_candidate_scores"] = [
            {"method": c["method"], "anchor": c["anchor"], "confidence": c["confidence"]}
            for c in scored_candidates
        ]
    else:
        best.pop("debug", None)

    return best
