"""
detection/ingredient_region.py

Locates the FULL Ingredients / Composition section of a package image.
Uses OCR line reconstruction, block detection, and semantic line scoring to
ensure a tight bounding box that excludes unrelated surrounding text.
"""

import numpy as np
import config
from detection.geometry import (
    union_rect,
    vertical_distance,
    horizontal_overlap_ratio,
    median_line_height,
    sort_reading_order,
    validate_region,
)
from detection.line_builder import reconstruct_lines
from detection.line_classifier import classify_lines
from detection.block_detector import detect_logical_blocks
from detection.ocr_detector import normalize_ocr_text, find_anchor_candidates
from detection.section_signals import classify_nutrition_row, detect_section_boundaries

def split_merged_lines(lines, image_width):
    """
    Splits any line that contains a horizontal gap wider than the threshold.
    """
    split_lines = []
    line_h = median_line_height([ln["rect"] for ln in lines]) or 20.0
    gap_threshold = max(25.0, min(image_width * 0.05, line_h * 1.2))
    
    for ln in lines:
        items = ln.get("items", [])
        if len(items) <= 1:
            split_lines.append(ln)
            continue
            
        sorted_items = sorted(items, key=lambda it: it["rect"][0])
        
        current_group = [sorted_items[0]]
        groups = [current_group]
        
        for it in sorted_items[1:]:
            prev_it = current_group[-1]
            gap = it["rect"][0] - prev_it["rect"][2]
            if gap > gap_threshold:
                current_group = [it]
                groups.append(current_group)
            else:
                current_group.append(it)
                
        if len(groups) == 1:
            split_lines.append(ln)
        else:
            for gp in groups:
                merged_text = " ".join(m["text"] for m in gp)
                merged_rect = union_rect([m["rect"] for m in gp])
                avg_conf = float(np.mean([m.get("confidence", 0.0) for m in gp]))
                
                new_ln = dict(ln)
                new_ln["text"] = merged_text
                new_ln["rect"] = merged_rect
                new_ln["items"] = gp
                new_ln["confidence"] = avg_conf
                
                x1, y1, x2, y2 = merged_rect
                new_ln["center"] = [float((x1 + x2) / 2.0), float((y1 + y2) / 2.0)]
                new_ln["height"] = float(y2 - y1)
                new_ln["width"] = float(x2 - x1)
                
                # Copy semantic scores if present
                for key in ["ingredient_score", "nutrition_score", "other_score"]:
                    if key in ln:
                        new_ln[key] = ln[key]
                        
                split_lines.append(new_ln)
                
    return split_lines

def find_ingredient_anchor_candidates(lines, top_n=None):
    """
    Finds lines that look like Ingredients headings.
    """
    top_n = top_n if top_n is not None else config.NUM_ANCHOR_CANDIDATES_TO_TRY
    candidates = find_anchor_candidates(
        lines, config.ALL_INGREDIENT_ANCHORS, threshold=config.FUZZY_ANCHOR_THRESHOLD
    )
    if not candidates:
        return []

    # Sort: strong anchors first, weak ("contains") later
    strong = [c for c in candidates if c["matched_anchor"] != "contains"]
    weak = [c for c in candidates if c["matched_anchor"] == "contains"]
    return (strong + weak)[:top_n]

def expand_ingredient_region(anchor_line, all_lines, image_shape, anchor_column_lines=None):
    """
    Expands the region from the anchor line, accepting lines based on geometry and semantics.
    """
    anchor_rect = anchor_line["rect"]
    if anchor_column_lines is not None:
        col_set = {id(ln) for ln in anchor_column_lines}
        others = [ln for ln in all_lines if ln is not anchor_line and id(ln) in col_set]
    else:
        others = [ln for ln in all_lines if ln is not anchor_line]
    ordered = sort_reading_order(others)

    line_h = median_line_height([ln["rect"] for ln in all_lines]) or 20.0
    max_gap = line_h * config.REGION_EXPANSION_MAX_LINE_GAP_FACTOR
    band_tolerance = line_h * config.REGION_BAND_LEFT_TOLERANCE_FACTOR
    max_block_height = line_h * config.REGION_MAX_BLOCK_HEIGHT_LINE_FACTOR

    collected = [anchor_line]
    current_rect = list(anchor_rect)
    band_left, band_right = anchor_rect[0], anchor_rect[2]

    rejected = []
    stop_reason = "exhausted_candidate_lines"

    for ln in ordered:
        if len(collected) >= config.REGION_EXPANSION_MAX_LINES:
            stop_reason = "max_lines_reached"
            break

        rect = ln["rect"]

        # If it is far above the anchor, skip it
        if rect[1] < anchor_rect[1] - line_h * 0.3:
            rejected.append({"text": ln["text"], "reason": "above_anchor"})
            continue

        # Check vertical gap
        gap = vertical_distance(current_rect, rect)
        if gap > max_gap:
            stop_reason = "vertical_gap_exceeded"
            rejected.append({"text": ln["text"], "reason": stop_reason})
            break

        # Check total height sanity
        prospective_height = max(rect[3], current_rect[3]) - min(anchor_rect[1], rect[1])
        if prospective_height > max_block_height:
            stop_reason = "max_block_height_exceeded"
            rejected.append({"text": ln["text"], "reason": stop_reason})
            break

        # Check left margin proximity or horizontal band overlap
        left_close = abs(rect[0] - band_left) <= band_tolerance
        overlap_ratio = horizontal_overlap_ratio((band_left, 0, band_right, 1), (rect[0], 0, rect[2], 1))
        overlaps_band = overlap_ratio >= config.REGION_BAND_OVERLAP_MIN_RATIO
        
        if not (left_close or overlaps_band):
            # If the line has very high ingredient score, we can be more lenient on geometry
            if ln.get("ingredient_score", 0.0) < 0.35:
                rejected.append({"text": ln["text"], "reason": "outside_paragraph_band"})
                continue

        # Semantic checks: stop if the line is clearly part of another section
        # e.g., a strong nutrition row
        if ln.get("nutrition_score", 0.0) > 0.65 or classify_nutrition_row(ln["text"])["is_strong_nutrition_row"]:
            stop_reason = "nutrition_row_detected"
            rejected.append({"text": ln["text"], "reason": stop_reason})
            break

        # Check stop word headings
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

    bbox = union_rect([ln["rect"] for ln in collected])
    debug_info = {
        "stop_reason": stop_reason,
        "rejected_lines": rejected,
        "num_lines_collected": len(collected),
        "column_band": [band_left, band_right],
    }
    return bbox, collected, debug_info

def score_ingredient_region(collected_lines, anchor_score, stop_reason, ingredient_vocab=None):
    """
    Computes a region confidence score based on line count, semantic scores,
    vocabulary density, and clean boundaries.
    """
    if not collected_lines:
        return 0.0

    avg_ing_score = float(np.mean([ln.get("ingredient_score", 0.0) for ln in collected_lines]))
    
    line_count_bonus = min(0.30, 0.04 * max(0, len(collected_lines) - 1))
    
    vocab_bonus = 0.0
    if ingredient_vocab:
        vocab_terms = [v.lower() for v in ingredient_vocab if len(v) >= 4]
        if vocab_terms:
            hits = 0
            for ln in collected_lines:
                norm = normalize_ocr_text(ln["text"])
                if any(term in norm for term in vocab_terms):
                    hits += 1
            vocab_bonus = min(0.15, (hits / len(collected_lines)) * 0.15)

    boundary_bonus = 0.12 if stop_reason and (stop_reason.startswith("stop_word") or stop_reason == "nutrition_row_detected") else 0.04

    contamination_penalty = 0.0
    for ln in collected_lines:
        if ln.get("nutrition_score", 0.0) > 0.6:
            contamination_penalty += 0.15
        if ln.get("other_score", 0.0) > 0.7:
            contamination_penalty += 0.1

    mean_ocr_confidence = float(np.mean([ln.get("confidence", 0.0) for ln in collected_lines]))
    ocr_confidence_bonus = mean_ocr_confidence * 0.10

    confidence = (
        (anchor_score / 100.0) * 0.4
        + avg_ing_score * 0.25
        + line_count_bonus
        + vocab_bonus
        + boundary_bonus
        + ocr_confidence_bonus
        - contamination_penalty
    )
    return round(float(max(0.0, min(0.98, confidence))), 3)

def _vocabulary_fallback_candidate(blocks, ingredient_vocab):
    """
    Fallback when no heading anchor is found: returns the best scoring INGREDIENT_BLOCK.
    """
    ingredient_blocks = [b for b in blocks if b["type"] == "INGREDIENT_BLOCK"]
    if not ingredient_blocks:
        return None

    # Score blocks by density of ingredients and comma count
    scored = []
    for b in ingredient_blocks:
        lines = b["lines"]
        avg_ing = float(np.mean([ln.get("ingredient_score", 0.0) for ln in lines]))
        confidence = min(0.7, 0.3 + 0.4 * avg_ing)

        scored.append({
            "bbox": b["rect"],
            "confidence": round(float(confidence), 3),
            "anchor": None,
            "matched_items": lines,
            "method": "vocabulary_fallback",
            "debug": {"num_lines_collected": len(lines), "stop_reason": "block_cluster"}
        })

    scored.sort(key=lambda x: x["confidence"], reverse=True)
    return scored[0] if scored else None

def detect_ingredient_region(items, image_shape, ingredient_vocab=None, debug=False):
    """
    Main entry point for Ingredients Region Detection.
    """
    include_debug = debug or config.DEBUG_REGION_DETECTION

    # 1. Reconstruct logical lines
    lines = reconstruct_lines(items, image_shape)
    if not lines:
        return {
            "bbox": None, "confidence": 0.0, "anchor": None,
            "matched_items": [], "method": "none",
        }

    lines = split_merged_lines(lines, image_shape[1])

    # 2. Score lines semantically
    lines = classify_lines(lines, ingredient_vocab)

    # 3. Detect logical blocks
    blocks = detect_logical_blocks(lines, image_shape, ingredient_vocab)

    # 4. Heading-driven candidates
    anchor_candidates = find_ingredient_anchor_candidates(lines)
    scored_candidates = []

    from detection.geometry import cluster_into_columns, find_column_for_line
    h, w = image_shape[:2]
    line_h = median_line_height([ln["rect"] for ln in lines]) or 20.0
    adaptive_ratio = min(config.COLUMN_GAP_MIN_WIDTH_RATIO, (line_h * 0.8) / w)

    for cand in anchor_candidates:
        anchor_line = cand["line"]
        anchor_text = cand["matched_anchor"]
        anchor_score = cand["score"]

        # Filter window lines near the anchor to avoid global header/footer connection
        y_min = anchor_line["rect"][1] - line_h * 0.5
        y_max = anchor_line["rect"][1] + line_h * 6.0
        window_lines = [ln for ln in lines if y_min <= ln["rect"][1] <= y_max]

        columns = cluster_into_columns(window_lines, w, min_gap_ratio=adaptive_ratio)
        anchor_window_col = find_column_for_line(anchor_line, columns)

        if anchor_window_col:
            col_left = min(ln["rect"][0] for ln in anchor_window_col)
            col_right = max(ln["rect"][2] for ln in anchor_window_col)
            
            # Filter all candidate lines on the page to those matching the column's horizontal span
            anchor_column_lines = []
            for ln in lines:
                cx = (ln["rect"][0] + ln["rect"][2]) / 2.0
                if col_left - 10 <= cx <= col_right + 10:
                    anchor_column_lines.append(ln)
        else:
            anchor_column_lines = lines

        bbox, collected, debug_info = expand_ingredient_region(
            anchor_line, lines, image_shape, anchor_column_lines=anchor_column_lines
        )
        confidence = score_ingredient_region(
            collected, anchor_score, debug_info["stop_reason"], ingredient_vocab
        )
        debug_info["anchor_text"] = anchor_line["text"]
        debug_info["anchor_score"] = anchor_score

        scored_candidates.append({
            "bbox": bbox,
            "confidence": confidence,
            "anchor": anchor_text,
            "matched_items": collected,
            "method": "anchor_expansion",
            "debug": debug_info,
        })

    # 5. Block/Vocabulary fallback candidate
    vocab_candidate = _vocabulary_fallback_candidate(blocks, ingredient_vocab)
    if vocab_candidate is not None:
        scored_candidates.append(vocab_candidate)

    if not scored_candidates:
        return {
            "bbox": None,
            "confidence": 0.0,
            "anchor": None,
            "matched_items": [],
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
