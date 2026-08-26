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

def expand_ingredient_region(anchor_line, all_lines, image_shape):
    """
    Expands the region from the anchor line, accepting lines based on geometry and semantics.
    """
    anchor_rect = anchor_line["rect"]
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

    confidence = (
        (anchor_score / 100.0) * 0.4
        + avg_ing_score * 0.25
        + line_count_bonus
        + vocab_bonus
        + boundary_bonus
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

    # 2. Score lines semantically
    lines = classify_lines(lines, ingredient_vocab)

    # 3. Detect logical blocks
    blocks = detect_logical_blocks(lines, image_shape, ingredient_vocab)

    # 4. Heading-driven candidates
    anchor_candidates = find_ingredient_anchor_candidates(lines)
    scored_candidates = []

    for cand in anchor_candidates:
        anchor_line = cand["line"]
        anchor_text = cand["matched_anchor"]
        anchor_score = cand["score"]

        bbox, collected, debug_info = expand_ingredient_region(anchor_line, lines, image_shape)
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
