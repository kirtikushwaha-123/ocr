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
from detection.ocr_detector import normalize_ocr_text, find_anchor_candidates, best_anchor_match
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

def cluster_semantic_lines(lines, target_class, max_gap_y, band_tolerance):
    """
    Groups lines with high target_class probability into spatial clusters.
    """
    candidates = [ln for ln in lines if ln.get("scores", {}).get(target_class, 0.0) >= 0.20]
    if not candidates:
        return []
        
    n = len(candidates)
    parent = list(range(n))
    
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
        
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
            
    for i in range(n):
        rect_i = candidates[i]["rect"]
        for j in range(i + 1, n):
            rect_j = candidates[j]["rect"]
            
            # Check vertical distance
            v_gap = vertical_distance(rect_i, rect_j)
            if v_gap > max_gap_y:
                continue
                
            # Check horizontal alignment
            left_close = abs(rect_i[0] - rect_j[0]) <= band_tolerance
            overlap = horizontal_overlap_ratio(rect_i, rect_j) >= config.REGION_BAND_OVERLAP_MIN_RATIO
            
            if left_close or overlap:
                union(i, j)
                
    groups = {}
    for i in range(n):
        root = find(i)
        groups.setdefault(root, []).append(candidates[i])
        
    clusters = []
    for g in groups.values():
        clusters.append(sorted(g, key=lambda ln: (ln["rect"][1], ln["rect"][0])))
    return clusters


def _spatial_semantic_fallback_candidate(lines, max_gap_y, band_tolerance, image_shape):
    """
    Fallback using spatial-semantic clustering when no anchor is found.
    """
    clusters = cluster_semantic_lines(lines, "ingredients", max_gap_y, band_tolerance)
    if not clusters:
        return None
        
    scored = []
    for c_lines in clusters:
        if len(c_lines) < 2:
            continue
        avg_ing = float(np.mean([ln.get("scores", {}).get("ingredients", 0.0) for ln in c_lines]))
        if avg_ing < 0.35:
            continue
            
        confidence = min(0.75, 0.3 + 0.45 * avg_ing)
        if confidence < 0.55:
            continue
            
        bbox = union_rect([ln["rect"] for ln in c_lines])
        
        scored.append({
            "bbox": bbox,
            "confidence": round(float(confidence), 3),
            "anchor": None,
            "matched_items": c_lines,
            "method": "spatial_semantic_fallback",
            "debug": {"num_lines_collected": len(c_lines), "stop_reason": "spatial_clustering"}
        })
        
    scored.sort(key=lambda x: x["confidence"], reverse=True)
    return scored[0] if scored else None


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

def detect_ingredient_region(layout_analysis, image_shape, ingredient_vocab=None, debug=False):
    """
    Main entry point for Ingredients Region Detection. Supports consuming unified document layout analysis.
    """
    # 1. Backward-compatibility / Direct call support
    if not isinstance(layout_analysis, dict) or "section_candidates" not in layout_analysis:
        from detection.document_layout import analyze_document
        layout_analysis = analyze_document(layout_analysis, image_shape, ingredient_vocab)

    include_debug = debug or config.DEBUG_REGION_DETECTION

    # Find candidates from layout_analysis candidates
    candidates = [c for c in layout_analysis["section_candidates"] if c["type"] == "ingredients"]
    
    # Target high-quality primary candidates first (semantic/logical blocks/clusters)
    primary_cands = [c for c in candidates if c["final_score"] >= 0.70]
    
    if primary_cands:
        best_cand = primary_cands[0]
        # Construct winning response
        result = {
            "bbox": best_cand["bbox"],
            "confidence": best_cand["final_score"],
            "anchor": None,
            "matched_items": best_cand["lines"],
            "method": best_cand["method"],
            "debug": {
                "final_score": best_cand["final_score"],
                "semantic_score": best_cand["semantic_score"],
                "vocabulary_score": best_cand["vocabulary_score"],
                "heading_score": best_cand["heading_score"],
                "spatial_score": best_cand["spatial_score"],
                "ocr_score": best_cand["ocr_score"],
                "all_candidate_scores": [
                    {"method": c["method"], "confidence": c["final_score"]} for c in candidates
                ]
            }
        }
        # Resolve anchor text if heading matched
        if best_cand["heading_score"] > 0.0:
            for ln in best_cand["lines"]:
                matched, _ = best_anchor_match(ln["text"], config.ALL_INGREDIENT_ANCHORS)
                if matched:
                    result["anchor"] = ln["text"]
                    break

        best_bbox = validate_region(result["bbox"], image_shape)
        result["bbox"] = best_bbox
        
        if not include_debug:
            result.pop("debug", None)
            
        return result

    # 2. Legacy fallback if no strong layout candidates found
    lines = layout_analysis["lines"]
    if not lines:
        return {
            "bbox": None, "confidence": 0.0, "anchor": None,
            "matched_items": [], "method": "none",
        }

    blocks = layout_analysis["blocks"]
    anchor_candidates = find_ingredient_anchor_candidates(lines)
    scored_candidates = []

    h, w = image_shape[:2]
    line_h = median_line_height([ln["rect"] for ln in lines]) or 20.0
    max_gap_y = line_h * config.REGION_EXPANSION_MAX_LINE_GAP_FACTOR
    band_tolerance = line_h * config.REGION_BAND_LEFT_TOLERANCE_FACTOR
    adaptive_ratio = min(config.COLUMN_GAP_MIN_WIDTH_RATIO, (line_h * 0.8) / w)

    for cand in anchor_candidates:
        anchor_line = cand["line"]
        anchor_text = cand["matched_anchor"]
        anchor_score = cand["score"]

        # Filter window lines near the anchor
        y_min = anchor_line["rect"][1] - line_h * 0.5
        y_max = anchor_line["rect"][1] + line_h * 6.0
        window_lines = [ln for ln in lines if y_min <= ln["rect"][1] <= y_max]

        from detection.geometry import cluster_into_columns, find_column_for_line
        columns = cluster_into_columns(window_lines, w, min_gap_ratio=adaptive_ratio)
        anchor_window_col = find_column_for_line(anchor_line, columns)

        if anchor_window_col:
            col_left = min(ln["rect"][0] for ln in anchor_window_col)
            col_right = max(ln["rect"][2] for ln in anchor_window_col)
            
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

    # Add other layout candidates with lower confidence
    for c in candidates:
        scored_candidates.append({
            "bbox": c["bbox"],
            "confidence": c["final_score"],
            "anchor": None,
            "matched_items": c["lines"],
            "method": c["method"],
            "debug": {"final_score": c["final_score"]}
        })

    if not scored_candidates:
        return {
            "bbox": None, "confidence": 0.0, "anchor": None,
            "matched_items": [], "method": "none",
        }

    scored_candidates.sort(key=lambda c: c["confidence"], reverse=True)
    best = scored_candidates[0]
    best_bbox = validate_region(best["bbox"], image_shape)
    best["bbox"] = best_bbox

    if include_debug:
        best["debug"] = best.get("debug", {})
        best["debug"]["all_candidate_scores"] = [
            {"method": c["method"], "anchor": c["anchor"], "confidence": c["confidence"]}
            for c in scored_candidates
        ]
    else:
        best.pop("debug", None)

    return best
