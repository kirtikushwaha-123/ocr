"""
detection/document_layout.py

Central shared document-analysis engine for PicWise.
Analyzes the page once, detecting columns, logical blocks, semantic clusters,
and section candidates.
"""

import numpy as np
import re
import config
from detection.geometry import (
    union_rect,
    vertical_distance,
    horizontal_overlap_ratio,
    median_line_height,
    sort_reading_order,
    cluster_into_columns,
    find_column_for_line,
)
from detection.line_builder import reconstruct_lines
from detection.line_classifier import classify_lines
from detection.block_detector import detect_logical_blocks
from detection.ocr_detector import normalize_ocr_text, best_anchor_match
from detection.section_signals import classify_nutrition_row, detect_section_boundaries

def analyze_document(ocr_items, image_shape, ingredient_vocab=None):
    """
    Main entry point for unified document layout analysis.
    Performs layout analysis ONCE per image.
    """
    if not ocr_items:
        return {
            "ocr_items": [],
            "lines": [],
            "classified_lines": [],
            "columns": [],
            "blocks": [],
            "semantic_clusters": [],
            "section_candidates": []
        }

    # 1. Reconstruct logical lines
    lines = reconstruct_lines(ocr_items, image_shape)
    
    # Split merged lines if they have wide horizontal gaps (like side-by-side layout)
    from detection.ingredient_region import split_merged_lines
    lines = split_merged_lines(lines, image_shape[1])

    # 2. Classify every line into 9 classes
    classified_lines = classify_lines(lines, ingredient_vocab)

    # 3. Column Detection
    h, w = image_shape[:2]
    line_h = median_line_height([ln["rect"] for ln in lines]) or 20.0
    adaptive_ratio = min(config.COLUMN_GAP_MIN_WIDTH_RATIO, (line_h * 0.8) / w)
    
    columns = cluster_into_columns(classified_lines, w, min_gap_ratio=adaptive_ratio)
    
    # Assign column_id to each classified line
    for col_idx, col_lines in enumerate(columns):
        for ln in col_lines:
            ln["column_id"] = col_idx

    # 4. Create Semantic Text Blocks (within columns)
    blocks = []
    for col_idx, col_lines in enumerate(columns):
        col_blocks = detect_logical_blocks(col_lines, image_shape, ingredient_vocab)
        for b in col_blocks:
            b["column_id"] = col_idx
        blocks.extend(col_blocks)

    # 5. Spatial + Semantic Clustering within Columns
    semantic_clusters = []
    cluster_counter = 0
    max_gap_y = line_h * config.REGION_EXPANSION_MAX_LINE_GAP_FACTOR
    band_tolerance = line_h * config.REGION_BAND_LEFT_TOLERANCE_FACTOR

    # Target semantic classes
    classes_to_cluster = ["ingredients", "nutrition", "allergen", "manufacturer", "storage", "directions", "regulatory", "product_information"]

    for col_idx, col_lines in enumerate(columns):
        for target_class in classes_to_cluster:
            # Gather candidates for this class in this column
            candidates = [ln for ln in col_lines if ln.get("scores", {}).get(target_class, 0.0) >= 0.20]
            if not candidates:
                continue

            # Run DSU clustering based on Y gap and X overlap
            n_cand = len(candidates)
            parent = list(range(n_cand))

            def find_parent(x):
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x

            def union_parent(a, b):
                ra, rb = find_parent(a), find_parent(b)
                if ra != rb:
                    parent[ra] = rb

            for i in range(n_cand):
                rect_i = candidates[i]["rect"]
                for j in range(i + 1, n_cand):
                    rect_j = candidates[j]["rect"]

                    v_gap = vertical_distance(rect_i, rect_j)
                    if v_gap > max_gap_y:
                        continue

                    left_close = abs(rect_i[0] - rect_j[0]) <= band_tolerance
                    overlap = horizontal_overlap_ratio(rect_i, rect_j) >= config.REGION_BAND_OVERLAP_MIN_RATIO

                    if left_close or overlap:
                        union_parent(i, j)

            groups = {}
            for i in range(n_cand):
                root = find_parent(i)
                groups.setdefault(root, []).append(candidates[i])

            for g in groups.values():
                sorted_g = sorted(g, key=lambda ln: (ln["rect"][1], ln["rect"][0]))
                
                # Fill in intermediate lines within the same column band to preserve contiguity
                y_min = sorted_g[0]["rect"][1]
                y_max = sorted_g[-1]["rect"][3]
                
                filled_g = []
                for ln in col_lines:
                    cy = (ln["rect"][1] + ln["rect"][3]) / 2.0
                    if y_min - 2.0 <= cy <= y_max + 2.0:
                        filled_g.append(ln)
                        
                if not filled_g:
                    filled_g = sorted_g
                else:
                    filled_g = sorted(filled_g, key=lambda ln: (ln["rect"][1], ln["rect"][0]))
                    
                bbox = union_rect([ln["rect"] for ln in filled_g])
                
                # Calculate scores over the complete/filled cluster
                avg_sem = float(np.mean([ln.get("scores", {}).get(target_class, 0.0) for ln in filled_g]))
                
                # Check for headings nearby or inside
                heading_score = 0.0
                heading_anchors = config.ALL_INGREDIENT_ANCHORS if target_class == "ingredients" else config.NUTRITION_ANCHORS
                for ln in col_lines:
                    is_inside = any(ln["rect"] == cln["rect"] for cln in filled_g)
                    is_above = ln["rect"][1] < bbox[1] and bbox[1] - ln["rect"][3] <= max_gap_y
                    if is_inside or is_above:
                        matched, score = best_anchor_match(ln["text"], heading_anchors)
                        if matched:
                            heading_score = max(heading_score, score / 100.0)
                            
                # Spatial coherence score based on height standard deviation
                heights = [ln["height"] for ln in filled_g]
                std_h = float(np.std(heights)) if len(heights) > 1 else 0.0
                mean_h = float(np.mean(heights)) if heights else 1.0
                spatial_coherence = max(0.0, 1.0 - (std_h / max(mean_h, 1e-6)))

                cluster_id = f"cluster_{target_class}_{cluster_counter}"
                cluster_counter += 1

                semantic_clusters.append({
                    "cluster_id": cluster_id,
                    "class": target_class,
                    "lines": filled_g,
                    "bbox": bbox,
                    "semantic_score": avg_sem,
                    "spatial_coherence": spatial_coherence,
                    "column_id": col_idx,
                    "heading_evidence": heading_score
                })

    # 6. Generate Section Candidates (Ingredients & Nutrition)
    section_candidates = []
    
    # Config weights
    w_ing_sem = getattr(config, "WEIGHT_ING_SEMANTIC", 0.35)
    w_ing_vocab = getattr(config, "WEIGHT_ING_VOCAB", 0.25)
    w_ing_heading = getattr(config, "WEIGHT_ING_HEADING", 0.15)
    w_ing_spatial = getattr(config, "WEIGHT_ING_SPATIAL", 0.15)
    w_ing_ocr = getattr(config, "WEIGHT_ING_OCR", 0.10)

    w_nut_vocab = getattr(config, "WEIGHT_NUT_VOCAB", 0.35)
    w_nut_num_unit = getattr(config, "WEIGHT_NUT_NUM_UNIT", 0.20)
    w_nut_table = getattr(config, "WEIGHT_NUT_TABLE", 0.20)
    w_nut_heading = getattr(config, "WEIGHT_NUT_HEADING", 0.15)
    w_nut_spatial = getattr(config, "WEIGHT_NUT_SPATIAL", 0.10)

    # Helper: Vocabulary matching density
    vocab_terms = [v.lower() for v in ingredient_vocab if len(v) >= 4] if ingredient_vocab else []

    # Iterate over semantic clusters to build candidates
    for c in semantic_clusters:
        c_lines = c["lines"]
        if not c_lines:
            continue
            
        c_class = c["class"]
        bbox = c["bbox"]
        
        # OCR score (average confidence)
        ocr_score = float(np.mean([ln.get("confidence", 0.0) for ln in c_lines]))
        
        # Calculate vocabulary density for ingredients
        vocab_score = 0.0
        if vocab_terms:
            hits = sum(1 for ln in c_lines if any(term in normalize_ocr_text(ln["text"]) for term in vocab_terms))
            vocab_score = hits / len(c_lines)

        if c_class == "ingredients":
            if len(c_lines) < 2:
                continue
            # Contamination penalty: check if contains lines with high nutrition/other class scores
            contamination = 0.0
            for ln in c_lines:
                if ln.get("scores", {}).get("nutrition", 0.0) > 0.50:
                    contamination += 0.2
                if ln.get("scores", {}).get("other", 0.0) > 0.60:
                    contamination += 0.1

            final_score = (
                w_ing_sem * c["semantic_score"]
                + w_ing_vocab * vocab_score
                + w_ing_heading * c["heading_evidence"]
                + w_ing_spatial * c["spatial_coherence"]
                + w_ing_ocr * ocr_score
                - contamination
            )

            section_candidates.append({
                "type": "ingredients",
                "method": "semantic_cluster" if c["heading_evidence"] == 0.0 else "heading_supported_cluster",
                "bbox": bbox,
                "lines": c_lines,
                "semantic_score": c["semantic_score"],
                "vocabulary_score": vocab_score,
                "heading_score": c["heading_evidence"],
                "spatial_score": c["spatial_coherence"],
                "ocr_score": ocr_score,
                "contamination_penalty": contamination,
                "final_score": round(float(max(0.0, min(1.0, final_score))), 3)
            })

        elif c_class == "nutrition":
            # Require at least 3 lines and 2 distinct nutrient keyword hits to be a valid nutrition region candidate
            if len(c_lines) < 3:
                continue
            matched_terms = set()
            for ln in c_lines:
                matched_terms.update(classify_nutrition_row(ln["text"])["keyword_hits"])
            if len(matched_terms) < 2:
                continue
                
            # Numeric/Unit score
            num_unit_hits = sum(1 for ln in c_lines if classify_nutrition_row(ln["text"])["has_number_unit"])
            num_unit_score = num_unit_hits / len(c_lines)
            
            # Nutrient keyword score
            nut_keyword_hits = sum(1 for ln in c_lines if classify_nutrition_row(ln["text"])["keyword_hits"])
            nut_keyword_score = nut_keyword_hits / len(c_lines)

            # Table structure score: lines height consistency and block density
            rects = [ln["rect"] for ln in c_lines]
            bbox_area = max(1.0, (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
            items_area = sum((r[2] - r[0]) * (r[3] - r[1]) for r in rects)
            compactness = min(1.0, items_area / bbox_area * 3.0)
            table_structure_score = 0.5 * c["spatial_coherence"] + 0.5 * compactness

            contamination = 0.0
            for ln in c_lines:
                if ln.get("scores", {}).get("ingredients", 0.0) > 0.50:
                    contamination += 0.2

            final_score = (
                w_nut_vocab * nut_keyword_score
                + w_nut_num_unit * num_unit_score
                + w_nut_table * table_structure_score
                + w_nut_heading * c["heading_evidence"]
                + w_nut_spatial * c["spatial_coherence"]
                - contamination
            )

            section_candidates.append({
                "type": "nutrition",
                "method": "semantic_cluster" if c["heading_evidence"] == 0.0 else "heading_supported_cluster",
                "bbox": bbox,
                "lines": c_lines,
                "semantic_score": c["semantic_score"],
                "vocabulary_score": nut_keyword_score,
                "heading_score": c["heading_evidence"],
                "spatial_score": c["spatial_coherence"],
                "ocr_score": ocr_score,
                "contamination_penalty": contamination,
                "final_score": round(float(max(0.0, min(1.0, final_score))), 3)
            })

    # Also build candidates from Logical Blocks
    for b in blocks:
        b_lines = b["lines"]
        if not b_lines:
            continue
        bbox = b["rect"]
        ocr_score = float(np.mean([ln.get("confidence", 0.0) for ln in b_lines]))

        if b["type"] == "INGREDIENT_BLOCK":
            if len(b_lines) < 2:
                continue
            avg_sem = float(np.mean([ln.get("scores", {}).get("ingredients", 0.0) for ln in b_lines]))
            vocab_score = 0.0
            if vocab_terms:
                hits = sum(1 for ln in b_lines if any(term in normalize_ocr_text(ln["text"]) for term in vocab_terms))
                vocab_score = hits / len(b_lines)
            
            final_score = (
                w_ing_sem * avg_sem
                + w_ing_vocab * vocab_score
                + w_ing_heading * 0.20 # baseline block head score
                + w_ing_spatial * 0.80
                + w_ing_ocr * ocr_score
            )
            section_candidates.append({
                "type": "ingredients",
                "method": "logical_block",
                "bbox": bbox,
                "lines": b_lines,
                "semantic_score": avg_sem,
                "vocabulary_score": vocab_score,
                "heading_score": 0.20,
                "spatial_score": 0.80,
                "ocr_score": ocr_score,
                "contamination_penalty": 0.0,
                "final_score": round(float(max(0.0, min(1.0, final_score))), 3)
            })
            
        elif b["type"] == "NUTRITION_TABLE_BLOCK":
            if len(b_lines) < 3:
                continue
            matched_terms = set()
            for ln in b_lines:
                matched_terms.update(classify_nutrition_row(ln["text"])["keyword_hits"])
            if len(matched_terms) < 2:
                continue
                
            avg_sem = float(np.mean([ln.get("scores", {}).get("nutrition", 0.0) for ln in b_lines]))
            num_unit_hits = sum(1 for ln in b_lines if classify_nutrition_row(ln["text"])["has_number_unit"])
            num_unit_score = num_unit_hits / len(b_lines)
            nut_keyword_hits = sum(1 for ln in b_lines if classify_nutrition_row(ln["text"])["keyword_hits"])
            nut_keyword_score = nut_keyword_hits / len(b_lines)

            final_score = (
                w_nut_vocab * nut_keyword_score
                + w_nut_num_unit * num_unit_score
                + w_nut_table * 0.80
                + w_nut_heading * 0.20
                + w_nut_spatial * 0.80
            )
            section_candidates.append({
                "type": "nutrition",
                "method": "logical_block",
                "bbox": bbox,
                "lines": b_lines,
                "semantic_score": avg_sem,
                "vocabulary_score": nut_keyword_score,
                "heading_score": 0.20,
                "spatial_score": 0.80,
                "ocr_score": ocr_score,
                "contamination_penalty": 0.0,
                "final_score": round(float(max(0.0, min(1.0, final_score))), 3)
            })

    # Sort all candidates by final_score descending
    section_candidates.sort(key=lambda x: x["final_score"], reverse=True)

    return {
        "ocr_items": ocr_items,
        "lines": lines,
        "classified_lines": classified_lines,
        "columns": columns,
        "blocks": blocks,
        "semantic_clusters": semantic_clusters,
        "section_candidates": section_candidates
    }
