"""
detection/tight_roi.py

Refines raw region candidates by applying semantic line filtering, boundary
detection, and calculating the tight union of accepted lines. Applies a small
padding relative to font size (median line height) and clips to image boundaries
to produce the final crop box.
"""

from detection.geometry import union_rect, validate_region, clip_rect, median_line_height

def _refine_roi_v2(candidate, image_shape, min_score_key, min_score_val=0.08):
    orig_bbox = candidate.get("bbox")
    lines = candidate.get("matched_items", []) or candidate.get("lines", [])
    
    if not orig_bbox:
        return {
            "original_candidate_bbox": None,
            "refined_bbox": None,
            "refinement_method": "none",
            "changed": False
        }

    if not lines:
        return {
            "original_candidate_bbox": orig_bbox,
            "refined_bbox": orig_bbox,
            "refinement_method": "keep_original",
            "changed": False
        }

    # Identify lines belonging to that candidate that meet semantic threshold
    filtered_lines = [ln for ln in lines if ln.get(min_score_key, 0.0) >= min_score_val]
    if not filtered_lines:
        filtered_lines = lines

    # Compute union bbox of filtered lines
    union_box = union_rect([ln["rect"] for ln in filtered_lines])
    if not union_box:
        return {
            "original_candidate_bbox": orig_bbox,
            "refined_bbox": orig_bbox,
            "refinement_method": "keep_original",
            "changed": False
        }

    # Adjust boundaries conservatively (don't expand beyond the original candidate bbox in X-direction to preserve column boundaries)
    x1, y1, x2, y2 = union_box
    ox1, oy1, ox2, oy2 = orig_bbox

    rx1 = max(x1, ox1)
    rx2 = min(x2, ox2)
    ry1 = y1
    ry2 = y2

    # Add a small padding (relative to line height)
    line_h = median_line_height([ln["rect"] for ln in lines]) or 20.0
    pad_x = int(round(0.3 * line_h))
    pad_y = int(round(0.2 * line_h))

    rx1 = max(0, rx1 - pad_x)
    rx2 = min(image_shape[1], rx2 + pad_x)
    ry1 = max(0, ry1 - pad_y)
    ry2 = min(image_shape[0], ry2 + pad_y)

    refined_bbox = [float(rx1), float(ry1), float(rx2), float(ry2)]
    changed = (refined_bbox != orig_bbox)

    return {
        "original_candidate_bbox": orig_bbox,
        "refined_bbox": refined_bbox,
        "refinement_method": "conservative_column_aligned",
        "changed": changed
    }

def refine_ingredient_roi(candidate, image_shape):
    """
    Computes a tight Ingredients ROI crop box.
    """
    if isinstance(candidate, list):
        bbox = union_rect([ln["rect"] for ln in candidate]) if candidate else None
        candidate_dict = {"bbox": bbox, "matched_items": candidate}
    else:
        candidate_dict = candidate

    return _refine_roi_v2(candidate_dict, image_shape, min_score_key="ingredient_score")

def refine_nutrition_roi(candidate, image_shape):
    """
    Computes a tight Nutrition Facts ROI crop box.
    """
    if isinstance(candidate, list):
        bbox = union_rect([ln["rect"] for ln in candidate]) if candidate else None
        candidate_dict = {"bbox": bbox, "matched_items": candidate}
    else:
        candidate_dict = candidate

    return _refine_roi_v2(candidate_dict, image_shape, min_score_key="nutrition_score")
