"""
detection/tight_roi.py

Refines raw region candidates by applying semantic line filtering, boundary
detection, and calculating the tight union of accepted lines. Applies a small
padding relative to font size (median line height) and clips to image boundaries
to produce the final crop box.
"""

from detection.geometry import union_rect, validate_region, clip_rect, median_line_height

def _refine_roi(lines, image_shape, min_score_key=None, min_score_val=0.1):
    """
    Core ROI refinement utility.
    """
    if not lines:
        return None

    # 1. Semantic line filtering (secondary check)
    filtered_lines = []
    for ln in lines:
        if min_score_key and ln.get(min_score_key, 0.0) < min_score_val:
            continue
        filtered_lines.append(ln)

    # Fallback to unfiltered lines if filtering leaves nothing
    if not filtered_lines:
        filtered_lines = lines

    # 2. Polygon/rect union of accepted lines
    rects = [ln["rect"] for ln in filtered_lines]
    union_box = union_rect(rects)
    if not union_box:
        return None

    # 3. Apply margin based on median line height (font size)
    # Derive line height from all candidate lines
    line_rects = [ln["rect"] for ln in lines]
    line_h = median_line_height(line_rects)

    # pad_x = 0.5 * median_line_height, pad_y = 0.3 * median_line_height
    pad_x = int(round(0.5 * line_h))
    pad_y = int(round(0.3 * line_h))

    x1, y1, x2, y2 = union_box
    x1 -= pad_x
    y1 -= pad_y
    x2 += pad_x
    y2 += pad_y

    # 4. Image clipping & validation
    final_bbox = validate_region([x1, y1, x2, y2], image_shape)
    return final_bbox

def refine_ingredient_roi(lines, image_shape):
    """
    Computes a tight Ingredients ROI crop box.
    """
    return _refine_roi(
        lines,
        image_shape,
        min_score_key="ingredient_score",
        min_score_val=0.08
    )

def refine_nutrition_roi(lines, image_shape):
    """
    Computes a tight Nutrition Facts ROI crop box.
    """
    return _refine_roi(
        lines,
        image_shape,
        min_score_key="nutrition_score",
        min_score_val=0.08
    )
