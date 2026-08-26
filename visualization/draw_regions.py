"""
visualization/draw_regions.py

Draws the final debugging visualization:
  GREEN  = Ingredients / Composition region
  BLUE   = Nutrition Information region
  RED    = other detected OCR text boxes (not part of either region)

Note: "other text" is now determined by bbox containment against the two
detected region boxes rather than object identity, since region
detectors operate on merged "lines" internally (see detection/geometry.py
group_ocr_boxes_into_lines) while `all_ocr_items` here are the original
raw per-box OCR items - containment is a representation-independent way
to classify them for display.
"""

import cv2

import config


def _rect_center_inside(rect, box):
    if box is None:
        return False
    cx = (rect[0] + rect[2]) / 2.0
    cy = (rect[1] + rect[3]) / 2.0
    x1, y1, x2, y2 = box
    return x1 <= cx <= x2 and y1 <= cy <= y2


def _draw_labeled_box(image, rect, color, label, thickness=3):
    x1, y1, x2, y2 = [int(round(v)) for v in rect]
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

    if label:
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = max(0.5, min(1.1, (x2 - x1) / 400.0))
        (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, 2)
        label_y1 = max(0, y1 - text_h - baseline - 6)
        cv2.rectangle(image, (x1, label_y1), (x1 + text_w + 8, y1), color, -1)
        cv2.putText(
            image,
            label,
            (x1 + 4, y1 - baseline - 2),
            font,
            font_scale,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )


def draw_regions(
    image,
    ingredient_result=None,
    nutrition_result=None,
    all_ocr_items=None,
    draw_other_text=True,
    draw_debug_rejected=False,
):
    """
    Returns a new BGR image (copy of `image`) with bounding boxes drawn.

    draw_debug_rejected: if True and a region result carries a "debug"
    dict (see detection/ingredient_region.py, detection/nutrition_region.py),
    draws rejected candidate lines as thin dashed gray boxes - useful for
    diagnosing why expansion stopped where it did.
    """
    vis = image.copy()

    ingredient_box = ingredient_result.get("bbox") if ingredient_result else None
    nutrition_box = nutrition_result.get("bbox") if nutrition_result else None

    if draw_other_text and all_ocr_items:
        for it in all_ocr_items:
            rect = it["rect"]
            if _rect_center_inside(rect, ingredient_box) or _rect_center_inside(rect, nutrition_box):
                continue
            _draw_labeled_box(vis, rect, config.COLOR_OTHER_TEXT, None, thickness=1)

    if ingredient_box:
        conf = ingredient_result.get("confidence", 0.0)
        _draw_labeled_box(vis, ingredient_box, config.COLOR_INGREDIENTS, f"INGREDIENTS ({conf:.2f})")

    if nutrition_box:
        conf = nutrition_result.get("confidence", 0.0)
        _draw_labeled_box(vis, nutrition_box, config.COLOR_NUTRITION, f"NUTRITION ({conf:.2f})")

    if draw_debug_rejected:
        gray = (150, 150, 150)
        for result in (ingredient_result, nutrition_result):
            if not result or "debug" not in result:
                continue
            debug_info = result["debug"]
            # rejected lines only carry text, not rects, by default - this
            # hook is a no-op unless callers extend debug_info with rects.
            for rej in debug_info.get("rejected_lines", []):
                if "rect" in rej:
                    _draw_labeled_box(vis, rej["rect"], gray, None, thickness=1)

    return vis
