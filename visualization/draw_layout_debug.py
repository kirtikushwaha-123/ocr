"""
visualization/draw_layout_debug.py

Generates detailed layout analysis debug visualizations for PicWise.
"""

import cv2
import os
import config
from detection.geometry import union_rect

# Colors for 9 classes
CLASS_COLORS = {
    "ingredients": (0, 200, 0),         # Green
    "nutrition": (200, 100, 0),         # Blue
    "allergen": (0, 0, 255),            # Red
    "manufacturer": (200, 0, 200),      # Magenta
    "storage": (0, 200, 200),           # Yellow
    "directions": (128, 0, 128),        # Purple
    "regulatory": (0, 128, 128),        # Teal
    "product_information": (128, 128, 0), # Olive
    "other": (128, 128, 128),           # Gray
}

def _draw_box(img, rect, color, label, thickness=2):
    x1, y1, x2, y2 = [int(round(v)) for v in rect]
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    if label:
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = max(0.4, min(0.8, (x2 - x1) / 300.0))
        (tw, th), bl = cv2.getTextSize(label, font, font_scale, 1)
        ly = max(0, y1 - th - bl - 4)
        cv2.rectangle(img, (x1, ly), (x1 + tw + 6, y1), color, -1)
        cv2.putText(img, label, (x1 + 3, y1 - bl - 1), font, font_scale, (255, 255, 255), 1, cv2.LINE_AA)

def draw_layout_debug(image, layout_analysis, ingredient_result, nutrition_result, output_dir):
    """
    Generates all 8 layout visual debugging images and saves them in the output folder.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. OCR Lines
    img_lines = image.copy()
    for idx, ln in enumerate(layout_analysis.get("lines", [])):
        _draw_box(img_lines, ln["rect"], (180, 100, 50), f"L{idx}", thickness=1)
    cv2.imwrite(os.path.join(output_dir, "layout_01_lines.jpg"), img_lines)

    # 2. Detected Columns
    img_cols = image.copy()
    for col_idx, col_lines in enumerate(layout_analysis.get("columns", [])):
        if col_lines:
            col_bbox = union_rect([ln["rect"] for ln in col_lines])
            _draw_box(img_cols, col_bbox, (200, 0, 100), f"Col {col_idx}", thickness=2)
    cv2.imwrite(os.path.join(output_dir, "layout_02_columns.jpg"), img_cols)

    # 3. 9-Class Semantic Labels
    img_classes = image.copy()
    for ln in layout_analysis.get("classified_lines", []):
        pred_c = ln.get("predicted_class", "other")
        color = CLASS_COLORS.get(pred_c, (128, 128, 128))
        _draw_box(img_classes, ln["rect"], color, pred_c.upper(), thickness=1)
    cv2.imwrite(os.path.join(output_dir, "layout_03_semantic_classes.jpg"), img_classes)

    # 4. Logical Blocks
    img_blocks = image.copy()
    for idx, b in enumerate(layout_analysis.get("blocks", [])):
        b_type = b["type"]
        color = (0, 150, 200) if "ING" in b_type else (200, 150, 0)
        _draw_box(img_blocks, b["rect"], color, f"Block {idx}: {b_type}", thickness=2)
    cv2.imwrite(os.path.join(output_dir, "layout_04_logical_blocks.jpg"), img_blocks)

    # 5. Semantic Clusters
    img_clusters = image.copy()
    for c in layout_analysis.get("semantic_clusters", []):
        c_class = c["class"]
        color = CLASS_COLORS.get(c_class, (128, 128, 128))
        _draw_box(img_clusters, c["bbox"], color, f"Cluster: {c_class.upper()}", thickness=2)
    cv2.imwrite(os.path.join(output_dir, "layout_05_semantic_clusters.jpg"), img_clusters)

    # 6. All Candidate Regions
    img_cands = image.copy()
    for idx, c in enumerate(layout_analysis.get("section_candidates", [])):
        c_type = c["type"]
        color = (0, 200, 0) if c_type == "ingredients" else (200, 100, 0)
        _draw_box(img_cands, c["bbox"], color, f"{c_type.upper()} Cand {idx} ({c['final_score']})", thickness=2)
    cv2.imwrite(os.path.join(output_dir, "layout_06_candidates.jpg"), img_cands)

    # 7. Final Selected ROI - Ingredients
    img_ing = image.copy()
    ing_box = ingredient_result.get("bbox")
    if ing_box:
        _draw_box(img_ing, ing_box, (0, 200, 0), f"Ingredients ({ingredient_result.get('confidence')})")
    cv2.imwrite(os.path.join(output_dir, "layout_07_final_ingredients.jpg"), img_ing)

    # 8. Final Selected ROI - Nutrition
    img_nut = image.copy()
    nut_box = nutrition_result.get("bbox")
    if nut_box:
        _draw_box(img_nut, nut_box, (200, 100, 0), f"Nutrition ({nutrition_result.get('confidence')})")
    cv2.imwrite(os.path.join(output_dir, "layout_08_final_nutrition.jpg"), img_nut)
