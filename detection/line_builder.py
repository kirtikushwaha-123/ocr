"""
detection/line_builder.py

Robust OCR line reconstruction. Groups individual word/fragment polygon items
into logical text lines using vertical center similarity, horizontal distance,
height similarity, and reading order. All thresholds scale with the median
OCR text height.
"""

import numpy as np
import config
from detection.geometry import poly_to_rect, union_rect, vertical_overlap_ratio, horizontal_distance

def reconstruct_lines(ocr_items, image_shape):
    """
    Groups individual OCR items (words/fragments) into logical text lines.
    
    Args:
        ocr_items: list of canonical OCR item dicts:
            {
                "text": "...",
                "confidence": 0.95,
                "polygon": [[x1,y1], [x2,y2], [x3,y3], [x4,y4]],
                "rect": [x1,y1,x2,y2],
                "center": [cx,cy],
                "width": ...,
                "height": ...
            }
        image_shape: tuple of (height, width, channels) or (height, width)

    Returns:
        list of line dicts sorted in reading order:
            {
                "text": "...",
                "items": [...],
                "polygon": [[x1,y1], [x2,y2], [x3,y3], [x4,y4]],
                "rect": [x1,y1,x2,y2],
                "center": [cx,cy],
                "height": ...,
                "width": ...,
                "mean_confidence": ...,
                "line_index": int
            }
    """
    if not ocr_items:
        return []

    # Calculate median height of OCR items to scale thresholds
    heights = [it["height"] for it in ocr_items if it["height"] > 0]
    median_h = float(np.median(heights)) if heights else 15.0

    # Config scaling thresholds
    v_tolerance = median_h * config.LINE_BUILDER_V_TOLERANCE_FACTOR
    max_h_gap = median_h * config.LINE_BUILDER_H_GAP_FACTOR
    h_tol = config.LINE_BUILDER_HEIGHT_TOLERANCE_FACTOR

    n = len(ocr_items)
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

    # Compare pairs of items to see if they belong to the same visual line
    for i in range(n):
        it_i = ocr_items[i]
        rect_i = it_i["rect"]
        h_i = it_i["height"]
        cy_i = it_i["center"][1]

        for j in range(i + 1, n):
            it_j = ocr_items[j]
            rect_j = it_j["rect"]
            h_j = it_j["height"]
            cy_j = it_j["center"][1]

            # 1. Vertical center similarity
            v_dist = abs(cy_i - cy_j)
            if v_dist > v_tolerance:
                # Also check vertical overlap ratio as fallback
                if vertical_overlap_ratio(rect_i, rect_j) < config.LINE_GROUP_Y_OVERLAP_THRESHOLD:
                    continue

            # 2. Height similarity
            height_diff = abs(h_i - h_j) / max(h_i, h_j, 1e-6)
            if height_diff > h_tol:
                continue

            # 3. Horizontal distance
            h_dist = horizontal_distance(rect_i, rect_j)
            if h_dist > max_h_gap:
                continue

            # If all checks pass, they are on the same line
            union(i, j)

    # Group items by parent
    groups = {}
    for i in range(n):
        root = find(i)
        groups.setdefault(root, []).append(ocr_items[i])

    reconstructed = []
    for members in groups.values():
        # Sort items left-to-right within the line
        sorted_members = sorted(members, key=lambda it: it["rect"][0])
        
        merged_text = " ".join(m["text"] for m in sorted_members)
        merged_rect = union_rect([m["rect"] for m in sorted_members])
        
        x1, y1, x2, y2 = merged_rect
        center = [float((x1 + x2) / 2.0), float((y1 + y2) / 2.0)]
        width = float(x2 - x1)
        height = float(y2 - y1)
        mean_conf = float(np.mean([m.get("confidence", 0.0) for m in sorted_members]))

        # Reconstruct polygon path through all word polygons
        merged_polygon = []
        # Upper boundary: collect top-left and top-right corners of sorted words
        for m in sorted_members:
            poly = m["polygon"]
            if len(poly) >= 4:
                merged_polygon.append(poly[0]) # Top-Left
                merged_polygon.append(poly[1]) # Top-Right
        
        # Lower boundary: collect bottom-right and bottom-left in reverse order
        for m in reversed(sorted_members):
            poly = m["polygon"]
            if len(poly) >= 4:
                merged_polygon.append(poly[2]) # Bottom-Right
                merged_polygon.append(poly[3]) # Bottom-Left

        # Fallback to simple rectangle corners if polygon construction failed
        if not merged_polygon:
            merged_polygon = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]

        reconstructed.append({
            "text": merged_text,
            "items": sorted_members,
            "polygon": merged_polygon,
            "rect": merged_rect,
            "center": center,
            "height": height,
            "width": width,
            "mean_confidence": mean_conf
        })

    # Sort lines in top-to-bottom reading order
    # (using a band to avoid minor vertical alignment jitter)
    line_h = float(np.median([ln["height"] for ln in reconstructed])) if reconstructed else 20.0
    y_band = max(line_h * 0.5, 5.0)
    
    reconstructed.sort(key=lambda ln: (round(ln["rect"][1] / y_band), ln["rect"][0]))

    # Assign line index
    for idx, ln in enumerate(reconstructed):
        ln["line_index"] = idx

    return reconstructed
