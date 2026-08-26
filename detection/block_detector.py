"""
detection/block_detector.py

Groups logical text lines into spatially and semantically coherent blocks.
It clusters adjacent lines based on proximity, alignment, and height,
and classifies each cluster into one of the key semantic block types.
"""

import numpy as np
import config
from detection.geometry import union_rect, vertical_distance, horizontal_overlap_ratio
from detection.ocr_detector import normalize_ocr_text

def detect_logical_blocks(lines, image_shape, ingredient_vocab=None):
    """
    Groups lines into semantic blocks.
    
    Args:
        lines: list of line dicts reconstructed by line_builder
        image_shape: tuple of (height, width) or (height, width, channels)
        ingredient_vocab: list of known ingredient name strings

    Returns:
        list of block dicts:
            {
                "type": "INGREDIENT_BLOCK" | "NUTRITION_TABLE_BLOCK" | ...
                "lines": [...],
                "rect": [x1,y1,x2,y2],
                "text": "..."
            }
    """
    if not lines:
        return []

    # Calculate median line height
    heights = [ln["height"] for ln in lines if ln["height"] > 0]
    median_h = float(np.median(heights)) if heights else 20.0

    n = len(lines)
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

    # Cluster lines that are geometrically close and aligned
    max_gap = median_h * config.BLOCK_DETECTOR_GAP_FACTOR
    band_tolerance = median_h * 3.0

    for i in range(n):
        rect_i = lines[i]["rect"]
        for j in range(i + 1, n):
            rect_j = lines[j]["rect"]

            # Check vertical gap
            v_gap = vertical_distance(rect_i, rect_j)
            if v_gap > max_gap:
                continue

            # Check horizontal alignment/overlap
            left_close = abs(rect_i[0] - rect_j[0]) <= band_tolerance
            h_overlap = horizontal_overlap_ratio(rect_i, rect_j) >= config.BLOCK_DETECTOR_MIN_OVERLAP_RATIO

            if left_close or h_overlap:
                union(i, j)

    # Group lines by parent
    groups = {}
    for i in range(n):
        root = find(i)
        groups.setdefault(root, []).append(lines[i])

    blocks = []
    for members in groups.values():
        # Sort lines within the block in reading order
        members_sorted = sorted(members, key=lambda ln: (ln["rect"][1], ln["rect"][0]))
        combined_text = " ".join(ln["text"] for ln in members_sorted)
        merged_rect = union_rect([ln["rect"] for ln in members_sorted])

        block_type = _classify_block_type(combined_text, members_sorted, ingredient_vocab)

        blocks.append({
            "type": block_type,
            "lines": members_sorted,
            "rect": merged_rect,
            "text": combined_text
        })

    # Sort blocks top-to-bottom
    blocks.sort(key=lambda b: (b["rect"][1], b["rect"][0]))
    return blocks

def _classify_block_type(combined_text, block_lines, ingredient_vocab=None):
    """
    Classifies a text block into a semantic block type based on keyword cues and line shapes.
    """
    norm = normalize_ocr_text(combined_text)

    # 1. Nutrition Table Check
    nut_keywords_found = [kw for kw in config.NUTRIENT_KEYWORDS if kw in norm]
    import re
    has_nut_vals = bool(re.search(config.NUTRITION_ROW_UNIT_PATTERN, norm, re.IGNORECASE))
    has_nut_headers = any(h in norm for h in config.NUTRITION_ANCHORS)
    
    if (has_nut_headers and len(nut_keywords_found) >= 1) or (len(nut_keywords_found) >= 3 and has_nut_vals):
        return "NUTRITION_TABLE_BLOCK"

    # 2. Ingredients Block Check
    has_ing_headers = any(h in norm for h in config.ALL_INGREDIENT_ANCHORS if h != "contains")
    comma_count = combined_text.count(",")
    
    vocab_hits = 0
    if ingredient_vocab:
        vocab_lower = [v.lower() for v in ingredient_vocab if len(v) >= 4]
        for v in vocab_lower[:300]: # check top 300 to keep it fast
            if v in norm:
                vocab_hits += 1

    if has_ing_headers or comma_count >= 5 or vocab_hits >= 5:
        return "INGREDIENT_BLOCK"

    # 3. Allergen Advice Check
    allergen_kws = ["allergen", "allergy", "contains:", "contains wheat", "contains milk", "contains soy", "may contain"]
    if any(ak in norm for ak in allergen_kws) or ("contains" in norm and any(ac in norm for ac in config.ALLERGEN_CONTEXT_WORDS)):
        return "ALLERGEN_BLOCK"

    # 4. Storage & Expiry Check
    storage_kws = ["storage", "store in", "keep in", "cool and dry", "best before", "expiry", "exp date", "use by", "batch"]
    if any(sk in norm for sk in storage_kws):
        return "STORAGE_BLOCK"

    # 5. Manufacturer Check
    mfg_kws = ["manufactured by", "mfg by", "marketed by", "pepsico", "hindustan unilever", "nestle", "britannia", "itc limited", "address:"]
    if any(mk in norm for mk in mfg_kws):
        return "MANUFACTURER_BLOCK"

    # 6. Instructions Check
    inst_kws = ["directions", "instructions", "how to prepare", "cooking time", "prep method"]
    if any(ik in norm for ik in inst_kws):
        return "INSTRUCTION_BLOCK"

    # 7. Contact Details / Feedback Check
    contact_kws = ["customer care", "contact us", "write to", "feedback", "queries", "toll free", "toll-free", "email id", "email:"]
    if any(ck in norm for ck in contact_kws):
        return "CONTACT_BLOCK"

    # 8. MRP Check
    mrp_kws = ["mrp", "max retail price", "retail price", "rs."]
    if any(mk in norm for mk in mrp_kws):
        return "MRP_BLOCK"

    return "OTHER_BLOCK"
