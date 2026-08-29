"""
detection/line_classifier.py

Classifies logical text lines semantically by computing three distinct scores:
- ingredient_score
- nutrition_score
- other_section_score

Uses a rich set of keyword cues, structural regex patterns (commas, units, parentheses,
INS numbers), and vocabulary matching.
"""

import re
import config
from detection.ocr_detector import normalize_ocr_text, best_anchor_match

# Regex patterns
_INS_NUMBER_RE = re.compile(r'\b(?:ins|e)\s*\d+[a-z]?(\([i|v|x]+\))?\b', re.IGNORECASE)
_PERCENTAGE_RE = re.compile(r'\b\d+(?:\.\d+)?\s*(?:%|percent)\b', re.IGNORECASE)
_PARENTHESIS_RE = re.compile(r'\(.*?\)')
_NUTRITION_UNIT_RE = re.compile(config.NUTRITION_ROW_UNIT_PATTERN, re.IGNORECASE)
_NUMERIC_RE = re.compile(r'\b\d+(?:\.\d+)?\b')

def classify_lines(lines, ingredient_vocab=None):
    """
    Computes semantic classification scores for each logical line.
    
    Args:
        lines: list of logical line dicts (from line_builder)
        ingredient_vocab: list/iterable of known ingredient name strings

    Returns:
        list of logical line dicts, each augmented with:
            {
                "ingredient_score": float (0.0 to 1.0),
                "nutrition_score": float (0.0 to 1.0),
                "other_section_score": float (0.0 to 1.0)
            }
    """
    if not lines:
        return []

    vocab_lower = [v.lower() for v in ingredient_vocab if len(v) >= 4] if ingredient_vocab else []

    classified_lines = []
    for ln in lines:
        text = ln["text"]
        norm = normalize_ocr_text(text)

        # -------------------------------------------------------------
        # 1. INGREDIENT SCORE COMPUTATION
        # -------------------------------------------------------------
        ing_score = 0.0

        # Anchor heading match
        matched_ing_anchor, ing_anchor_score = best_anchor_match(
            text, config.ALL_INGREDIENT_ANCHORS, threshold=config.FUZZY_ANCHOR_THRESHOLD
        )
        if matched_ing_anchor:
            ing_score += 0.85 * (ing_anchor_score / 100.0)

        # Comma separated items
        comma_count = text.count(",")
        if comma_count >= 2:
            ing_score += min(0.4, 0.15 * comma_count)
        
        # Parentheses
        if _PARENTHESIS_RE.search(text):
            ing_score += 0.15

        # Percentages
        if _PERCENTAGE_RE.search(norm):
            ing_score += 0.2

        # INS / E-numbers
        if _INS_NUMBER_RE.search(norm):
            ing_score += 0.3

        # Ingredient vocabulary density
        if vocab_lower:
            vocab_hits = sum(1 for v in vocab_lower[:400] if v in norm) # limit check size for performance
            if vocab_hits > 0:
                ing_score += min(0.35, 0.1 * vocab_hits)

        ing_score = min(1.0, max(0.0, ing_score))

        # -------------------------------------------------------------
        # 2. NUTRITION SCORE COMPUTATION
        # -------------------------------------------------------------
        nut_score = 0.0

        # Anchor heading match
        matched_nut_anchor, nut_anchor_score = best_anchor_match(
            text, config.NUTRITION_ANCHORS, threshold=config.FUZZY_ANCHOR_THRESHOLD
        )
        if matched_nut_anchor:
            nut_score += 0.9 * (nut_anchor_score / 100.0)

        # Nutrient keywords
        nut_hits = sum(1 for kw in config.NUTRIENT_KEYWORDS if kw in norm)
        if nut_hits > 0:
            nut_score += min(0.45, 0.15 * nut_hits)

        # Number + Unit pattern
        if _NUTRITION_UNIT_RE.search(norm):
            nut_score += 0.35

        # Serving information / headers
        serving_kws = ["serving", "servings", "serves", "pack size", "per 100g", "per 100 g", "per serving", "approx."]
        if any(sk in norm for sk in serving_kws):
            nut_score += 0.25

        # Check raw numeric alignment/presence
        numeric_count = len(_NUMERIC_RE.findall(norm))
        if numeric_count >= 2:
            nut_score += min(0.2, 0.08 * numeric_count)

        nut_score = min(1.0, max(0.0, nut_score))

        # -------------------------------------------------------------
        # 3. OTHER CLASSES SCORE COMPUTATION
        # -------------------------------------------------------------
        # 3. allergen
        allergen_score = 0.0
        allergen_kws = ["allergen", "allergy", "contains:", "may contain", "contains wheat", "contains milk", "contains soy", "contains peanut", "free from"]
        matched_allergen, allergen_anchor_score = best_anchor_match(
            text, allergen_kws, threshold=config.STOP_WORD_CONTEXTUAL_SCORE
        )
        if matched_allergen:
            allergen_score += 0.8 * (allergen_anchor_score / 100.0)
        if any(ak in norm for ak in allergen_kws):
            allergen_score += 0.3
        allergen_score = min(1.0, max(0.0, allergen_score))

        # 4. manufacturer
        mfg_score = 0.0
        mfg_kws = ["manufactured by", "mfg by", "marketed by", "imported by", "distributor", "packed by", "packager", "manufactured for", "address", "registered office", "regd office"]
        matched_mfg, mfg_anchor_score = best_anchor_match(
            text, mfg_kws, threshold=config.STOP_WORD_CONTEXTUAL_SCORE
        )
        if matched_mfg:
            mfg_score += 0.8 * (mfg_anchor_score / 100.0)
        if any(mk in norm for mk in mfg_kws):
            mfg_score += 0.3
        mfg_score = min(1.0, max(0.0, mfg_score))

        # 5. storage
        storage_score = 0.0
        storage_kws = ["storage", "store in", "keep in", "cool and dry", "cool, dry", "refrigerate", "do not freeze", "keep container tightly"]
        matched_storage, storage_anchor_score = best_anchor_match(
            text, storage_kws, threshold=config.STOP_WORD_CONTEXTUAL_SCORE
        )
        if matched_storage:
            storage_score += 0.8 * (storage_anchor_score / 100.0)
        if any(sk in norm for sk in storage_kws):
            storage_score += 0.3
        storage_score = min(1.0, max(0.0, storage_score))

        # 6. directions
        directions_score = 0.0
        directions_kws = ["directions", "instructions", "how to prepare", "cooking time", "prep method", "dosage", "directions for use", "suggested use"]
        matched_directions, directions_anchor_score = best_anchor_match(
            text, directions_kws, threshold=config.STOP_WORD_CONTEXTUAL_SCORE
        )
        if matched_directions:
            directions_score += 0.8 * (directions_anchor_score / 100.0)
        if any(dk in norm for dk in directions_kws):
            directions_score += 0.3
        directions_score = min(1.0, max(0.0, directions_score))

        # 7. regulatory
        regulatory_score = 0.0
        regulatory_kws = ["fssai", "lic. no.", "lic no", "grade", "standard", "regulatory", "hallmark", "registered", "isi mark"]
        matched_regulatory, regulatory_anchor_score = best_anchor_match(
            text, regulatory_kws, threshold=config.STOP_WORD_CONTEXTUAL_SCORE
        )
        if matched_regulatory:
            regulatory_score += 0.8 * (regulatory_anchor_score / 100.0)
        if any(rk in norm for rk in regulatory_kws):
            regulatory_score += 0.3
        regulatory_score = min(1.0, max(0.0, regulatory_score))

        # 8. product_information
        prod_info_score = 0.0
        prod_info_kws = ["mrp", "rs.", "batch", "expiry", "best before", "use by", "net weight", "net wt", "net quantity", "pack size", "manufacturing date", "mfg. date", "mfg date", "expiry date", "exp date", "customer care", "contact us", "feedback", "queries"]
        matched_prod_info, prod_info_anchor_score = best_anchor_match(
            text, prod_info_kws, threshold=config.STOP_WORD_CONTEXTUAL_SCORE
        )
        if matched_prod_info:
            prod_info_score += 0.8 * (prod_info_anchor_score / 100.0)
        if any(pk in norm for pk in prod_info_kws):
            prod_info_score += 0.3
        prod_info_score = min(1.0, max(0.0, prod_info_score))

        # 9. other
        other_score = 0.0
        mkt_kws = ["all rights reserved", "trademark", "patent", "visit us", "www.", "http", ".com", "address", "registered office"]
        if any(mk in norm for mk in mkt_kws):
            other_score += 0.3
        other_score = min(1.0, max(0.0, other_score))

        # Build scores dictionary
        scores = {
            "ingredients": round(float(ing_score), 3),
            "nutrition": round(float(nut_score), 3),
            "allergen": round(float(allergen_score), 3),
            "manufacturer": round(float(mfg_score), 3),
            "storage": round(float(storage_score), 3),
            "directions": round(float(directions_score), 3),
            "regulatory": round(float(regulatory_score), 3),
            "product_information": round(float(prod_info_score), 3),
            "other": round(float(other_score), 3),
        }
        
        predicted_class = max(scores, key=scores.get)
        if max(scores.values()) == 0.0:
            predicted_class = "other"

        # Calculate legacy other_score for backward-compatible region stop conditions
        other_heading_kws = [
            "storage", "directions", "cooking", "manufactured", "marketed",
            "customer care", "contact us", "feedback", "mrp", "batch",
            "best before", "expiry", "use by", "fssai", "barcode", "net weight",
            "net wt", "net quantity"
        ]
        matched_other, other_anchor_score = best_anchor_match(
            text, other_heading_kws, threshold=config.STOP_WORD_CONTEXTUAL_SCORE
        )
        legacy_other = 0.0
        if matched_other:
            legacy_other += 0.8 * (other_anchor_score / 100.0)
        if any(mk in norm for mk in ["all rights reserved", "trademark", "patent", "visit us", "www.", "http", ".com", "address", "registered office"]):
            legacy_other += 0.3
        legacy_other_score = min(1.0, max(0.0, legacy_other))

        # Add scores to augmented line dict
        augmented = dict(ln)
        augmented["ingredient_score"] = scores["ingredients"]
        augmented["nutrition_score"] = scores["nutrition"]
        augmented["other_score"] = round(float(legacy_other_score), 3)
        augmented["scores"] = scores
        augmented["predicted_class"] = predicted_class
        
        classified_lines.append(augmented)

    return classified_lines
