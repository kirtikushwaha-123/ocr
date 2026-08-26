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
        # 3. OTHER SECTION SCORE COMPUTATION
        # -------------------------------------------------------------
        other_score = 0.0

        # Stop words / Other section headings
        other_heading_kws = [
            "storage", "directions", "cooking", "manufactured", "marketed",
            "customer care", "contact us", "feedback", "mrp", "batch",
            "best before", "expiry", "use by", "fssai", "barcode", "net weight",
            "net wt", "net quantity"
        ]
        
        matched_other, other_anchor_score = best_anchor_match(
            text, other_heading_kws, threshold=config.STOP_WORD_CONTEXTUAL_SCORE
        )
        if matched_other:
            other_score += 0.8 * (other_anchor_score / 100.0)

        # Common non-product keywords (marketing, legal, barcode numbers)
        mkt_kws = ["all rights reserved", "trademark", "patent", "visit us", "www.", "http", ".com", "address", "registered office"]
        if any(mk in norm for mk in mkt_kws):
            other_score += 0.3

        other_score = min(1.0, max(0.0, other_score))

        # Add scores to augmented line dict
        augmented = dict(ln)
        augmented["ingredient_score"] = round(float(ing_score), 3)
        augmented["nutrition_score"] = round(float(nut_score), 3)
        augmented["other_score"] = round(float(other_score), 3)
        
        classified_lines.append(augmented)

    return classified_lines
