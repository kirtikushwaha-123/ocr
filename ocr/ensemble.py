"""
ocr/ensemble.py

Runs OCR across multiple preprocessing variants of an ROI and selects the
best result using a combined score (not just raw OCR confidence), per
STEP 14 of the spec.
"""

import re

from ocr.paddle_engine import run_ocr
import config


GARBAGE_CHAR_PATTERN = re.compile(r"[^a-zA-Z0-9%.,:;()\-\s]")


def _garbage_penalty(text):
    """
    Rough heuristic: fraction of characters that are neither alphanumeric
    nor common punctuation, plus a penalty for very short "words" that
    are pure symbols.
    """
    if not text:
        return 1.0
    junk = len(GARBAGE_CHAR_PATTERN.findall(text))
    return junk / max(len(text), 1)


def _word_count(text):
    return len(re.findall(r"[A-Za-z]{2,}", text))


def _keyword_hits(text_lower, keywords):
    return sum(1 for kw in keywords if kw in text_lower)


def _unit_hits(text_lower):
    return sum(1 for u in config.NUTRIENT_UNITS if u in text_lower)


def _number_hits(text):
    return len(re.findall(r"\d+(\.\d+)?", text))


def score_ocr_items(items, mode):
    """
    mode: "ingredient" | "nutrition" | "generic"

    Returns (score: float, joined_text: str, avg_confidence: float)
    """
    if not items:
        return 0.0, "", 0.0

    joined_text = " ".join(i["text"] for i in items)
    text_lower = joined_text.lower()
    avg_conf = sum(i["confidence"] for i in items) / len(items)

    words = _word_count(joined_text)
    garbage = _garbage_penalty(joined_text)

    if mode == "ingredient":
        ingredient_kw = _keyword_hits(text_lower, config.INGREDIENT_ANCHORS)
        # reasonable line structure: presence of commas (ingredient lists
        # are typically comma separated)
        comma_bonus = min(joined_text.count(","), 10) * 0.05
        score = (
            avg_conf * 2.0
            + words * 0.05
            + ingredient_kw * 0.3
            + comma_bonus
            - garbage * 1.5
        )
    elif mode == "nutrition":
        nutrient_kw = _keyword_hits(text_lower, config.NUTRIENT_KEYWORDS)
        units = _unit_hits(text_lower)
        numbers = _number_hits(joined_text)
        # crude "table structure" signal: multiple lines each containing a number
        lines_with_numbers = sum(
            1 for i in items if re.search(r"\d", i["text"])
        )
        score = (
            avg_conf * 2.0
            + nutrient_kw * 0.35
            + units * 0.15
            + min(numbers, 20) * 0.05
            + lines_with_numbers * 0.1
            - garbage * 1.5
        )
    else:
        score = avg_conf * 2.0 + words * 0.05 - garbage * 1.5

    return float(score), joined_text, float(avg_conf)


def run_variant_ocr(variants, mode="generic"):
    """
    variants: dict of {variant_name: image_array} (see
              preprocessing.enhancement.preprocess_roi)
    mode: passed through to score_ocr_items

    Returns:
        {
          "best_variant": str or None,
          "best_score": float,
          "best_text": str,
          "best_confidence": float,
          "best_items": [...],
          "all_variants": {
              variant_name: {
                  "score": float, "text": str, "confidence": float,
                  "items": [...]
              }
          }
        }
    """
    all_results = {}
    best_variant = None
    best_score = float("-inf")

    for name, img in variants.items():
        items = run_ocr(img)
        score, text, conf = score_ocr_items(items, mode)
        all_results[name] = {
            "score": score,
            "text": text,
            "confidence": conf,
            "items": items,
        }
        if score > best_score:
            best_score = score
            best_variant = name

    if best_variant is None:
        return {
            "best_variant": None,
            "best_score": 0.0,
            "best_text": "",
            "best_confidence": 0.0,
            "best_items": [],
            "all_variants": all_results,
        }

    best = all_results[best_variant]
    return {
        "best_variant": best_variant,
        "best_score": best["score"],
        "best_text": best["text"],
        "best_confidence": best["confidence"],
        "best_items": best["items"],
        "all_variants": all_results,
    }
