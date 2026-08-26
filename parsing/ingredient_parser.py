"""
parsing/ingredient_parser.py

Parses raw OCR'd ingredient-section text into a clean list of ingredient
names (STEP 15). Handles:
  - commas / semicolons as separators
  - parenthetical sub-ingredients / additive codes, e.g. "Citric Acid (INS 330)"
  - INS/E numbers
  - line breaks
  - percentages, e.g. "Wheat Flour (65%)"
  - heading text like "INGREDIENTS:" stripped from the front

Care is taken NOT to split multi-word ingredient names such as
"Sodium Benzoate" or "Black Pepper" - splitting only happens on the
designated separator characters, never on whitespace.
"""

import re

import config
from detection.ocr_detector import normalize_ocr_text, best_anchor_match


HEADING_STRIP_PATTERN = re.compile(
    r"^\s*(ingredients?|composition|contents)\s*:?\s*-?\s*", re.IGNORECASE
)

# e.g. "INS 330", "E 330", "E330", "INS330"
INS_E_NUMBER_PATTERN = re.compile(r"\b(?:ins|e)\s?-?\s?\d{3,4}[a-z]?\b", re.IGNORECASE)

# e.g. "(65%)", "65%", "(65 %)"
PERCENT_PATTERN = re.compile(r"\(?\s?\d{1,3}(\.\d+)?\s?%\s?\)?")

# Splits only on comma / semicolon / bullet-like separators, never on
# bare whitespace, so multi-word names stay intact.
SPLIT_PATTERN = re.compile(r"[;,]|(?:\n)+")


def _strip_leading_heading(text):
    return HEADING_STRIP_PATTERN.sub("", text, count=1)


def _strip_parenthetical_codes(token):
    """
    Removes purely additive-code parentheticals like "(INS 330)" or
    "(E330)" but KEEPS parentheticals that look like alternate/sub names,
    e.g. "Vegetable Oil (Palm)" -> we keep "(Palm)" text but drop it from
    being treated as a separate ingredient; the outer name is what we
    keep as the primary token.
    """
    def repl(match):
        inner = match.group(0)
        if INS_E_NUMBER_PATTERN.search(inner):
            return " "
        return inner

    token = re.sub(r"\([^)]*\)", repl, token)
    return token


def _clean_token(token):
    token = _strip_parenthetical_codes(token)
    token = PERCENT_PATTERN.sub(" ", token)
    token = INS_E_NUMBER_PATTERN.sub(" ", token)

    # strip stray parentheses/brackets left over, but keep inner text if
    # it wasn't a code (e.g. "(Palm)" -> "Palm")
    token = token.replace("(", " ").replace(")", " ")
    token = token.replace("[", " ").replace("]", " ")

    token = re.sub(r"\s+", " ", token).strip(" .-")
    return token


def parse_ingredients(raw_text):
    """
    Parses raw ingredient-section OCR text into a clean, de-duplicated
    list of lowercase ingredient name strings, preserving multi-word
    names intact.
    """
    if not raw_text or not raw_text.strip():
        return []

    text = _strip_leading_heading(raw_text.strip())

    raw_tokens = SPLIT_PATTERN.split(text)

    ingredients = []
    seen = set()

    for tok in raw_tokens:
        cleaned = _clean_token(tok)
        if not cleaned:
            continue

        cleaned_lower = cleaned.lower()

        # Drop tokens that are just noise: pure numbers, single characters,
        # or things that are actually stop-word section headings that
        # leaked in (e.g. "nutrition information" at the tail of a crop).
        if len(cleaned_lower) < 2:
            continue
        if re.fullmatch(r"[\d.\s]+", cleaned_lower):
            continue

        stop_match, stop_score = best_anchor_match(
            cleaned_lower, config.SECTION_STOP_WORDS, threshold=88
        )
        if stop_match is not None:
            continue

        if cleaned_lower in seen:
            continue

        seen.add(cleaned_lower)
        ingredients.append(cleaned_lower)

    return ingredients
