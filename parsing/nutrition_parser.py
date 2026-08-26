"""
parsing/nutrition_parser.py

Parses raw OCR'd nutrition-table text into a structured dict of
{nutrient_key: {"value": float, "unit": str}} per STEP 16.
"""

import re

import config


# Canonical nutrient key -> list of text patterns (already lowercase) that
# should map to it. Longer/more specific phrases are listed first so they
# match before their shorter substrings (e.g. "total sugars" before "sugar").
NUTRIENT_KEY_PATTERNS = [
    ("energy", [r"energy", r"calories"]),
    ("protein", [r"protein"]),
    ("total_carbohydrate", [r"total\s*carbohydrate", r"carbohydrate"]),
    ("total_sugars", [r"total\s*sugars", r"added\s*sugars", r"sugars?"]),
    ("total_fat", [r"total\s*fat", r"fat"]),
    ("saturated_fat", [r"saturated\s*fat"]),
    ("trans_fat", [r"trans\s*fat"]),
    ("dietary_fibre", [r"dietary\s*fib(?:re|er)", r"fib(?:re|er)"]),
    ("sodium", [r"sodium"]),
    ("salt", [r"salt"]),
    ("cholesterol", [r"cholesterol"]),
    ("calcium", [r"calcium"]),
    ("iron", [r"iron"]),
    ("vitamin", [r"vitamin\s*[a-z0-9]*"]),
]

# Units we recognize, ordered longest-first so e.g. "kcal" matches before "cal"
UNIT_PATTERN = r"(kcal|kj|mcg|mg|iu|g|%)"

VALUE_UNIT_PATTERN = re.compile(
    r"(\d+(?:\.\d+)?)\s*" + UNIT_PATTERN, re.IGNORECASE
)


def _find_key(line_lower):
    for key, patterns in NUTRIENT_KEY_PATTERNS:
        for pat in patterns:
            if re.search(r"\b" + pat + r"\b", line_lower):
                return key
    return None


def parse_nutrition(raw_text):
    """
    Parses raw nutrition OCR text (potentially multi-line, one nutrient
    per line, e.g. "Energy 450 kcal") into a structured dict:

        {
            "energy": {"value": 450.0, "unit": "kcal"},
            "protein": {"value": 8.0, "unit": "g"},
            ...
        }

    If a recognized nutrient key appears without an extractable
    value/unit, it is skipped (we do not fabricate numbers).
    If the SAME key is seen more than once, the first confident match is
    kept.
    """
    if not raw_text or not raw_text.strip():
        return {}

    lines = re.split(r"[\n]+", raw_text)
    # Also split lines that clearly contain multiple nutrient entries
    # concatenated (common with OCR merging rows), using nutrient keyword
    # boundaries as extra split points.
    expanded_lines = []
    for line in lines:
        # crude re-split: insert a break before any nutrient keyword that
        # isn't at the start of the line
        parts = re.split(
            r"(?=(?:" + "|".join(config.NUTRIENT_KEYWORDS) + r"))",
            line,
            flags=re.IGNORECASE,
        )
        expanded_lines.extend([p for p in parts if p.strip()])

    result = {}

    for line in expanded_lines:
        line_lower = line.lower()
        key = _find_key(line_lower)
        if key is None:
            continue
        if key in result:
            continue

        match = VALUE_UNIT_PATTERN.search(line)
        if not match:
            continue

        value_str, unit = match.group(1), match.group(2).lower()
        try:
            value = float(value_str)
        except ValueError:
            continue

        result[key] = {"value": value, "unit": unit}

    return result
