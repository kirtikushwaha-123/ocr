"""
detection/section_signals.py

Shared, section-agnostic evidence used by BOTH detection/ingredient_region.py
and detection/nutrition_region.py, so neither module depends on the other
and both apply the *same* rules when deciding whether a candidate line
belongs to their section or to the OTHER section.

This is the mechanism that keeps Ingredients and Nutrition independent
even when they sit close together (side-by-side, stacked, or with a
narrow gutter): each detector's expansion loop calls
`classify_nutrition_row()` / `classify_ingredient_line()` on every
candidate line and DISQUALIFIES it outright if it looks like the other
section's content - regardless of geometric proximity, band alignment,
or vertical gap. Geometry gets you a candidate; semantics gets you the
right boundary.
"""

import re

import config
from detection.ocr_detector import normalize_ocr_text, best_anchor_match


_NUTRITION_UNIT_RE = re.compile(config.NUTRITION_ROW_UNIT_PATTERN, re.IGNORECASE)


def classify_nutrition_row(line_text):
    """
    Decides how strongly a single line looks like a Nutrition table row.

    Returns:
        {
            "keyword_hits": [...],      # nutrient keywords found
            "has_number_unit": bool,    # e.g. "123 kcal", "8 g", "121 mg"
            "is_strong_nutrition_row": bool,
        }

    A line is a STRONG nutrition row only when it has a nutrient keyword
    AND a number+unit pattern (e.g. "Protein 8 g") where the number is after
    and close to the keyword, or a nutrient keyword alone when the line is
    short and the keyword is the leading token. This avoids false positives.
    """
    norm = normalize_ocr_text(line_text)
    keyword_hits = [kw for kw in config.NUTRIENT_KEYWORDS if kw in norm]
    has_number_unit = bool(_NUTRITION_UNIT_RE.search(norm))

    is_strong = False
    if keyword_hits and has_number_unit:
        # Require the number-unit pattern to be after and close to the keyword (within 15 chars)
        for kw in keyword_hits:
            for m_kw in re.finditer(re.escape(kw), norm):
                kw_end = m_kw.end()
                for m_unit in _NUTRITION_UNIT_RE.finditer(norm):
                    unit_start = m_unit.start()
                    if unit_start >= kw_end and (unit_start - kw_end) <= 15:
                        is_strong = True
                        break
                if is_strong:
                    break
            if is_strong:
                break
    elif keyword_hits and len(norm) <= 24:
        # Require the keyword to be the line's leading token (ignoring prefix punctuation/noise)
        is_leading = False
        for kw in keyword_hits:
            if re.match(r'^[-\s:]*' + re.escape(kw) + r'\b', norm):
                is_leading = True
                break
        if is_leading:
            is_strong = True

    return {
        "keyword_hits": keyword_hits,
        "has_number_unit": has_number_unit,
        "is_strong_nutrition_row": is_strong,
    }


def classify_ingredient_line(line_text, ingredient_vocab=None):
    """
    Decides how strongly a single line looks like Ingredients paragraph
    content: a comma-separated list and/or known ingredient vocabulary
    hits. Marketing copy like "100% WHEAT" or "HIGH WHEAT FIBRE" has zero
    commas and (at most) one vocab hit, so it does NOT qualify as strong
    evidence on its own (STEP 7 / STEP 19 requirement).

    Returns:
        {
            "comma_count": int,
            "vocab_hits": int,
            "is_strong_ingredient_line": bool,
        }
    """
    norm = normalize_ocr_text(line_text)
    comma_count = line_text.count(",")

    vocab_hits = 0
    if ingredient_vocab:
        for term in ingredient_vocab:
            if term and len(term) >= 4 and term.lower() in norm:
                vocab_hits += 1

    is_strong = comma_count >= config.INGREDIENT_LINE_MIN_COMMAS or vocab_hits >= 2

    return {
        "comma_count": comma_count,
        "vocab_hits": vocab_hits,
        "is_strong_ingredient_line": is_strong,
    }


def detect_section_boundaries(line_text, stop_words):
    """
    Decides whether a line marks the start of a NEW, unrelated section
    (and expansion should therefore stop before including it).

    Returns (is_stop: bool, matched_stop: str or None, score: float, reason: str)

    Multi-signal, not single-keyword:
      - "contains" is ambiguous (weak ingredient anchor word AND allergen
        lead-in) - only treated as a stop if the line also mentions a
        common allergen term.
      - High-confidence fuzzy matches (>= STOP_WORD_AUTO_SCORE) always stop.
      - Medium-confidence matches only stop if the line is short/heading-
        like (long lines are more likely body text that merely contains a
        stop-ish substring by coincidence).
    """
    norm = normalize_ocr_text(line_text)
    matched, score = best_anchor_match(
        line_text, stop_words, threshold=config.STOP_WORD_CONTEXTUAL_SCORE
    )
    if matched is None:
        return False, None, score, "no_match"

    if matched == "contains":
        has_allergen_word = any(w in norm for w in config.ALLERGEN_CONTEXT_WORDS)
        if not has_allergen_word:
            return False, matched, score, "contains_without_allergen_context"
        return True, matched, score, "contains_with_allergen_context"

    if score >= config.STOP_WORD_AUTO_SCORE:
        return True, matched, score, "high_confidence_stop"

    if len(norm) <= 40:
        return True, matched, score, "contextual_stop_short_heading"

    return False, matched, score, "contextual_stop_rejected_long_line"
