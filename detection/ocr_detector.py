"""
detection/ocr_detector.py

- normalize_ocr_text(): cleans OCR text for matching.
- run_full_image_ocr(): runs PaddleOCR on the whole (normalized) image and
  converts raw items into the enriched item format used everywhere else
  in the detection package (adds "rect" = axis-aligned bbox).
- anchor matching helpers (fuzzy, via rapidfuzz when available, else a
  pure-python fallback).
"""

import re
import unicodedata

from ocr.paddle_engine import run_ocr
from detection.geometry import poly_to_rect
import config

try:
    from rapidfuzz import fuzz as _rf_fuzz

    _HAS_RAPIDFUZZ = True
except ImportError:
    _HAS_RAPIDFUZZ = False


def normalize_ocr_text(text):
    """
    Normalize OCR text for robust matching:
      - unicode normalization (NFKC)
      - lowercase
      - trim + collapse whitespace
      - normalize common punctuation variants (colon, dash, bullets)
    """
    if text is None:
        return ""

    text = unicodedata.normalize("NFKC", text)
    text = text.lower()

    # normalize common OCR punctuation noise
    text = text.replace("|", "l")
    text = re.sub(r"[·•●○]", " ", text)
    text = re.sub(r"[-–—]", "-", text)
    text = re.sub(r"[:：]", ":", text)

    text = re.sub(r"\s+", " ", text).strip()
    return text


def _simple_ratio(a, b):
    """
    Pure-python fallback similarity (0-100) when rapidfuzz is unavailable.
    Uses a basic Levenshtein-ratio implementation.
    """
    if a == b:
        return 100.0
    if not a or not b:
        return 0.0

    la, lb = len(a), len(b)
    # standard DP levenshtein distance
    prev = list(range(lb + 1))
    for i in range(1, la + 1):
        curr = [i] + [0] * lb
        for j in range(1, lb + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            curr[j] = min(
                prev[j] + 1,
                curr[j - 1] + 1,
                prev[j - 1] + cost,
            )
        prev = curr
    distance = prev[lb]
    max_len = max(la, lb)
    return 100.0 * (1.0 - distance / max_len) if max_len else 100.0


def _simple_partial_ratio(a, b):
    """
    Pure-python fallback approximating rapidfuzz's partial_ratio: slides
    the SHORTER string across the longer one and returns the best
    windowed Levenshtein-ratio match. Unlike a plain whole-string ratio,
    this does not get dragged down just because one string (typically a
    whole merged OCR line) is much longer than the other (a short anchor
    phrase like "allergen advice").
    """
    if not a or not b:
        return 0.0
    shorter, longer = (a, b) if len(a) <= len(b) else (b, a)
    if len(shorter) >= len(longer):
        return _simple_ratio(a, b)

    best = 0.0
    step = max(1, len(shorter) // 4)  # coarse slide, good enough for short anchors
    for start in range(0, len(longer) - len(shorter) + 1, step):
        window = longer[start:start + len(shorter)]
        score = _simple_ratio(shorter, window)
        if score > best:
            best = score
    return best


def fuzzy_score(a, b):
    """
    Returns similarity 0-100 between two normalized strings.

    Uses the MAX of a whole-string comparison (token_sort_ratio) and a
    partial/substring comparison (partial_ratio). The whole-string score
    alone badly under-scores exactly the case that matters most for
    section-boundary detection: a short heading phrase (e.g. "allergen
    advice") that OCR-merged onto the same line as a much longer trailing
    sentence, or that has a couple of OCR-noise characters. Ratio-based
    measures penalize that length mismatch heavily even when the phrase
    itself is a near-perfect match, which is what let expansion run past
    real section boundaries and produce oversized crops. partial_ratio
    finds the best-matching substring instead, so it is not penalized by
    surrounding text.
    """
    if _HAS_RAPIDFUZZ:
        return max(_rf_fuzz.token_sort_ratio(a, b), _rf_fuzz.partial_ratio(a, b))
    return max(_simple_ratio(a, b), _simple_partial_ratio(a, b))


def best_anchor_match(text, anchor_list, threshold=None):
    """
    Given a (raw) OCR text string and a list of anchor phrases, returns
    (matched_anchor, score) for the best fuzzy match at/above threshold,
    or (None, best_score_seen) if nothing clears the bar.

    Matching also checks substring containment (common for OCR text that
    includes the anchor plus extra words, e.g. "INGREDIENTS: Wheat...").
    """
    threshold = threshold if threshold is not None else config.FUZZY_ANCHOR_THRESHOLD
    norm_text = normalize_ocr_text(text)
    if not norm_text:
        return None, 0.0

    best_anchor = None
    best_score = 0.0

    for anchor in anchor_list:
        norm_anchor = normalize_ocr_text(anchor)
        if not norm_anchor:
            continue

        if norm_anchor in norm_text or norm_text in norm_anchor:
            score = 100.0
        else:
            score = fuzzy_score(norm_text, norm_anchor)

        if score > best_score:
            best_score = score
            best_anchor = norm_anchor

    if best_score >= threshold:
        return best_anchor, best_score
    return None, best_score


# def enrich_items(raw_items):
#     """
#     Adds "rect" (axis-aligned [x1,y1,x2,y2]) and "norm_text" fields to raw
#     OCR items coming out of ocr/paddle_engine.run_ocr.
#     """
#     enriched = []
#     for it in raw_items:
#         try:
#             rect = poly_to_rect(it["bbox"])
#         except Exception:
#             continue
#         enriched.append(
#             {
#                 "text": it["text"],
#                 "norm_text": normalize_ocr_text(it["text"]),
#                 "confidence": it.get("confidence", 0.0),
#                 "bbox": it["bbox"],
#                 "rect": rect,
#             }
#         )
#     return enriched

def enrich_items(raw_items):
    """
    Convert raw PaddleOCR items into the canonical OCR representation.

    IMPORTANT:
    The original OCR polygon is preserved.

    `polygon`:
        Original PaddleOCR polygon.

    `bbox`:
        Backward-compatible alias for the original polygon.

    `rect`:
        Axis-aligned [x1, y1, x2, y2] rectangle used by existing
        geometry/detection code.

    Additional geometry:
        center, width, height.

    Keeping both polygon and rect is important because later stages
    may need polygon-aware perspective/rotation processing.
    """

    enriched = []

    for it in raw_items:
        try:
            # ---------------------------------------------------------
            # 1. Preserve the original OCR polygon
            # ---------------------------------------------------------
            polygon = it.get("bbox")

            if polygon is None:
                continue

            # ---------------------------------------------------------
            # 2. Convert polygon -> axis-aligned rectangle
            # ---------------------------------------------------------
            rect = poly_to_rect(polygon)

            x1, y1, x2, y2 = rect

            # ---------------------------------------------------------
            # 3. Derived geometry
            # ---------------------------------------------------------
            width = max(
                0.0,
                float(x2 - x1)
            )

            height = max(
                0.0,
                float(y2 - y1)
            )

            center = [
                float((x1 + x2) / 2.0),
                float((y1 + y2) / 2.0),
            ]

            # ---------------------------------------------------------
            # 4. Canonical OCR item
            # ---------------------------------------------------------
            enriched.append(
                {
                    # OCR text
                    "text": it.get("text", ""),

                    # Normalized text used by matching/detection
                    "norm_text": normalize_ocr_text(
                        it.get("text", "")
                    ),

                    # OCR confidence
                    "confidence": float(
                        it.get("confidence", 0.0)
                    ),

                    # -------------------------------------------------
                    # ORIGINAL POLYGON
                    # -------------------------------------------------
                    "polygon": polygon,

                    # -------------------------------------------------
                    # Backward compatibility
                    #
                    # Existing project code already expects "bbox".
                    # Do NOT remove it.
                    # -------------------------------------------------
                    "bbox": polygon,

                    # -------------------------------------------------
                    # Axis-aligned rectangle
                    # -------------------------------------------------
                    "rect": rect,

                    # -------------------------------------------------
                    # Derived geometry
                    # -------------------------------------------------
                    "center": center,
                    "width": width,
                    "height": height,
                }
            )

        except Exception:
            # Keep the existing behavior:
            # one malformed OCR item must not break the complete OCR run.
            continue

    return enriched


def run_full_image_ocr(image):
    """
    Runs OCR on the full (already normalized/resized) image and returns
    the enriched item list.
    """
    raw_items = run_ocr(image)
    return enrich_items(raw_items)


def find_anchor_items(items, anchor_list, threshold=None):
    """
    Scans enriched OCR items and returns a list of
        {"item": item, "matched_anchor": str, "score": float}
    for every item whose text fuzzy-matches something in anchor_list.
    Sorted by score descending.
    """
    threshold = threshold if threshold is not None else config.FUZZY_ANCHOR_THRESHOLD
    matches = []
    for it in items:
        matched, score = best_anchor_match(it["text"], anchor_list, threshold)
        if matched is not None:
            matches.append({"item": it, "matched_anchor": matched, "score": score})

    matches.sort(key=lambda m: m["score"], reverse=True)
    return matches


def find_anchor_candidates(lines, anchor_list, threshold=None, min_text_len=2):
    """
    Line-oriented version of find_anchor_items(): scans a list of "line"
    dicts (see detection.geometry.group_ocr_boxes_into_lines - each has a
    'text' field) and returns candidates whose text fuzzy-matches
    something in anchor_list, sorted by score descending.

    `min_text_len` guards against 1-2 character OCR noise being accepted
    as a heading match purely because a short fuzzy string happens to
    score high.

    Returns: [{"line": line_dict, "matched_anchor": str, "score": float}, ...]
    """
    threshold = threshold if threshold is not None else config.FUZZY_ANCHOR_THRESHOLD
    candidates = []
    for ln in lines:
        norm = normalize_ocr_text(ln["text"])
        if len(norm) < min_text_len:
            continue
        matched, score = best_anchor_match(ln["text"], anchor_list, threshold)
        if matched is not None:
            candidates.append({"line": ln, "matched_anchor": matched, "score": score})

    candidates.sort(key=lambda c: c["score"], reverse=True)
    return candidates
