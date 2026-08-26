"""
ocr/paddle_engine.py

Thin wrapper around PaddleOCR that:
  1. Initializes the engine once (lazily, cached).
  2. Normalizes whatever output structure the installed PaddleOCR version
     returns into a single standardized format:

        [
            {
                "text": "...",
                "confidence": 0.95,
                "bbox": [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
            },
            ...
        ]

PaddleOCR's output format has changed across versions (2.x "predict"
style list-of-lists vs newer "OCRResult" dict-like objects with
'rec_texts' / 'rec_scores' / 'rec_polys' / 'dt_polys' keys, plus older
`.ocr(img, cls=True)` vs newer `.predict(img)` / `.ocr(img)` calling
conventions). Rather than hard-coding one shape, this module inspects
whatever comes back and adapts.
"""

import numpy as np

import config

_OCR_ENGINE = None
_INIT_ERROR = None


def _init_engine():
    global _OCR_ENGINE, _INIT_ERROR

    if _OCR_ENGINE is not None or _INIT_ERROR is not None:
        return

    try:
        from paddleocr import PaddleOCR
    except ImportError as e:
        _INIT_ERROR = (
            "PaddleOCR is not installed. Install it with:\n"
            "  pip install paddlepaddle paddleocr\n"
            f"(original error: {e})"
        )
        return

    # Different PaddleOCR versions accept different constructor kwargs.
    # We try progressively simpler kwarg sets until one works.
    attempts = [
        dict(lang=config.OCR_LANG, use_angle_cls=config.OCR_USE_ANGLE_CLS, show_log=False),
        dict(lang=config.OCR_LANG, use_angle_cls=config.OCR_USE_ANGLE_CLS),
        dict(lang=config.OCR_LANG),
        dict(),
    ]

    last_err = None
    for kwargs in attempts:
        try:
            _OCR_ENGINE = PaddleOCR(**kwargs)
            return
        except TypeError as e:
            last_err = e
            continue
        except Exception as e:
            last_err = e
            continue

    _INIT_ERROR = f"Failed to initialize PaddleOCR with any known kwarg set: {last_err}"


def is_available():
    _init_engine()
    return _OCR_ENGINE is not None


def get_init_error():
    return _INIT_ERROR


def _to_float(x, default=0.0):
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def _normalize_bbox(raw_box):
    """
    Ensures a bbox is a list of 4 [x, y] points (floats).
    Accepts numpy arrays, nested lists, or flat [x1,y1,x2,y2] rectangles.
    """
    arr = np.array(raw_box)

    if arr.shape == (4, 2):
        return arr.astype(float).tolist()

    if arr.size == 4:
        # Could be [x1,y1,x2,y2] rectangle -> expand to 4 corners
        flat = arr.reshape(-1).astype(float)
        x1, y1, x2, y2 = flat
        return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]

    if arr.ndim == 2 and arr.shape[1] == 2:
        # polygon with != 4 points -> take bounding rect
        xs = arr[:, 0]
        ys = arr[:, 1]
        x1, x2 = float(xs.min()), float(xs.max())
        y1, y2 = float(ys.min()), float(ys.max())
        return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]

    raise ValueError(f"Unrecognized bbox shape: {arr.shape}")


def _parse_legacy_list_result(result):
    """
    Handles the classic PaddleOCR .ocr() output:

        result = [
            [
                [box, (text, confidence)],
                [box, (text, confidence)],
                ...
            ]
        ]

    or sometimes just the inner list without the outer wrapper, depending
    on version / whether multiple images were passed.
    """
    items = []

    def handle_page(page):
        for entry in page:
            if entry is None:
                continue
            try:
                box, rec = entry
                text, confidence = rec[0], rec[1]
            except (ValueError, IndexError, TypeError):
                continue
            try:
                bbox = _normalize_bbox(box)
            except ValueError:
                continue
            items.append(
                {
                    "text": str(text),
                    "confidence": _to_float(confidence),
                    "bbox": bbox,
                }
            )

    if not result:
        return items

    # result could be [page] or just page (list of entries)
    first = result[0]
    if isinstance(first, list) and len(first) > 0 and isinstance(first[0], (list, tuple)):
        # Could be a single page's entries, or a list-of-pages.
        # Heuristic: an "entry" looks like [box, (text, conf)]. A "page"
        # is a list of entries. Check the shape of first[0].
        candidate = first[0]
        looks_like_entry = (
            len(candidate) == 2
            and hasattr(candidate[0], "__len__")
        )
        if looks_like_entry and not isinstance(first[0][0], list):
            # `result` itself is a single page
            handle_page(result)
        else:
            for page in result:
                if page:
                    handle_page(page)
    else:
        handle_page(result)

    return items


def _parse_dict_like_result(result):
    """
    Handles newer PaddleOCR output where each page is a dict-like /
    OCRResult object exposing keys such as:
        rec_texts, rec_scores, rec_polys (or dt_polys / rec_boxes)
    """
    items = []

    for page in result:
        # page might be a dict, or an object with attribute access, or
        # something supporting __getitem__ for these keys.
        def get(key, default=None):
            if page is None:
                return default
            if isinstance(page, dict):
                return page.get(key, default)
            return getattr(page, key, default)

        texts = get("rec_texts")
        scores = get("rec_scores")
        polys = get("rec_polys")
        if polys is None:
            polys = get("dt_polys")
        if polys is None:
            polys = get("rec_boxes")

        if texts is None or polys is None:
            continue

        scores = scores if scores is not None else [1.0] * len(texts)

        for text, score, poly in zip(texts, scores, polys):
            try:
                bbox = _normalize_bbox(poly)
            except ValueError:
                continue
            items.append(
                {
                    "text": str(text),
                    "confidence": _to_float(score),
                    "bbox": bbox,
                }
            )

    return items


def _standardize_result(result):
    """
    Top-level adapter: figures out which shape `result` is in and routes
    to the right parser. Never raises on unexpected shapes - returns
    whatever it could successfully parse (possibly an empty list).
    """
    if result is None:
        return []

    if not isinstance(result, (list, tuple)):
        result = [result]

    if len(result) == 0:
        return []

    first_page = result[0]

    is_dict_like = isinstance(first_page, dict) or hasattr(first_page, "rec_texts")

    if is_dict_like:
        try:
            parsed = _parse_dict_like_result(result)
            if parsed:
                return parsed
        except Exception:
            pass

    try:
        parsed = _parse_legacy_list_result(result)
        if parsed:
            return parsed
    except Exception:
        pass

    return []


def run_ocr(image):
    """
    Runs PaddleOCR on `image` (a BGR or grayscale numpy array) and returns
    a standardized list of:

        {"text": str, "confidence": float, "bbox": [[x,y]*4]}

    Returns an empty list (never raises) if PaddleOCR is unavailable or
    fails - callers should treat an empty list as "no text found" and the
    caller/CLI is responsible for surfacing get_init_error() to the user.
    """
    _init_engine()
    if _OCR_ENGINE is None:
        return []

    if image is None or image.size == 0:
        return []

    # Some preprocessing variants are single-channel; PaddleOCR generally
    # accepts grayscale, but we convert to 3-channel BGR to be safe across
    # versions.
    img = image
    if len(img.shape) == 2:
        import cv2

        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    raw_result = None

    # Newer API: .predict(img) -> list of OCRResult-like dicts
    if hasattr(_OCR_ENGINE, "predict"):
        try:
            raw_result = _OCR_ENGINE.predict(img)
        except Exception:
            raw_result = None

    if raw_result is None and hasattr(_OCR_ENGINE, "ocr"):
        for kwargs in (dict(cls=config.OCR_USE_ANGLE_CLS), dict()):
            try:
                raw_result = _OCR_ENGINE.ocr(img, **kwargs)
                break
            except TypeError:
                continue
            except Exception:
                raw_result = None
                break

    return _standardize_result(raw_result)
