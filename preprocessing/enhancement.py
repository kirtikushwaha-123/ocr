"""
preprocessing/enhancement.py

Generates multiple preprocessed variants of a cropped ROI so that the OCR
stage can be run on each and the best result chosen later (see
ocr/ensemble.py and detection scoring in ingredient_region.py /
nutrition_region.py).
"""

import cv2
import numpy as np


SHARPEN_KERNEL = np.array(
    [
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0],
    ]
)


def _enlarge(roi, min_height=400, max_scale=3.0):
    """
    Upscale small ROI crops so downstream OCR has enough pixels to work
    with. Caps the scale factor to avoid absurd upscaling on already-large
    crops.
    """
    h, w = roi.shape[:2]
    if h == 0 or w == 0:
        return roi
    if h >= min_height:
        return roi
    scale = min(max_scale, min_height / float(h))
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    return cv2.resize(roi, (new_w, new_h), interpolation=cv2.INTER_CUBIC)


def to_gray(roi):
    if len(roi.shape) == 2:
        return roi
    return cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)


def apply_clahe(gray, clip_limit=2.0, tile_grid_size=(8, 8)):
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(gray)


def apply_sharpen(gray):
    return cv2.filter2D(gray, -1, SHARPEN_KERNEL)


def apply_otsu(gray):
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return otsu


def apply_adaptive_threshold(gray, block_size=31, c=11):
    # block_size must be odd and > 1
    if block_size % 2 == 0:
        block_size += 1
    return cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block_size,
        c,
    )


def apply_denoise(gray, h=10):
    return cv2.fastNlMeansDenoising(gray, None, h=h, templateWindowSize=7, searchWindowSize=21)


def preprocess_roi(roi):
    """
    Generate a dict of named preprocessing variants for a single ROI:

      original          - enlarged BGR image (as OCR-fed baseline)
      gray               - plain grayscale
      clahe              - CLAHE-enhanced grayscale
      sharpened          - CLAHE + sharpen
      otsu               - CLAHE + OTSU threshold
      adaptive_threshold - CLAHE + adaptive threshold
      denoised_clahe     - denoise + CLAHE
      sharpened_threshold- sharpen + OTSU threshold

    Each variant is a numpy array (grayscale or BGR) suitable for feeding
    directly into the OCR engine.
    """
    if roi is None or roi.size == 0:
        return {}

    enlarged = _enlarge(roi)
    gray = to_gray(enlarged)

    clahe = apply_clahe(gray)
    sharpened = apply_sharpen(clahe)
    otsu = apply_otsu(clahe)
    adaptive = apply_adaptive_threshold(clahe)

    denoised = apply_denoise(gray)
    denoised_clahe = apply_clahe(denoised)

    sharpened_threshold = apply_otsu(sharpened)

    variants = {
        "original": enlarged,
        "gray": gray,
        "clahe": clahe,
        "sharpened": sharpened,
        "otsu": otsu,
        "adaptive_threshold": adaptive,
        "denoised_clahe": denoised_clahe,
        "sharpened_threshold": sharpened_threshold,
    }

    return variants
