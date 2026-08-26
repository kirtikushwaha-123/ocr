"""
detection/packet_region.py

Lightweight OpenCV-only packet/foreground isolation (NO YOLO). Tries to
find the packaging in the frame and crop to it. Falls back to the full
image whenever confidence is not high enough, since a bad crop is worse
than no crop.
"""

import cv2
import numpy as np

import config


def _largest_quad_or_box(contours, image_area):
    best = None
    best_score = 0.0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < image_area * config.PACKET_MIN_AREA_RATIO:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        rect_area = w * h
        if rect_area == 0:
            continue

        extent = area / rect_area  # how "filled" the bounding box is
        area_ratio = area / image_area

        # Prefer contours that are large AND reasonably rectangular
        score = area_ratio * 0.7 + extent * 0.3

        if score > best_score:
            best_score = score
            best = (x, y, w, h, extent, area_ratio)

    return best, best_score


def detect_packet_region(image):
    """
    Attempts to isolate the packaging from the background.

    Returns:
        packet_crop: cropped BGR image (or the original image if no
                     confident detection was made)
        packet_bbox: [x1, y1, x2, y2] in the ORIGINAL image's coordinates
        detection_confidence: float in [0, 1]
    """
    h, w = image.shape[:2]
    image_area = h * w

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, config.PACKET_CANNY_LOW, config.PACKET_CANNY_HIGH)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, config.PACKET_MORPH_KERNEL)
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    closed = cv2.dilate(closed, kernel, iterations=1)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return image, [0, 0, w, h], 0.0

    best, best_score = _largest_quad_or_box(contours, image_area)

    if best is None:
        return image, [0, 0, w, h], 0.0

    x, y, bw, bh, extent, area_ratio = best

    confidence = float(min(1.0, best_score))

    if confidence < config.PACKET_MIN_CONFIDENCE_TO_CROP:
        # Not confident enough - use the full image so we don't risk
        # cropping away the ingredients/nutrition panels.
        return image, [0, 0, w, h], confidence

    # Add a small margin so we don't clip text right at the packet edge
    margin_x = int(bw * 0.03)
    margin_y = int(bh * 0.03)
    x1 = max(0, x - margin_x)
    y1 = max(0, y - margin_y)
    x2 = min(w, x + bw + margin_x)
    y2 = min(h, y + bh + margin_y)

    crop = image[y1:y2, x1:x2].copy()
    if crop.size == 0:
        return image, [0, 0, w, h], 0.0

    return crop, [x1, y1, x2, y2], confidence
