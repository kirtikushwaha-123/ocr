"""
preprocessing/deskew.py

Estimates and corrects small rotational skew in a cropped ROI, using a
combination of:
  - Hough line angle voting
  - Minimum-area-rectangle angle of the largest ink blob

We deliberately only correct SMALL skew angles (<= ~15 degrees) with
reasonable confidence. Large "skew" is more likely a genuine layout
feature (e.g. vertical text) or a bad detection - we do not want to
mangle those cases.
"""

import cv2
import numpy as np


MAX_CORRECTABLE_ANGLE = 15.0
MIN_CONFIDENCE_TO_ROTATE = 0.35


def _angle_from_hough(gray):
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180, threshold=80, minLineLength=gray.shape[1] // 4, maxLineGap=10
    )
    if lines is None or len(lines) == 0:
        return None, 0.0

    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        dx, dy = (x2 - x1), (y2 - y1)
        if dx == 0:
            continue
        angle = np.degrees(np.arctan2(dy, dx))
        # Fold to [-45, 45] since text lines are close to horizontal
        if angle > 45:
            angle -= 90
        elif angle < -45:
            angle += 90
        angles.append(angle)

    if not angles:
        return None, 0.0

    angles = np.array(angles)
    median_angle = float(np.median(angles))
    # confidence: how tightly the angles cluster around the median
    spread = float(np.std(angles))
    confidence = max(0.0, 1.0 - min(spread / 20.0, 1.0))
    return median_angle, confidence


def _angle_from_min_area_rect(gray):
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    coords = cv2.findNonZero(binary)
    if coords is None or len(coords) < 20:
        return None, 0.0

    rect = cv2.minAreaRect(coords)
    angle = rect[-1]
    # OpenCV's minAreaRect angle convention varies; normalize to [-45, 45]
    if angle < -45:
        angle = 90 + angle
    if angle > 45:
        angle = angle - 90
    confidence = 0.3  # weaker signal on its own
    return float(angle), confidence


def estimate_skew_angle(roi):
    """
    Returns (angle_degrees, confidence in [0,1]).
    Positive angle = rotated counter-clockwise (text leans up to the right).
    """
    if roi is None or roi.size == 0:
        return 0.0, 0.0

    gray = roi if len(roi.shape) == 2 else cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    hough_angle, hough_conf = _angle_from_hough(gray)
    rect_angle, rect_conf = _angle_from_min_area_rect(gray)

    candidates = []
    if hough_angle is not None:
        candidates.append((hough_angle, hough_conf * 1.0))
    if rect_angle is not None:
        candidates.append((rect_angle, rect_conf * 0.6))

    if not candidates:
        return 0.0, 0.0

    total_weight = sum(c for _, c in candidates) or 1e-6
    weighted_angle = sum(a * c for a, c in candidates) / total_weight
    combined_confidence = min(1.0, total_weight)

    return float(weighted_angle), float(combined_confidence)


def deskew(roi):
    """
    Estimate skew and rotate the ROI to correct it, but ONLY if the
    estimated angle is within a correctable range and confidence is high
    enough. Otherwise returns the original ROI unchanged.

    Returns (corrected_roi, applied_angle_or_none).
    """
    if roi is None or roi.size == 0:
        return roi, None

    angle, confidence = estimate_skew_angle(roi)

    if confidence < MIN_CONFIDENCE_TO_ROTATE:
        return roi, None
    if abs(angle) < 0.5:
        return roi, None
    if abs(angle) > MAX_CORRECTABLE_ANGLE:
        return roi, None

    h, w = roi.shape[:2]
    center = (w / 2.0, h / 2.0)
    rot_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Compute new bounding dimensions so we don't clip corners
    cos = abs(rot_matrix[0, 0])
    sin = abs(rot_matrix[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))
    rot_matrix[0, 2] += (new_w / 2.0) - center[0]
    rot_matrix[1, 2] += (new_h / 2.0) - center[1]

    border_value = 255 if len(roi.shape) == 2 else (255, 255, 255)
    rotated = cv2.warpAffine(
        roi,
        rot_matrix,
        (new_w, new_h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_value,
    )

    return rotated, angle
