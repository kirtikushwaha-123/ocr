"""
preprocessing/perspective.py

Conditional perspective correction. We only warp when there is strong
geometric evidence of a quadrilateral with real perspective distortion
(i.e. the four corners are clearly non-rectangular and the region we
found is large enough to trust). Applying perspective correction
blindly on small text-crops does more harm than good, so this is
intentionally conservative.
"""

import cv2
import numpy as np


MIN_AREA_RATIO_FOR_CORRECTION = 0.35
MIN_CORNER_DISTORTION_PX = 8


def _order_points(pts):
    """
    Orders 4 points as: top-left, top-right, bottom-right, bottom-left.
    """
    pts = np.array(pts, dtype="float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).reshape(-1)

    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]

    return np.array([tl, tr, br, bl], dtype="float32")


def find_quadrilateral(roi):
    """
    Attempts to find a strong 4-point contour (e.g. a printed label edge)
    within the ROI. Returns ordered 4 points or None.
    """
    gray = roi if len(roi.shape) == 2 else cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 40, 120)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    image_area = roi.shape[0] * roi.shape[1]
    best_quad = None
    best_area = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < image_area * MIN_AREA_RATIO_FOR_CORRECTION:
            continue
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 4 and area > best_area:
            best_quad = approx.reshape(4, 2)
            best_area = area

    return best_quad


def _has_real_distortion(pts):
    """
    Checks whether the quadrilateral deviates enough from an axis-aligned
    rectangle to justify a perspective warp.
    """
    ordered = _order_points(pts)
    tl, tr, br, bl = ordered

    top_diff = abs(tl[1] - tr[1])
    bottom_diff = abs(bl[1] - br[1])
    left_diff = abs(tl[0] - bl[0])
    right_diff = abs(tr[0] - br[0])

    max_diff = max(top_diff, bottom_diff, left_diff, right_diff)
    return max_diff >= MIN_CORNER_DISTORTION_PX


def correct_perspective(roi):
    """
    Attempts a perspective correction on roi. Returns (corrected_roi, applied: bool).
    If no strong quadrilateral evidence is found, returns the original ROI
    unchanged with applied=False.
    """
    if roi is None or roi.size == 0:
        return roi, False

    quad = find_quadrilateral(roi)
    if quad is None:
        return roi, False

    if not _has_real_distortion(quad):
        return roi, False

    ordered = _order_points(quad)
    (tl, tr, br, bl) = ordered

    width_a = np.linalg.norm(br - bl)
    width_b = np.linalg.norm(tr - tl)
    max_width = max(int(width_a), int(width_b))

    height_a = np.linalg.norm(tr - br)
    height_b = np.linalg.norm(tl - bl)
    max_height = max(int(height_a), int(height_b))

    if max_width < 10 or max_height < 10:
        return roi, False

    dst = np.array(
        [
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1],
        ],
        dtype="float32",
    )

    transform_matrix = cv2.getPerspectiveTransform(ordered, dst)
    warped = cv2.warpPerspective(roi, transform_matrix, (max_width, max_height))

    return warped, True
