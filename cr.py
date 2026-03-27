"""Cosmic-ray streak detection helpers."""

from __future__ import annotations

import numpy as np
from skimage.feature import canny
from skimage.draw import line as draw_line
from skimage.transform import probabilistic_hough_line
from scipy.ndimage import binary_dilation, gaussian_filter


def detect_streak_mask(
    image: np.ndarray,
    sigma_bg: float = 12.0,
    canny_sigma: float = 2.0,
    line_length: int = 150,
    line_gap: int = 8,
    dilate_iter: int = 12,
    threshold_sigma: float = 3.0,
) -> tuple[np.ndarray, list[tuple[tuple[int, int], tuple[int, int]]]]:
    """Detect long bright streaks and return ``(mask, line_segments)``."""
    img = np.asarray(image, dtype=float)

    bg = gaussian_filter(img, sigma=sigma_bg)
    hp = img - bg

    med = np.median(hp)
    mad = np.median(np.abs(hp - med))
    robust_sigma = 1.4826 * mad if mad > 0 else np.std(hp)

    bright = hp > (med + threshold_sigma * robust_sigma)
    edges = canny(bright.astype(float), sigma=canny_sigma)
    lines = probabilistic_hough_line(
        edges,
        threshold=7,
        line_length=line_length,
        line_gap=line_gap,
    )

    mask = np.zeros_like(img, dtype=bool)
    for (x0, y0), (x1, y1) in lines:
        rr, cc = draw_line(y0, x0, y1, x1)
        mask[rr, cc] = True

    mask = binary_dilation(mask, iterations=dilate_iter)

    return mask, lines
