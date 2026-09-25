"""Engineered geometric features from the object mask and depth (late-fusion inputs).

About ten numbers, concatenated with the CNN embedding before the final classifier head.
They cost nothing to compute and leave the pretrained RGB backbone untouched.
"""
from __future__ import annotations

import cv2
import numpy as np

GEOM_FEATURE_NAMES = [
    "area_mm2", "length_mm", "width_mm", "elongation",
    "max_height_mm", "mean_height_mm", "p90_height_mm", "height_std_mm",
    "volume_mm3", "invalid_depth_frac",
]


def geometric_features(mask: np.ndarray, height_mm: np.ndarray | None,
                       depth_invalid: np.ndarray | None, mm_per_px: float) -> np.ndarray:
    f = np.zeros(len(GEOM_FEATURE_NAMES), dtype=np.float32)
    obj = mask > 0
    n = int(obj.sum())
    if n == 0:
        return f
    px_area = mm_per_px ** 2
    f[0] = n * px_area
    pts = cv2.findNonZero(mask)
    (_, _), (w, h), _ = cv2.minAreaRect(pts)
    length, width = max(w, h) * mm_per_px, max(min(w, h), 1.0) * mm_per_px
    f[1], f[2], f[3] = length, width, length / width
    if height_mm is not None:
        hv = height_mm[obj]
        valid = hv[hv > 0]
        if valid.size:
            f[4], f[5] = valid.max(), valid.mean()
            f[6], f[7] = np.percentile(valid, 90), valid.std()
        f[8] = float(hv.sum() * px_area)
    if depth_invalid is not None:
        f[9] = float(depth_invalid[obj].mean())
    return f
