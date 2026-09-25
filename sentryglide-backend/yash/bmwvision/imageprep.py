"""Shared preparation for uploaded and imported single-item photographs."""
from __future__ import annotations

import cv2
import numpy as np


def square_inspection_view(image: np.ndarray, side: int = 600) -> np.ndarray:
    h, w = image.shape[:2]
    scale = min(side / w, side / h)
    resized = cv2.resize(image, (max(1, round(w * scale)), max(1, round(h * scale))),
                         interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    canvas = np.full((side, side, 3), 150, np.uint8)
    y = (side - resized.shape[0]) // 2
    x = (side - resized.shape[1]) // 2
    canvas[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
    return canvas


def foreground_mask(image: np.ndarray) -> tuple[np.ndarray, float]:
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    border_px = max(3, min(image.shape[:2]) // 50)
    border = np.concatenate((lab[:border_px].reshape(-1, 3), lab[-border_px:].reshape(-1, 3),
                             lab[:, :border_px].reshape(-1, 3), lab[:, -border_px:].reshape(-1, 3)))
    background = np.median(border, axis=0)
    distance = np.linalg.norm(lab - background, axis=2)
    mask = (distance > 22).astype(np.uint8) * 255
    k = max(3, round(min(image.shape[:2]) / 70))
    if k % 2 == 0:
        k += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    if n > 1:
        _, best = max((stats[i, cv2.CC_STAT_AREA], i) for i in range(1, n))
        mask = np.where(labels == best, 255, 0).astype(np.uint8)
    return mask, float((mask > 0).mean())


def crop_bbox(image: np.ndarray, bbox, padding: float = 0.08) -> np.ndarray:
    x, y, w, h = [float(v) for v in bbox]
    pad = padding * max(w, h)
    x0, y0 = max(0, int(x - pad)), max(0, int(y - pad))
    x1 = min(image.shape[1], int(np.ceil(x + w + pad)))
    y1 = min(image.shape[0], int(np.ceil(y + h + pad)))
    return image[y0:y1, x0:x1]


def prepare_upload(image: np.ndarray, side: int = 600) -> tuple[np.ndarray, np.ndarray, float]:
    first = square_inspection_view(image, side)
    mask, fraction = foreground_mask(first)
    points = cv2.findNonZero(mask)
    if points is None or fraction < 0.001:
        return first, mask, fraction
    x, y, w, h = cv2.boundingRect(points)
    crop = crop_bbox(first, (x, y, w, h), 0.10)
    view = square_inspection_view(crop, side)
    mask, fraction = foreground_mask(view)
    return view, mask, fraction

