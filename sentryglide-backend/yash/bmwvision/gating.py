"""Classical CV gates that run before the classifier: empty reference, settle, occupancy, quality.

Everything here works on the rectified ROI (fixed mm per pixel), so thresholds given in
mm and mm^2 in the config mean the same thing on every installation.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .types import Occupancy


# ---------------------------------------------------------------- empty reference
@dataclass
class EmptyReference:
    bgr: np.ndarray                  # median colour of the empty chamber
    lab: np.ndarray                  # float32 Lab of the same
    depth: np.ndarray | None         # median depth (mm), 0 where mostly invalid
    depth_noise: np.ndarray | None   # robust per-pixel std (mm)
    depth_valid_frac: np.ndarray | None
    t: float = 0.0

    @classmethod
    def build(cls, bgrs: list[np.ndarray], depths: list[np.ndarray] | None, t: float = 0.0):
        bgr = np.median(np.stack(bgrs), axis=0).astype(np.uint8)
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        if not depths or depths[0] is None:
            return cls(bgr, lab, None, None, None, t)
        d = np.stack(depths).astype(np.float32)
        valid = d > 0
        dn = np.where(valid, d, np.nan)
        with np.errstate(all="ignore"):
            med = np.nanmedian(dn, axis=0)
            mad = np.nanmedian(np.abs(dn - med), axis=0) * 1.4826
        return cls(bgr, lab, np.nan_to_num(med, nan=0.0), np.nan_to_num(mad, nan=0.0),
                   valid.mean(axis=0).astype(np.float32), t)

    def save(self, path) -> None:
        arrays = {"bgr": self.bgr, "t": np.array(self.t)}
        if self.depth is not None:
            arrays.update(depth=self.depth, depth_noise=self.depth_noise, depth_valid_frac=self.depth_valid_frac)
        np.savez_compressed(path, **arrays)

    @classmethod
    def load(cls, path) -> "EmptyReference":
        z = np.load(path)
        bgr = z["bgr"]
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        if "depth" in z:
            return cls(bgr, lab, z["depth"], z["depth_noise"], z["depth_valid_frac"], float(z["t"]))
        return cls(bgr, lab, None, None, None, float(z["t"]))


# ---------------------------------------------------------------- settle
class SettleDetector:
    """Settled = k consecutive frame pairs in which almost no pixels changed.

    Uses the fraction of changed pixels rather than the mean difference: a small object
    moving 4 mm changes only its outline, which a whole-ROI mean dilutes below sensor noise.
    """

    def __init__(self, k_frames: int, rgb_delta: float, depth_delta_mm: float, max_changed_frac: float):
        self.k, self.rgb_delta, self.depth_delta = k_frames, rgb_delta, depth_delta_mm
        self.max_changed = max_changed_frac
        self.reset()

    @classmethod
    def from_config(cls, s: dict) -> "SettleDetector":
        return cls(s["k_frames"], s["rgb_delta"], s["depth_delta_mm"], s["max_changed_frac"])

    def reset(self):
        self._prev = None
        self.count = 0
        self.last = (float("nan"), float("nan"))

    def update(self, bgr: np.ndarray, depth: np.ndarray | None) -> bool:
        g = cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), None, fx=0.5, fy=0.5,
                       interpolation=cv2.INTER_AREA).astype(np.float32)
        d = None if depth is None else depth[::2, ::2].astype(np.float32)
        if self._prev is None:
            self._prev = (g, d)
            return False
        pg, pd = self._prev
        c_rgb = float((np.abs(g - pg) > self.rgb_delta).mean())
        c_dep = 0.0
        if d is not None and pd is not None:
            both = (d > 0) & (pd > 0)
            c_dep = float((np.abs(d - pd)[both] > self.depth_delta).mean()) if both.any() else 0.0
        self.last = (c_rgb, c_dep)
        self._prev = (g, d)
        still = c_rgb < self.max_changed and c_dep < self.max_changed
        self.count = self.count + 1 if still else 0
        return self.count >= self.k


# ---------------------------------------------------------------- occupancy
@dataclass
class OccupancyResult:
    state: Occupancy
    mask: np.ndarray                     # uint8 0/255, union of kept components
    height_mm: np.ndarray | None         # height above the empty plate (>= 0)
    depth_invalid: np.ndarray | None     # bool, depth invalid now but valid when empty
    component_areas_mm2: list[float] = field(default_factory=list)
    total_area_mm2: float = 0.0
    volume_mm3: float = 0.0
    cue_px: dict = field(default_factory=dict)
    reason: str | None = None


def _odd(n: float) -> int:
    n = max(1, int(round(n)))
    return n if n % 2 else n + 1


def analyze_occupancy(bgr: np.ndarray, depth: np.ndarray | None, ref: EmptyReference,
                      cfg: dict, mm_per_px: float) -> OccupancyResult:
    """Decide NONE / ONE / MULTIPLE / OUT_OF_POSITION from colour and depth change vs the empty chamber.

    Cues are OR-ed: colour change, height above the plate, and new depth holes. Depth alone
    fails on glass, polished metal and needles; colour alone fails on objects that match the
    background. Speckle is removed with an area filter rather than morphological opening,
    because opening erases a 1 to 3 px wide needle.
    """
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    dl = (lab - ref.lab)
    dist = np.sqrt(cfg["l_weight"] * dl[..., 0] ** 2 + dl[..., 1] ** 2 + dl[..., 2] ** 2)
    fg_rgb = dist > cfg["lab_thresh"]

    height = invalid_new = None
    fg = fg_rgb.copy()
    cue = {"rgb": int(fg_rgb.sum())}
    if depth is not None and ref.depth is not None:
        both = (depth > 0) & (ref.depth > 0)
        height = np.where(both, ref.depth - depth.astype(np.float32), 0.0).astype(np.float32)
        height = np.clip(height, 0, None)
        thr = np.maximum(cfg["height_min_mm"], cfg["height_k_noise"] * ref.depth_noise)
        fg_d = both & (height > thr)
        fg |= fg_d
        cue["depth"] = int(fg_d.sum())
        invalid_new = (depth <= 0) & (ref.depth_valid_frac > 0.9)
        if cfg["use_invalid_depth"]:
            fg |= invalid_new
            cue["invalid_depth"] = int(invalid_new.sum())

    m = fg.astype(np.uint8) * 255
    k = _odd(cfg["close_mm"] / mm_per_px)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)

    px_area = mm_per_px ** 2
    min_px = cfg["min_component_area_mm2"] / px_area
    margin = int(round(cfg["border_margin_mm"] / mm_per_px))
    H, W = m.shape
    keep, areas, touches = [], [], False
    for i in range(1, n):
        x, y, w, h, a = stats[i]
        if a < min_px:
            continue
        keep.append(i)
        areas.append(a * px_area)
        if x <= margin or y <= margin or x + w >= W - margin or y + h >= H - margin:
            touches = True
    mask = (np.isin(labels, keep).astype(np.uint8) * 255) if keep else np.zeros_like(m)
    total = float(sum(areas))
    vol = float(height[mask > 0].sum() * px_area) if height is not None else 0.0

    res = OccupancyResult(Occupancy.NONE, mask, height, invalid_new, areas, total, vol, cue)
    if not keep:
        return res
    if len(keep) >= 2:
        res.state, res.reason = Occupancy.MULTIPLE, f"{len(keep)}_components"
    elif touches:
        res.state, res.reason = Occupancy.OUT_OF_POSITION, "touches_roi_border"
    elif cfg.get("max_total_area_mm2") and total > cfg["max_total_area_mm2"]:
        res.state, res.reason = Occupancy.MULTIPLE, "area_above_single_item_bound"
    elif cfg.get("max_volume_mm3") and vol > cfg["max_volume_mm3"]:
        res.state, res.reason = Occupancy.MULTIPLE, "volume_above_single_item_bound"
    else:
        res.state = Occupancy.ONE
    return res


# ---------------------------------------------------------------- quality
@dataclass
class QualityResult:
    ok: bool
    sharpness: float
    saturated_frac: float
    motion: float
    reasons: list[str]


def assess_quality(bgr: np.ndarray, mask: np.ndarray, prev_bgr: np.ndarray | None,
                   cfg: dict, mm_per_px: float) -> QualityResult:
    """Sharpness, exposure and residual motion around the object.

    Sharpness is var(Laplacian) / var(intensity) in a band around the object outline. A plain
    Laplacian variance depends on object contrast and texture (a flat blue glove scores lower
    than a stained dressing at equal focus); normalising by intensity variance removes most of
    that. Very thin, low-contrast objects are dominated by sensor noise and read as sharp.
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_odd(2.0 / mm_per_px),) * 2)
    band = (cv2.dilate(mask, se) > 0) & ~(cv2.erode(mask, se) > 0)
    region = cv2.dilate(mask, se) > 0
    if not band.any():
        return QualityResult(False, 0.0, 0.0, 0.0, ["empty_region"])
    lap = cv2.Laplacian(gray, cv2.CV_32F)
    sharp = float(lap[band].var() / (gray[band].var() + 1e-6))
    sat = float((bgr[region] >= 250).all(axis=1).mean())
    motion = 0.0
    if prev_bgr is not None:
        pg = cv2.cvtColor(prev_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        motion = float(np.abs(gray - pg)[region].mean())
    reasons = []
    if sharp < cfg["min_sharpness"]:
        reasons.append("blur")
    if sat > cfg["max_saturated_frac"]:
        reasons.append("saturation")
    if motion > cfg["max_motion"]:
        reasons.append("motion")
    return QualityResult(not reasons, sharp, sat, motion, reasons)
