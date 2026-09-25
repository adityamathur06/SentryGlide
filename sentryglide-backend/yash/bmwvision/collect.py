"""Placement-based data collection on the real rig.

Unit of data = one placement: one drop of one physical item. Each placement stores a few
rectified frames (the exact input the classifier sees at runtime), depth, mask and geometric
features, and one manifest row per frame carrying the physical item id. Train / validation
splits must be made by physical_id, never by frame or placement.

Folder layout:  <out>/<type>/<physical_id>/<placement_id>/rgb_00.png, depth_00.png, mask.png
Manifest:       <out>/manifest.csv
"""
from __future__ import annotations

import csv
import time
from pathlib import Path

import cv2
import numpy as np

from .calibration import Calibration
from .features import GEOM_FEATURE_NAMES, geometric_features
from .gating import EmptyReference, SettleDetector, analyze_occupancy
from .hardware import Camera
from .types import Occupancy

MANIFEST_FIELDS = ["rgb", "depth", "mask", "type", "physical_id", "placement_id", "frame",
                   "occupancy", "t"] + GEOM_FEATURE_NAMES

# Labels that are expected NOT to be a single valid item.
SPECIAL_LABELS = {"empty": Occupancy.NONE, "multiple": Occupancy.MULTIPLE}


def _next_placement(folder: Path, physical_id: str) -> str:
    n = len([p for p in folder.glob(f"{physical_id}_p*") if p.is_dir()]) if folder.exists() else 0
    return f"{physical_id}_p{n:03d}"


def collect_placement(camera: Camera, calib: Calibration, ref: EmptyReference, cfg: dict,
                      out_root: str | Path, type_name: str, physical_id: str) -> dict:
    out_root = Path(out_root)
    s, occ_cfg, ccfg = cfg["settle"], cfg["occupancy"], cfg["collect"]
    mpp = calib.mm_per_px
    settle = SettleDetector.from_config(s)

    def read():
        f = camera.read()
        return calib.rectify(f.bgr), (None if f.depth is None else calib.rectify(f.depth, nearest=True))

    t0 = time.monotonic()
    while True:
        bgr, depth = read()
        if settle.update(bgr, depth):
            break
        if time.monotonic() - t0 > s["timeout_s"]:
            return {"saved": False, "reason": "settle_timeout"}

    occ = analyze_occupancy(bgr, depth, ref, occ_cfg, mpp)
    expected = SPECIAL_LABELS.get(type_name, Occupancy.ONE)
    if occ.state != expected:
        return {"saved": False, "reason": f"occupancy_{occ.state.value}_expected_{expected.value}",
                "areas_mm2": [round(a, 1) for a in occ.component_areas_mm2]}

    folder = out_root / type_name / physical_id
    pid = _next_placement(folder, physical_id)
    d = folder / pid
    d.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(d / "mask.png"), occ.mask)

    manifest = out_root / "manifest.csv"
    new = not manifest.exists()
    rows = []
    last_t = 0.0
    for i in range(ccfg["frames_per_placement"]):
        while time.monotonic() - last_t < ccfg["min_gap_s"]:
            time.sleep(0.01)
        if i > 0:
            bgr, depth = read()
            occ = analyze_occupancy(bgr, depth, ref, occ_cfg, mpp)
        last_t = time.monotonic()
        cv2.imwrite(str(d / f"rgb_{i:02d}.png"), bgr)
        dpath = ""
        if depth is not None:
            dpath = str((d / f"depth_{i:02d}.png").relative_to(out_root))
            cv2.imwrite(str(out_root / dpath), np.clip(depth, 0, 65535).astype(np.uint16))
        g = geometric_features(occ.mask, occ.height_mm, occ.depth_invalid, mpp)
        rows.append({
            "rgb": str((d / f"rgb_{i:02d}.png").relative_to(out_root)), "depth": dpath,
            "mask": str((d / "mask.png").relative_to(out_root)), "type": type_name,
            "physical_id": physical_id, "placement_id": pid, "frame": i,
            "occupancy": occ.state.value, "t": round(time.time(), 3),
            **{k: round(float(v), 4) for k, v in zip(GEOM_FEATURE_NAMES, g)},
        })
    with open(manifest, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        if new:
            w.writeheader()
        w.writerows(rows)
    return {"saved": True, "placement_id": pid, "frames": len(rows), "dir": str(d),
            "area_mm2": round(rows[-1]["area_mm2"], 1)}
