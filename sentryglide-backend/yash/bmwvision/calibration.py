"""Installation calibration: plate homography from ArUco markers, ROI rectification, drift check.

Coordinates: plate millimetres with x to the right and y down, origin at the plate's
top-left corner. The ROI is defined in plate mm, so the crop does not depend on where
the camera happens to be mounted.

The rectified ROI has a fixed physical scale (mm_per_px). Objects are never rescaled to
fill the crop, so apparent size equals physical size and is a usable feature.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np


def _dictionary(name: str):
    return cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, name))


def detect_markers(bgr: np.ndarray, dictionary: str) -> dict[int, np.ndarray]:
    """Return {marker_id: 4x2 corner array (TL, TR, BR, BL) in image px}."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if bgr.ndim == 3 else bgr
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    detector = cv2.aruco.ArucoDetector(_dictionary(dictionary), params)
    corners, ids, _ = detector.detectMarkers(gray)
    if ids is None:
        return {}
    return {int(i): c.reshape(4, 2).astype(np.float64) for i, c in zip(ids.ravel(), corners)}


def marker_plate_corners(pos_mm, size_mm) -> np.ndarray:
    x, y = pos_mm
    s = size_mm
    return np.array([[x, y], [x + s, y], [x + s, y + s], [x, y + s]], dtype=np.float64)


@dataclass
class Calibration:
    H: np.ndarray                    # 3x3, image px -> plate mm
    roi_mm: tuple[float, float, float, float]
    mm_per_px: float
    image_size: tuple[int, int]      # (w, h) of the camera image
    marker_corners_px: dict[int, np.ndarray] = field(default_factory=dict)
    reprojection_error_mm: float = 0.0

    # ---- geometry ----
    @property
    def out_size(self) -> tuple[int, int]:
        x0, y0, x1, y1 = self.roi_mm
        return (int(round((x1 - x0) / self.mm_per_px)), int(round((y1 - y0) / self.mm_per_px)))

    @property
    def px_area_mm2(self) -> float:
        return self.mm_per_px ** 2

    def image_to_roi(self) -> np.ndarray:
        x0, y0 = self.roi_mm[:2]
        s = 1.0 / self.mm_per_px
        A = np.array([[s, 0, -x0 * s], [0, s, -y0 * s], [0, 0, 1]], dtype=np.float64)
        return A @ self.H

    def rectify(self, img: np.ndarray, nearest: bool = False) -> np.ndarray:
        """Warp a camera image (colour or depth) to the top-down ROI at fixed mm/px.
        Use nearest=True for depth so invalid (0) pixels are not blended into valid ones."""
        flags = cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR
        return cv2.warpPerspective(img, self.image_to_roi(), self.out_size, flags=flags,
                                   borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    def native_mm_per_px(self) -> float:
        """Coarsest camera sampling (mm per camera pixel) at the ROI centre.
        If this is larger than mm_per_px, the rectified ROI is upsampled and fine
        detail (needles) is not really there."""
        x0, y0, x1, y1 = self.roi_mm
        c_plate = np.array([[(x0 + x1) / 2, (y0 + y1) / 2]], dtype=np.float64)
        Hinv = np.linalg.inv(self.H)
        c_img = cv2.perspectiveTransform(c_plate[None], Hinv)[0, 0]
        pts = np.array([[c_img, c_img + [1, 0], c_img + [0, 1]]], dtype=np.float64)
        p = cv2.perspectiveTransform(pts, self.H)[0]
        return float(max(np.linalg.norm(p[1] - p[0]), np.linalg.norm(p[2] - p[0])))

    # ---- persistence ----
    def save(self, path: str | Path) -> None:
        d = {
            "H": self.H.tolist(), "roi_mm": list(self.roi_mm), "mm_per_px": self.mm_per_px,
            "image_size": list(self.image_size),
            "marker_corners_px": {str(k): v.tolist() for k, v in self.marker_corners_px.items()},
            "reprojection_error_mm": self.reprojection_error_mm,
        }
        Path(path).write_text(json.dumps(d, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "Calibration":
        d = json.loads(Path(path).read_text())
        return cls(
            H=np.array(d["H"], dtype=np.float64), roi_mm=tuple(d["roi_mm"]),
            mm_per_px=float(d["mm_per_px"]), image_size=tuple(d["image_size"]),
            marker_corners_px={int(k): np.array(v) for k, v in d["marker_corners_px"].items()},
            reprojection_error_mm=float(d.get("reprojection_error_mm", 0.0)),
        )


def calibrate(bgr: np.ndarray, geom_cfg: dict, min_markers: int = 3) -> Calibration:
    """Compute the image -> plate homography from the fixed markers on the plate."""
    m = geom_cfg["markers"]
    found = detect_markers(bgr, m["dictionary"])
    used = [i for i in m["positions_mm"] if i in found]
    if len(used) < min_markers:
        raise RuntimeError(f"Calibration needs >= {min_markers} markers, found {sorted(found)}; "
                           f"expected ids {sorted(m['positions_mm'])}")
    img_pts = np.concatenate([found[i] for i in used])
    plate_pts = np.concatenate([marker_plate_corners(m["positions_mm"][i], m["size_mm"]) for i in used])
    H, _ = cv2.findHomography(img_pts, plate_pts, 0)
    proj = cv2.perspectiveTransform(img_pts[None], H)[0]
    err = float(np.linalg.norm(proj - plate_pts, axis=1).mean())
    h, w = bgr.shape[:2]
    return Calibration(H=H, roi_mm=tuple(geom_cfg["roi_mm"]), mm_per_px=float(geom_cfg["mm_per_px"]),
                       image_size=(w, h), marker_corners_px={i: found[i] for i in used},
                       reprojection_error_mm=err)


def resolution_report(calib: Calibration, geom_cfg: dict) -> dict:
    """Check that the thinnest safety-critical feature is resolved by both the camera and the ROI."""
    need = geom_cfg["min_feature_mm"] / geom_cfg["min_px_across_feature"]
    native = calib.native_mm_per_px()
    return {
        "required_mm_per_px": round(need, 4),
        "camera_native_mm_per_px": round(native, 4),
        "roi_mm_per_px": calib.mm_per_px,
        "roi_size_px": calib.out_size,
        "camera_ok": native <= need,
        "roi_ok": calib.mm_per_px <= need,
    }


def marker_drift(bgr: np.ndarray, calib: Calibration, dictionary: str) -> tuple[float, int]:
    """Max corner displacement (px) of calibration markers now visible, and how many were seen."""
    found = detect_markers(bgr, dictionary)
    drifts = [float(np.linalg.norm(found[i] - c, axis=1).max())
              for i, c in calib.marker_corners_px.items() if i in found]
    return (max(drifts) if drifts else float("inf")), len(drifts)


def make_marker_board(geom_cfg: dict, px_per_mm: float, background: int = 150) -> np.ndarray:
    """Render the plate with its markers (BGR). Print at px_per_mm * 25.4 DPI, 100% scale,
    and check marker size with a ruler. Also used by the simulator as the plate texture."""
    pw, ph = geom_cfg["plate_mm"]
    img = np.full((int(round(ph * px_per_mm)), int(round(pw * px_per_mm)), 3), background, np.uint8)
    m = geom_cfg["markers"]
    side = int(round(m["size_mm"] * px_per_mm))
    quiet = max(2, int(round(2 * px_per_mm)))  # white quiet zone around each marker
    for mid, (x, y) in m["positions_mm"].items():
        mk = cv2.aruco.generateImageMarker(_dictionary(m["dictionary"]), mid, side)
        x0, y0 = int(round(x * px_per_mm)), int(round(y * px_per_mm))
        img[max(0, y0 - quiet):y0 + side + quiet, max(0, x0 - quiet):x0 + side + quiet] = 255
        img[y0:y0 + side, x0:x0 + side] = mk[..., None]
    return img
