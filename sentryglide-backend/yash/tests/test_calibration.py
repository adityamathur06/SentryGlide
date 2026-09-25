import cv2
import numpy as np

from bmwvision.calibration import Calibration, marker_drift, resolution_report
from bmwvision.features import geometric_features
from bmwvision.gating import analyze_occupancy
from bmwvision.sim import make_item
from conftest import rect


def test_homography_accuracy(calib, cfg):
    assert calib.reprojection_error_mm < 0.2
    assert len(calib.marker_corners_px) == 4
    assert calib.out_size == (600, 600)
    rep = resolution_report(calib, cfg["geometry"])
    assert rep["camera_ok"] and rep["roi_ok"]


def test_rectified_roi_has_physical_scale(rig, calib, ref, cfg):
    """A 100 mm x 4 mm tube must measure ~100 mm in the rectified ROI wherever it lies."""
    scene, cam = rig
    for center, angle in [((130, 100), 0), ((110, 80), 35), ((150, 120), 90)]:
        scene.items = [make_item("iv_set_tubing", center, angle)]
        bgr, depth = rect(calib, cam.read())
        occ = analyze_occupancy(bgr, depth, ref, cfg["occupancy"], calib.mm_per_px)
        f = geometric_features(occ.mask, occ.height_mm, occ.depth_invalid, calib.mm_per_px)
        assert abs(f[1] - 100) < 3, f[1]


def test_save_load_roundtrip(calib, tmp_path):
    calib.save(tmp_path / "c.json")
    c2 = Calibration.load(tmp_path / "c.json")
    assert np.allclose(c2.H, calib.H) and c2.roi_mm == calib.roi_mm


def test_marker_drift_detects_camera_movement(rig, calib, cfg):
    scene, cam = rig
    d = cfg["geometry"]["markers"]["dictionary"]
    drift, seen = marker_drift(cam.read().bgr, calib, d)
    assert seen == 4 and drift < 1.0
    cam.H_tex_to_img = np.array([[1, 0, 6], [0, 1, 4], [0, 0, 1]]) @ cam.H_tex_to_img  # bump camera
    drift, _ = marker_drift(cam.read().bgr, calib, d)
    assert drift > cfg["geometry"]["max_marker_drift_px"]
