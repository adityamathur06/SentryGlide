import numpy as np
import pytest

from bmwvision.features import GEOM_FEATURE_NAMES, geometric_features
from bmwvision.gating import SettleDetector, analyze_occupancy, assess_quality
from bmwvision.sim import make_item
from bmwvision.types import Occupancy
from conftest import rect


def occ_of(rig, calib, ref, cfg, items, blur=0.0):
    scene, cam = rig
    scene.items, scene.blur_sigma = items, blur
    bgr, depth = rect(calib, cam.read())
    return analyze_occupancy(bgr, depth, ref, cfg["occupancy"], calib.mm_per_px), bgr


def test_empty_chamber_is_none(rig, calib, ref, cfg):
    occ, _ = occ_of(rig, calib, ref, cfg, [])
    assert occ.state == Occupancy.NONE


@pytest.mark.parametrize("t", ["needle", "gloves", "glass_vial", "metal_implant", "soiled_gauze_dressing",
                               "syringe_with_fixed_needle", "iv_set_tubing"])
def test_single_items_are_one(rig, calib, ref, cfg, t):
    occ, _ = occ_of(rig, calib, ref, cfg, [make_item(t, (128, 98), 20)])
    assert occ.state == Occupancy.ONE, (t, occ.reason, occ.component_areas_mm2)


def test_needle_shaft_is_kept_in_mask(rig, calib, ref, cfg):
    """Area filtering instead of opening: the 0.5 mm shaft must survive (length ~42 mm, not the 7 mm hub)."""
    occ, _ = occ_of(rig, calib, ref, cfg, [make_item("needle", (120, 100), 0)])
    f = geometric_features(occ.mask, occ.height_mm, occ.depth_invalid, calib.mm_per_px)
    assert f[GEOM_FEATURE_NAMES.index("length_mm")] > 38


def test_glass_detected_from_depth_holes(rig, calib, ref, cfg):
    occ, _ = occ_of(rig, calib, ref, cfg, [make_item("glass_vial")])
    assert occ.cue_px["invalid_depth"] > 1000
    f = geometric_features(occ.mask, occ.height_mm, occ.depth_invalid, calib.mm_per_px)
    assert f[GEOM_FEATURE_NAMES.index("invalid_depth_frac")] > 0.5


def test_two_items_are_multiple(rig, calib, ref, cfg):
    occ, _ = occ_of(rig, calib, ref, cfg, [make_item("gloves", (100, 100)), make_item("needle", (160, 140), 90)])
    assert occ.state == Occupancy.MULTIPLE


def test_item_at_border_is_out_of_position(rig, calib, ref, cfg):
    occ, _ = occ_of(rig, calib, ref, cfg, [make_item("gloves", (62, 100))])
    assert occ.state == Occupancy.OUT_OF_POSITION


def test_area_bound_flags_touching_pair(rig, calib, ref, cfg):
    """Two overlapping items form one blob; the single-item area bound still catches them."""
    single, _ = occ_of(rig, calib, ref, cfg, [make_item("gloves", (110, 100))])
    c = dict(cfg["occupancy"], max_total_area_mm2=1.2 * single.total_area_mm2)
    scene, cam = rig
    for items, expected in [([make_item("gloves", (110, 100))], Occupancy.ONE),
                            ([make_item("gloves", (105, 92)), make_item("gloves", (120, 112))], Occupancy.MULTIPLE)]:
        scene.items = items
        bgr, depth = rect(calib, cam.read())
        occ = analyze_occupancy(bgr, depth, ref, c, calib.mm_per_px)
        assert occ.state == expected, (occ.state, occ.reason, occ.component_areas_mm2)


@pytest.mark.parametrize("t", ["gloves", "soiled_gauze_dressing", "glass_vial", "metal_implant"])
def test_quality_sharp_vs_blurred(rig, calib, ref, cfg, t):
    occ, bgr = occ_of(rig, calib, ref, cfg, [make_item(t)])
    assert assess_quality(bgr, occ.mask, None, cfg["quality"], calib.mm_per_px).ok
    occ, bgr = occ_of(rig, calib, ref, cfg, [make_item(t)], blur=2.0)
    q = assess_quality(bgr, occ.mask, None, cfg["quality"], calib.mm_per_px)
    assert not q.ok and "blur" in q.reasons


def test_settle_detector(rig, calib, cfg):
    scene, cam = rig
    s = cfg["settle"]
    sd = SettleDetector.from_config(s)
    scene.items = [make_item("gloves")]
    scene.jitter_frames, scene.jitter_mm = 4, 4.0
    settled_at = None
    for i in range(20):
        if sd.update(*rect(calib, cam.read())):
            settled_at = i
            break
    assert settled_at is not None and settled_at >= 4 + s["k_frames"] - 1
