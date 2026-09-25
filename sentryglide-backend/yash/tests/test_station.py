import json

import numpy as np
import pytest

from bmwvision.sim import Delivery, SimulatedController, make_item
from bmwvision.station import RunLogger, Station
from bmwvision.types import Outcome


def make_station(cfg, rig, calib, tax, sim_classifier, deliveries, tmp_path=None, **ctrl_kw):
    scene, cam = rig
    ctrl = SimulatedController(scene, deliveries, **ctrl_kw)
    clf, ood = sim_classifier(scene)
    logger = RunLogger(tmp_path) if tmp_path else None
    return Station(cfg, cam, ctrl, calib, clf, tax, ood, logger), ctrl


EXPECTED = {"needle": "WHITE", "syringe_with_fixed_needle": "WHITE", "gloves": "RED",
            "iv_set_tubing": "RED", "soiled_gauze_dressing": "YELLOW", "glass_vial": "BLUE",
            "metal_implant": "BLUE"}


def test_routes_each_bin(cfg, rig, calib, tax, sim_classifier, tmp_path):
    dels = [Delivery(t, [make_item(t, (130, 100), 30)]) for t in EXPECTED]
    st, ctrl = make_station(cfg, rig, calib, tax, sim_classifier, dels, tmp_path)
    recs = list(st.run(stop_when_idle=True))
    got = {r.item_id: r.outcome for r in recs}
    assert got == EXPECTED
    assert [c[1] for c in ctrl.commands] == list(EXPECTED.values())
    assert all(r.frames < cfg["capture"]["max_frames"] for r in recs)   # stopped early once accepted
    lines = (tmp_path / "records.jsonl").read_text().splitlines()
    assert len(lines) == len(EXPECTED) and json.loads(lines[0])["trace"][-1] == "WAIT_FOR_OBJECT"
    assert (tmp_path / "items" / "needle" / "rgb_00.png").exists()


def test_unknown_object_is_held_as_ood(cfg, rig, calib, tax, sim_classifier):
    st, ctrl = make_station(cfg, rig, calib, tax, sim_classifier, [Delivery("x", [make_item("phone")])])
    rec = st.run_cycle()
    assert rec.outcome == "UNKNOWN" and rec.reason == "out_of_distribution"
    assert ctrl.commands[-1][1] == "HOLD"


@pytest.mark.parametrize("items,reason", [
    ([make_item("gloves", (100, 100)), make_item("needle", (160, 140), 90)], "occupancy_MULTIPLE"),
    ([make_item("gloves", (62, 100))], "occupancy_OUT_OF_POSITION"),
])
def test_bad_occupancy_is_held(cfg, rig, calib, tax, sim_classifier, items, reason):
    st, ctrl = make_station(cfg, rig, calib, tax, sim_classifier, [Delivery("x", items)])
    rec = st.run_cycle()
    assert rec.outcome == "UNKNOWN" and rec.reason.startswith(reason)
    assert ctrl.commands == [("x", "HOLD", rec.summary())]


def test_confusable_pair_is_held(cfg, rig, calib, tax, sim_classifier):
    scene, _ = rig
    st, ctrl = make_station(cfg, rig, calib, tax, lambda s: sim_classifier(
        s, confusion={"syringe_without_needle": ("syringe_with_fixed_needle", 7.0)}),
        [Delivery("x", [make_item("syringe_without_needle")])])
    rec = st.run_cycle()
    assert rec.outcome == "UNKNOWN" and rec.frames == cfg["capture"]["max_frames"]


@pytest.mark.parametrize("delivery,kw,reason", [
    (Delivery("x", []), {}, "delivered_but_chamber_empty"),
    (Delivery("x", [make_item("needle")], stuck=True), {}, "chamber_not_empty_after_drop"),
    (Delivery("x", [make_item("needle")]), {"ack": False}, "no_controller_ack"),
])
def test_faults(cfg, rig, calib, tax, sim_classifier, delivery, kw, reason):
    st, ctrl = make_station(cfg, rig, calib, tax, sim_classifier, [delivery], **kw)
    rec = st.run_cycle()
    assert rec.outcome == "FAULT" and rec.reason == reason
    assert st.faulted == reason and ctrl.faults == [reason]
    with pytest.raises(RuntimeError):
        st.run_cycle()


def test_camera_moved_faults_while_idle(cfg, rig, calib, tax, sim_classifier):
    scene, cam = rig
    st, ctrl = make_station(cfg, rig, calib, tax, sim_classifier, [])
    st.build_reference()
    cam.H_tex_to_img = np.array([[1, 0, 8], [0, 1, 0], [0, 0, 1]]) @ cam.H_tex_to_img
    st._last_marker_check = -1e9
    rec = st.run_cycle()
    assert rec.outcome == "FAULT" and rec.reason.startswith("camera_moved")


def test_blurred_item_held_for_quality(cfg, rig, calib, tax, sim_classifier):
    scene, _ = rig
    st, ctrl = make_station(cfg, rig, calib, tax, sim_classifier, [Delivery("x", [make_item("gloves")])])
    st.build_reference()
    scene.blur_sigma = 2.5
    rec = st.run_cycle()
    assert rec.outcome == "UNKNOWN" and rec.reason.startswith("image_quality")
