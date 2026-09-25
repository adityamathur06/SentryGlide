"""Inspection station state machine.

WAIT_FOR_OBJECT -> SETTLE -> CHECK_OCCUPANCY -> CAPTURE_AND_CLASSIFY -> DECIDE
   -> SEND_BIN_COMMAND -> WAIT_FOR_DROP_CONFIRMATION -> LOG -> WAIT_FOR_OBJECT
Side exits: HOLD (item goes to the quarantine position, then LOG) and FAULT (stop, manual reset).

Everything uncertain resolves to HOLD. Everything that suggests the machine itself is wrong
(camera moved, item vanished, chamber not empty after a drop, no ack) resolves to FAULT.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path

import cv2
import numpy as np

from .calibration import Calibration, marker_drift
from .classifier import KnnOOD, TypeClassifier, softmax
from .decision import CostModel, Decision, TemporalAggregator, decide_item
from .features import GEOM_FEATURE_NAMES, geometric_features
from .gating import EmptyReference, OccupancyResult, SettleDetector, analyze_occupancy, assess_quality
from .hardware import Camera, Controller
from .taxonomy import Taxonomy
from .types import ROWS, Occupancy, Outcome


class State(str, Enum):
    WAIT_FOR_OBJECT = "WAIT_FOR_OBJECT"
    SETTLE = "SETTLE"
    CHECK_OCCUPANCY = "CHECK_OCCUPANCY"
    CAPTURE_AND_CLASSIFY = "CAPTURE_AND_CLASSIFY"
    DECIDE = "DECIDE"
    SEND_BIN_COMMAND = "SEND_BIN_COMMAND"
    WAIT_FOR_DROP_CONFIRMATION = "WAIT_FOR_DROP_CONFIRMATION"
    LOG = "LOG"
    HOLD = "HOLD"
    FAULT = "FAULT"


class StationFault(Exception):
    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


@dataclass
class CycleRecord:
    item_id: str
    outcome: str = ""
    p: float = 0.0
    frames: int = 0
    ood: float | None = None
    reason: str | None = None
    occupancy: str | None = None
    metal: bool | None = None
    row_probs: dict = field(default_factory=dict)
    expected_cost: dict = field(default_factory=dict)
    top_types: list = field(default_factory=list)
    geom: dict = field(default_factory=dict)
    trace: list = field(default_factory=list)
    t_start: float = 0.0
    duration_s: float = 0.0

    def summary(self) -> dict:
        """Compact message for the controller and operator display."""
        return {"item_id": self.item_id, "class": self.outcome, "p": round(self.p, 4),
                "frames": self.frames, "ood": None if self.ood is None else round(self.ood, 4),
                "reason": self.reason}


class RunLogger:
    """Appends one JSON line per cycle; optionally saves the rectified frames and mask.
    Held items' frames are the most valuable future training data."""

    def __init__(self, out_dir: str | Path, save_frames: bool = True):
        self.dir = Path(out_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.save_frames = save_frames

    def log(self, rec: CycleRecord, frames: list[tuple[np.ndarray, np.ndarray | None]],
            mask: np.ndarray | None) -> None:
        with open(self.dir / "records.jsonl", "a") as f:
            f.write(json.dumps(asdict(rec), default=_json_default) + "\n")
        if self.save_frames and frames:
            d = self.dir / "items" / str(rec.item_id)
            d.mkdir(parents=True, exist_ok=True)
            for i, (bgr, depth) in enumerate(frames):
                cv2.imwrite(str(d / f"rgb_{i:02d}.png"), bgr)
                if depth is not None:
                    cv2.imwrite(str(d / f"depth_{i:02d}.png"), np.clip(depth, 0, 65535).astype(np.uint16))
            if mask is not None:
                cv2.imwrite(str(d / "mask.png"), mask)


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Enum):
        return o.value
    raise TypeError(type(o))


class Station:
    def __init__(self, cfg: dict, camera: Camera, controller: Controller, calib: Calibration,
                 classifier: TypeClassifier, taxonomy: Taxonomy, ood: KnnOOD | None = None,
                 logger: RunLogger | None = None, reference: EmptyReference | None = None):
        self.cfg, self.cam, self.ctrl, self.calib = cfg, camera, controller, calib
        self.clf, self.tax, self.ood, self.logger = classifier, taxonomy, ood, logger
        self.cost = CostModel.from_config(cfg["decision"])
        self.temperature = float(getattr(classifier.meta, "temperature", 1.0))
        s = cfg["settle"]
        self.settle = SettleDetector.from_config(s)
        self.ref = reference
        self.faulted: str | None = None
        now = time.monotonic()
        self._last_marker_check = now
        self._marker_misses = 0

    # ------------------------------------------------------------ helpers
    @property
    def mpp(self) -> float:
        return self.calib.mm_per_px

    def _read(self):
        f = self.cam.read()
        bgr = self.calib.rectify(f.bgr)
        depth = None if f.depth is None else self.calib.rectify(f.depth, nearest=True)
        return f, bgr, depth

    def _occupancy(self, bgr, depth) -> OccupancyResult:
        return analyze_occupancy(bgr, depth, self.ref, self.cfg["occupancy"], self.mpp)

    def _wait_settled(self, timeout: float):
        self.settle.reset()
        t0 = time.monotonic()
        while True:
            f, bgr, depth = self._read()
            if self.settle.update(bgr, depth):
                return f, bgr, depth
            if time.monotonic() - t0 > timeout:
                raise StationFault("settle_timeout")

    # ------------------------------------------------------------ reference and drift
    def build_reference(self) -> EmptyReference:
        """Capture the empty-chamber reference. Call only when the chamber is known to be empty."""
        n = self.cfg["reference"]["n_frames"]
        bgrs, depths = [], []
        for _ in range(n):
            _, bgr, depth = self._read()
            bgrs.append(bgr)
            depths.append(depth)
        self.ref = EmptyReference.build(bgrs, depths if depths[0] is not None else None, time.monotonic())
        return self.ref

    def check_markers(self) -> None:
        g = self.cfg["geometry"]
        f = self.cam.read()
        drift, seen = marker_drift(f.bgr, self.calib, g["markers"]["dictionary"])
        if seen == 0:
            self._marker_misses += 1
            if self._marker_misses >= 3:
                raise StationFault("calibration_markers_not_visible")
            return
        self._marker_misses = 0
        if drift > g["max_marker_drift_px"]:
            raise StationFault(f"camera_moved_marker_drift_{drift:.1f}px")

    def idle_maintenance(self) -> None:
        now = time.monotonic()
        g = self.cfg["geometry"]
        if now - self._last_marker_check > g["marker_check_interval_s"]:
            self._last_marker_check = now
            self.check_markers()
        if self.ref is None or now - self.ref.t > self.cfg["reference"]["refresh_interval_s"]:
            # Refresh only if the chamber still looks empty against the current reference.
            if self.ref is not None:
                _, bgr, depth = self._wait_settled(self.cfg["settle"]["timeout_s"])
                if self._occupancy(bgr, depth).state != Occupancy.NONE:
                    raise StationFault("chamber_not_empty_while_idle")
            self.build_reference()

    # ------------------------------------------------------------ main cycle
    def run_cycle(self) -> CycleRecord | None:
        """Process at most one item. Returns None when no item was delivered."""
        if self.faulted:
            raise RuntimeError(f"Station is faulted ({self.faulted}); call reset() after inspection")
        self.ctrl.heartbeat()
        try:
            if self.ref is None:
                self.build_reference()
            item_id = self.ctrl.wait_delivered(self.cfg["controller"]["poll_timeout_s"])
            if item_id is None:
                self.idle_maintenance()
                return None
        except StationFault as e:
            rec = CycleRecord(item_id="-", trace=[State.WAIT_FOR_OBJECT.value])
            return self._fault(rec, e.reason, [], None)

        rec = CycleRecord(item_id=str(item_id), t_start=time.monotonic(), trace=[State.WAIT_FOR_OBJECT.value])
        frames: list = []
        mask = None
        try:
            rec.trace.append(State.SETTLE.value)
            _, bgr, depth = self._wait_settled(self.cfg["settle"]["timeout_s"])

            rec.trace.append(State.CHECK_OCCUPANCY.value)
            occ = self._occupancy(bgr, depth)
            rec.occupancy, mask = occ.state.value, occ.mask
            frames.append((bgr, depth))
            if occ.state == Occupancy.NONE:
                raise StationFault("delivered_but_chamber_empty")
            if occ.state != Occupancy.ONE:
                return self._hold(rec, f"occupancy_{occ.state.value}_{occ.reason}", frames, mask)

            rec.trace.append(State.CAPTURE_AND_CLASSIFY.value)
            decision, frames, mask = self._capture_and_classify(rec)

            rec.trace.append(State.DECIDE.value)
            self._fill_decision(rec, decision)
            if decision.outcome == Outcome.UNKNOWN:
                return self._hold(rec, decision.reason, frames, mask)
            self._send_and_confirm(rec, decision.outcome.value)
            return self._finish(rec, frames, mask)
        except StationFault as e:
            return self._fault(rec, e.reason, frames, mask)

    def _capture_and_classify(self, rec: CycleRecord):
        cap, dcfg = self.cfg["capture"], self.cfg["decision"]
        agg = TemporalAggregator(self.tax, min_frames=cap["min_frames"])
        rec.metal = self.ctrl.metal_present()
        frames, mask, prev = [], None, None
        quality_failures = 0
        t0 = time.monotonic()

        def decide():
            ood_score = agg.ood_score()
            flag = bool(dcfg.get("ood_enabled", True) and self.ood is not None and self.ood.is_ood(ood_score))
            ms = dcfg.get("metal_sensor") or {}
            model = (ms["p_detect"], ms["p_false_alarm"]) if ms.get("enabled") else None
            return decide_item(agg, self.cost, min_consistency=cap["min_consistency"], ood_flag=flag,
                               metal_reading=rec.metal, metal_model=model)

        while agg.n < cap["max_frames"] and time.monotonic() - t0 < cap["timeout_s"]:
            _, bgr, depth = self._read()
            occ = self._occupancy(bgr, depth)
            if occ.state == Occupancy.NONE:
                raise StationFault("object_disappeared_during_capture")
            if occ.state != Occupancy.ONE:
                return Decision(Outcome.UNKNOWN, f"occupancy_changed_{occ.state.value}"), frames, occ.mask
            q = assess_quality(bgr, occ.mask, prev, self.cfg["quality"], self.mpp)
            prev = bgr
            if not q.ok:
                quality_failures += 1
                if quality_failures > cap["max_quality_failures"]:
                    return Decision(Outcome.UNKNOWN, "image_quality_" + "+".join(q.reasons)), frames, occ.mask
                continue
            geom = geometric_features(occ.mask, occ.height_mm, occ.depth_invalid, self.mpp)
            out = self.clf(bgr, geom)
            score = self.ood.score(out.embedding) if (self.ood is not None and out.embedding is not None) else None
            agg.add(softmax(out.logits, self.temperature), score, geom)
            frames.append((bgr, depth))
            mask = occ.mask
            rec.frames = agg.n
            if agg.ready():
                d = decide()
                self._stash(rec, agg)
                if d.outcome != Outcome.UNKNOWN or d.reason == "out_of_distribution":
                    return d, frames, mask          # early stop: accepted, or OOD (more frames won't help)
        if agg.n == 0:
            return Decision(Outcome.UNKNOWN, "no_usable_frames"), frames, mask
        self._stash(rec, agg)
        return decide(), frames, mask

    def _stash(self, rec: CycleRecord, agg: TemporalAggregator) -> None:
        tp = agg.type_probs()
        top = np.argsort(tp)[::-1][:3]
        rec.top_types = [[self.tax.type_names[i], round(float(tp[i]), 4)] for i in top]
        rec.ood = agg.ood_score()
        rec.geom = {k: round(float(v), 3) for k, v in zip(GEOM_FEATURE_NAMES, np.mean(agg.geoms, axis=0))}

    def _fill_decision(self, rec: CycleRecord, d: Decision) -> None:
        rec.outcome, rec.reason, rec.p = d.outcome.value, d.reason, d.p
        if d.row_probs is not None:
            rec.row_probs = {r: round(float(v), 5) for r, v in zip(ROWS, d.row_probs)}
        if d.expected_cost is not None:
            rec.expected_cost = {r: (None if not np.isfinite(v) else round(float(v), 4))
                                 for r, v in zip(ROWS[:4], d.expected_cost)}

    # ------------------------------------------------------------ exits
    def _send_and_confirm(self, rec: CycleRecord, command: str) -> None:
        c = self.cfg["controller"]
        rec.trace.append(State.SEND_BIN_COMMAND.value)
        if not self.ctrl.send_bin(rec.item_id, command, rec.summary(), c["ack_timeout_s"]):
            raise StationFault("no_controller_ack")
        rec.trace.append(State.WAIT_FOR_DROP_CONFIRMATION.value)
        if not self.ctrl.wait_drop_confirmed(rec.item_id, c["drop_timeout_s"]):
            raise StationFault("drop_not_confirmed")
        _, bgr, depth = self._wait_settled(self.cfg["settle"]["timeout_s"])
        if self._occupancy(bgr, depth).state != Occupancy.NONE:
            raise StationFault("chamber_not_empty_after_drop")

    def _hold(self, rec: CycleRecord, reason: str | None, frames, mask) -> CycleRecord:
        rec.trace.append(State.HOLD.value)
        rec.outcome, rec.reason = Outcome.UNKNOWN.value, reason
        try:
            self._send_and_confirm(rec, "HOLD")
        except StationFault as e:
            return self._fault(rec, e.reason, frames, mask)
        return self._finish(rec, frames, mask)

    def _fault(self, rec: CycleRecord, reason: str, frames, mask) -> CycleRecord:
        rec.trace.append(State.FAULT.value)
        rec.outcome, rec.reason = Outcome.FAULT.value, reason
        self.faulted = reason
        self.ctrl.fault(reason)
        return self._finish(rec, frames, mask, next_state=False)

    def _finish(self, rec: CycleRecord, frames, mask, next_state: bool = True) -> CycleRecord:
        rec.trace.append(State.LOG.value)
        rec.duration_s = round(time.monotonic() - rec.t_start, 3) if rec.t_start else 0.0
        if next_state:
            rec.trace.append(State.WAIT_FOR_OBJECT.value)
        if self.logger:
            self.logger.log(rec, frames, mask)
        return rec

    def reset(self) -> None:
        """Manual reset after an operator has inspected the fault. Rebuilds the reference."""
        self.faulted = None
        self.ref = None

    def run(self, max_cycles: int | None = None, idle_sleep: float = 0.0, stop_when_idle: bool = False):
        n = 0
        while not self.faulted and (max_cycles is None or n < max_cycles):
            rec = self.run_cycle()
            if rec is None:
                if stop_when_idle:
                    return
                if idle_sleep:
                    time.sleep(idle_sleep)
                continue
            n += 1
            yield rec
