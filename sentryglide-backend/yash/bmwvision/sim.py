"""Simulated rig: camera, controller and a stand-in classifier.

Purpose: exercise calibration, gating, the state machine and the decision rule end to end
before hardware exists. The simulated classifier reads the true item type from the scene,
so simulation results say nothing about real classification accuracy.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from .calibration import make_marker_board
from .classifier import ClassifierOutput, ModelMeta
from .types import Frame


# ---------------------------------------------------------------- items
@dataclass
class Shape:
    kind: str                        # "rect" or "ellipse"
    offset: tuple[float, float]      # mm from item centre, in the item frame
    size: tuple[float, float]        # mm (length, width)
    color: tuple[int, int, int]      # BGR
    height: float                    # mm above plate
    invalid_depth: bool = False      # glass / polished metal: depth sensor returns holes


@dataclass
class SimItem:
    type_name: str
    shapes: list[Shape]
    metal: bool = False
    center: tuple[float, float] = (130.0, 100.0)   # plate mm
    angle: float = 0.0                              # degrees


def _needle(off=(0, 0)):
    return [Shape("rect", (off[0] + 20, off[1]), (38, 0.5), (185, 185, 190), 0.8),
            Shape("rect", (off[0], off[1]), (7, 5), (40, 140, 240), 5)]


PROTOTYPES: dict[str, callable] = {
    "needle": lambda: SimItem("needle", _needle(), metal=True),
    "scalpel_blade": lambda: SimItem("scalpel_blade", [
        Shape("rect", (0, 0), (48, 11), (175, 175, 180), 2, invalid_depth=True),
        Shape("rect", (-27, 0), (14, 8), (60, 120, 170), 5)], metal=True),
    "lancet": lambda: SimItem("lancet", [
        Shape("rect", (0, 0), (28, 7), (210, 210, 215), 4),
        Shape("rect", (15, 0), (6, 1), (170, 170, 175), 1)], metal=True),
    "syringe_without_needle": lambda: SimItem("syringe_without_needle", [
        Shape("rect", (0, 0), (70, 13), (225, 228, 215), 13),
        Shape("rect", (-42, 0), (16, 10), (245, 245, 245), 10)]),
    "syringe_with_fixed_needle": lambda: SimItem("syringe_with_fixed_needle", [
        Shape("rect", (0, 0), (60, 12), (225, 228, 215), 12),
        Shape("rect", (-36, 0), (14, 9), (245, 245, 245), 9)] + _needle((32, 0)), metal=True),
    "gloves": lambda: SimItem("gloves", [
        Shape("ellipse", (0, 0), (85, 48), (200, 140, 60), 8),
        Shape("rect", (40, 0), (30, 30), (200, 140, 60), 6)]),
    "iv_set_tubing": lambda: SimItem("iv_set_tubing", [
        Shape("rect", (0, 0), (100, 4), (225, 232, 232), 4),
        Shape("rect", (-10, 0), (18, 12), (230, 230, 230), 10)]),
    "catheter": lambda: SimItem("catheter", [
        Shape("rect", (0, 0), (105, 3), (210, 225, 225), 3),
        Shape("ellipse", (-45, 0), (15, 12), (70, 130, 220), 8)]),
    "urine_bag": lambda: SimItem("urine_bag", [
        Shape("rect", (0, 0), (75, 55), (215, 225, 205), 5),
        Shape("rect", (44, 0), (24, 4), (220, 230, 215), 4)]),
    "vacutainer": lambda: SimItem("vacutainer", [
        Shape("rect", (0, 0), (75, 14), (205, 210, 200), 14),
        Shape("rect", (-39, 0), (8, 16), (80, 60, 150), 16)]),
    "plastic_bottle": lambda: SimItem("plastic_bottle", [
        Shape("ellipse", (0, 0), (82, 42), (190, 220, 210), 32),
        Shape("rect", (-45, 0), (12, 20), (80, 150, 210), 20)]),
    "soiled_gauze_dressing": lambda: SimItem("soiled_gauze_dressing", [
        Shape("rect", (0, 0), (60, 55), (236, 236, 236), 4),
        Shape("ellipse", (8, -5), (25, 18), (50, 50, 160), 4)]),
    "soiled_cotton": lambda: SimItem("soiled_cotton", [
        Shape("ellipse", (0, 0), (42, 35), (235, 235, 235), 13),
        Shape("ellipse", (5, -3), (18, 15), (45, 45, 145), 13)]),
    "plaster_cast": lambda: SimItem("plaster_cast", [
        Shape("rect", (0, 0), (85, 38), (225, 225, 215), 18),
        Shape("rect", (15, 0), (42, 30), (200, 200, 190), 20)]),
    "glass_vial": lambda: SimItem("glass_vial", [
        Shape("rect", (0, 0), (45, 20), (205, 215, 200), 20, invalid_depth=True),
        Shape("rect", (-25, 0), (8, 16), (60, 60, 200), 18)]),
    "glass_ampoule": lambda: SimItem("glass_ampoule", [
        Shape("ellipse", (0, 0), (55, 17), (210, 220, 210), 17, invalid_depth=True),
        Shape("rect", (-31, 0), (12, 5), (205, 215, 205), 5, invalid_depth=True)]),
    "broken_glass": lambda: SimItem("broken_glass", [
        Shape("rect", (-12, -5), (35, 18), (195, 210, 205), 5, invalid_depth=True),
        Shape("rect", (17, 7), (28, 13), (205, 215, 210), 4, invalid_depth=True)]),
    "metal_implant": lambda: SimItem("metal_implant", [
        Shape("rect", (0, 0), (60, 9), (175, 175, 180), 4, invalid_depth=True)], metal=True),
    "invalid": lambda: SimItem("invalid", [
        Shape("rect", (0, 0), (72, 45), (55, 110, 70), 12),
        Shape("ellipse", (18, -8), (18, 18), (35, 80, 45), 14)]),
    # Not in the taxonomy: tests the OOD path.
    "phone": lambda: SimItem("phone", [Shape("rect", (0, 0), (110, 55), (35, 35, 35), 9)]),
}


def make_item(type_name: str, center=(130.0, 100.0), angle=0.0) -> SimItem:
    it = PROTOTYPES[type_name]()
    it.center, it.angle = center, angle
    return it


# ---------------------------------------------------------------- scene + camera
@dataclass
class SimScene:
    geom_cfg: dict
    items: list[SimItem] = field(default_factory=list)
    blur_sigma: float = 0.0
    jitter_frames: int = 0            # remaining frames of post-delivery motion
    jitter_mm: float = 3.0


class SimulatedCamera:
    """Renders the plate in mm space, then projects it through a fixed homography with mild
    perspective, as a real overhead camera would see it."""

    def __init__(self, scene: SimScene, image_size=(1600, 1200), tex_px_per_mm: float = 6.0,
                 cam_dist_mm: float = 400.0, seed: int = 0):
        self.scene, self.W, self.Hh = scene, *image_size
        self.s = tex_px_per_mm
        self.cam_dist = cam_dist_mm
        self.rng = np.random.default_rng(seed)
        cv2.setRNGSeed(seed)
        pw, ph = scene.geom_cfg["plate_mm"]
        plate = np.float32([[0, 0], [pw, 0], [pw, ph], [0, ph]])
        img = np.float32([[70, 55], [1535, 70], [1520, 1150], [80, 1135]])
        H_mm_to_img = cv2.getPerspectiveTransform(plate, img)
        self.H_tex_to_img = H_mm_to_img @ np.diag([1 / self.s, 1 / self.s, 1.0])
        self.board = make_marker_board(scene.geom_cfg, self.s)

    def _poly(self, it: SimItem, sh: Shape, jitter):
        a = np.deg2rad(it.angle)
        R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
        c = np.array(it.center) + R @ np.array(sh.offset) + jitter
        return ((c[0] * self.s, c[1] * self.s), (sh.size[0] * self.s, sh.size[1] * self.s), it.angle)

    def read(self) -> Frame:
        sc = self.scene
        tex = self.board.copy()
        hmap = np.zeros(tex.shape[:2], np.float32)
        inval = np.zeros(tex.shape[:2], np.uint8)
        jitter = np.zeros(2)
        if sc.jitter_frames > 0:
            jitter = self.rng.normal(0, sc.jitter_mm, 2)
            sc.jitter_frames -= 1
        for it in sc.items:
            for sh in it.shapes:
                box = self._poly(it, sh, jitter)
                if sh.kind == "ellipse":
                    cv2.ellipse(tex, box, sh.color, -1, cv2.LINE_AA)
                    cv2.ellipse(hmap, box, float(sh.height), -1)
                    if sh.invalid_depth:
                        cv2.ellipse(inval, box, 1, -1)
                else:
                    pts = cv2.boxPoints(box).astype(np.int32)
                    cv2.fillPoly(tex, [pts], sh.color, cv2.LINE_AA)
                    cv2.fillPoly(hmap, [pts], float(sh.height))
                    if sh.invalid_depth:
                        cv2.fillPoly(inval, [pts], 1)
        size = (self.W, self.Hh)
        bgr = cv2.warpPerspective(tex, self.H_tex_to_img, size, flags=cv2.INTER_LINEAR,
                                  borderValue=(40, 40, 40))
        h = cv2.warpPerspective(hmap, self.H_tex_to_img, size, flags=cv2.INTER_NEAREST, borderValue=0)
        inv = cv2.warpPerspective(inval, self.H_tex_to_img, size, flags=cv2.INTER_NEAREST, borderValue=0)
        if sc.blur_sigma > 0:
            bgr = cv2.GaussianBlur(bgr, (0, 0), sc.blur_sigma)
        # cv2 RNG is much faster than numpy for full-frame noise
        n3 = np.empty(bgr.shape, np.float32)
        cv2.randn(n3, 0, 1.5)
        bgr = cv2.convertScaleAbs(bgr.astype(np.float32) + n3)
        n1 = np.empty(h.shape, np.float32)
        cv2.randn(n1, 0, 0.6)
        depth = (self.cam_dist - h) + n1
        depth[inv > 0] = 0
        u = np.empty(h.shape, np.uint16)
        cv2.randu(u, 0, 10000)
        depth[u < 10] = 0                                       # 0.1% random dropouts
        return Frame(bgr=bgr, depth=depth, t=time.monotonic())


# ---------------------------------------------------------------- controller
@dataclass
class Delivery:
    item_id: str
    items: list[SimItem]
    stuck: bool = False               # item fails to leave the chamber after the drop


class SimulatedController:
    def __init__(self, scene: SimScene, deliveries: list[Delivery], metal_sensor: bool = True,
                 ack: bool = True):
        self.scene, self.queue = scene, list(deliveries)
        self.metal_sensor, self.ack = metal_sensor, ack
        self.current: Delivery | None = None
        self.commands: list[tuple[str, str, dict]] = []
        self.faults: list[str] = []
        self.heartbeats = 0

    def wait_delivered(self, timeout: float) -> str | None:
        if not self.queue:
            return None
        self.current = self.queue.pop(0)
        self.scene.items = list(self.current.items)
        self.scene.jitter_frames = 3
        return self.current.item_id

    def send_bin(self, item_id, command, payload, timeout) -> bool:
        self.commands.append((item_id, command, payload))
        return self.ack

    def wait_drop_confirmed(self, item_id, timeout) -> bool:
        if self.current and not self.current.stuck:
            self.scene.items = []
        return True

    def metal_present(self) -> bool | None:
        if not self.metal_sensor:
            return None
        return any(it.metal for it in self.scene.items)

    def fault(self, reason: str) -> None:
        self.faults.append(reason)

    def heartbeat(self) -> None:
        self.heartbeats += 1


# ---------------------------------------------------------------- classifier stand-in
class SimulatedClassifier:
    """Returns noisy logits peaked on the true type of the item in the scene.
    `confusion` maps a true type to (confused type, logit given to it) to model hard pairs.
    Types not in meta.type_names (e.g. "phone") get a confident wrong answer and an
    embedding far from the training bank, which the OOD check must catch."""

    def __init__(self, scene: SimScene, meta: ModelMeta, confidence: float = 7.0, noise: float = 0.7,
                 confusion: dict[str, tuple[str, float]] | None = None, emb_dim: int = 32, seed: int = 0):
        self.scene, self.meta = scene, meta
        self.conf, self.noise, self.confusion = confidence, noise, confusion or {}
        self.rng = np.random.default_rng(seed)
        g = np.random.default_rng(1234)
        self.centroids = {t: g.normal(size=emb_dim) for t in meta.type_names}
        self.unknown_centroid = g.normal(size=emb_dim)

    def _embed(self, t: str) -> np.ndarray:
        c = self.centroids.get(t, self.unknown_centroid)
        return c + self.rng.normal(0, 0.25, c.shape)

    def make_bank(self, n_per_type: int = 50) -> np.ndarray:
        return np.stack([self._embed(t) for t in self.meta.type_names for _ in range(n_per_type)])

    def __call__(self, bgr_roi, geom) -> ClassifierOutput:
        K = len(self.meta.type_names)
        logits = self.rng.normal(0, self.noise, K)
        t = self.scene.items[0].type_name if self.scene.items else "invalid"
        if t in self.meta.type_names:
            logits[self.meta.type_names.index(t)] += self.conf
        else:
            logits[self.meta.type_names.index("gloves")] += self.conf   # confident and wrong
        if t in self.confusion:
            other, v = self.confusion[t]
            logits[self.meta.type_names.index(other)] += v
        return ClassifierOutput(logits, self._embed(t))
