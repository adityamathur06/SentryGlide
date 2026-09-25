"""Hardware interfaces. Implement these two protocols for your camera and your controller.

The station only talks to hardware through these methods, so the same state machine runs
against the simulator (bmwvision.sim) and the real rig.
"""
from __future__ import annotations

import time
from typing import Protocol

import numpy as np

from .types import Frame


class Camera(Protocol):
    def read(self) -> Frame:
        """Block until the next frame. Depth must be aligned to colour, in mm, 0 = invalid."""
        ...


class Controller(Protocol):
    """Mechanical controller (PLC / microcontroller) handshake.

    Fail-safe requirement on the controller side: if heartbeat() stops arriving for longer
    than its watchdog period, the controller must hold the item and stop routing. It must
    never act on a stale classification.
    """

    def wait_delivered(self, timeout: float) -> str | None:
        """Return an item id once an item has been placed in the chamber, else None on timeout."""
        ...

    def send_bin(self, item_id: str, command: str, payload: dict, timeout: float) -> bool:
        """command in RED, YELLOW, BLUE, WHITE, HOLD. Return True when the controller acknowledges."""
        ...

    def wait_drop_confirmed(self, item_id: str, timeout: float) -> bool:
        """True once the chute has released the item to the commanded position."""
        ...

    def metal_present(self) -> bool | None:
        """Inductive sensor reading for the current item, or None if no sensor is fitted."""
        ...

    def fault(self, reason: str) -> None:
        """Stop the chute and raise the alarm. Cleared only by a manual reset."""
        ...

    def heartbeat(self) -> None:
        ...


class RealSenseCamera:
    """Adapter for Intel RealSense D4xx cameras. Not tested on hardware in this repo;
    check stream settings against your model. Needs `pip install pyrealsense2`."""

    def __init__(self, width: int = 1280, height: int = 720, fps: int = 30):
        import pyrealsense2 as rs
        self.rs = rs
        self.pipe = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        cfg.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        profile = self.pipe.start(cfg)
        self.depth_scale_mm = profile.get_device().first_depth_sensor().get_depth_scale() * 1000.0
        self.align = rs.align(rs.stream.color)
        # Lock exposure and white balance so the empty reference stays valid.
        color = profile.get_device().first_color_sensor()
        color.set_option(rs.option.enable_auto_exposure, 0)
        color.set_option(rs.option.enable_auto_white_balance, 0)

    def read(self) -> Frame:
        frames = self.align.process(self.pipe.wait_for_frames())
        c, d = frames.get_color_frame(), frames.get_depth_frame()
        bgr = np.asanyarray(c.get_data()).copy()
        depth = np.asanyarray(d.get_data()).astype(np.float32) * self.depth_scale_mm
        return Frame(bgr=bgr, depth=depth, t=time.monotonic())

    def close(self):
        self.pipe.stop()


class ConsoleController:
    """Bench-test controller: the operator places and removes items by hand and reads the
    commanded bin from the console. Use before the PLC link exists."""

    def wait_delivered(self, timeout: float) -> str | None:
        s = input("Place ONE item in the chamber and press Enter (q to quit): ").strip()
        if s.lower() == "q":
            raise KeyboardInterrupt
        self._n = getattr(self, "_n", 0) + 1
        return f"bench_{self._n:05d}"

    def send_bin(self, item_id, command, payload, timeout) -> bool:
        print(f"[{item_id}] -> {command}  {payload}")
        return True

    def wait_drop_confirmed(self, item_id, timeout) -> bool:
        input("Remove the item, then press Enter: ")
        return True

    def metal_present(self) -> bool | None:
        return None

    def fault(self, reason: str) -> None:
        print(f"FAULT: {reason}")

    def heartbeat(self) -> None:
        pass
