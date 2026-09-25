from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np


class Bin(str, Enum):
    RED = "RED"
    YELLOW = "YELLOW"
    BLUE = "BLUE"
    WHITE = "WHITE"


BINS: list[Bin] = [Bin.RED, Bin.YELLOW, Bin.BLUE, Bin.WHITE]
# Rows of the bin-probability vector and the cost matrix: the four bins plus INVALID.
ROWS: list[str] = [b.value for b in BINS] + ["INVALID"]


class Outcome(str, Enum):
    RED = "RED"
    YELLOW = "YELLOW"
    BLUE = "BLUE"
    WHITE = "WHITE"
    UNKNOWN = "UNKNOWN"   # routed to the hold / quarantine position
    FAULT = "FAULT"       # system stops, no routing


class Occupancy(str, Enum):
    NONE = "NONE"
    ONE = "ONE"
    MULTIPLE = "MULTIPLE"
    OUT_OF_POSITION = "OUT_OF_POSITION"


@dataclass
class Frame:
    """One RGB-D capture. Depth is aligned to colour, in mm, 0 = invalid."""
    bgr: np.ndarray
    depth: np.ndarray | None
    t: float
