"""Fine-grained object type -> bin mapping.

The classifier predicts object types (needle, glove, glass vial, ...). Bin probabilities are
sums over the types mapped to each bin. When the rules change, edit taxonomy.yaml; no retraining.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from .types import BINS, ROWS, Bin


class Taxonomy:
    def __init__(self, type_names: list[str], type_to_bin: dict[str, str | None],
                 metal_types: list[str] | None = None, metal_unknown: list[str] | None = None):
        missing = [t for t in type_names if t not in type_to_bin]
        if missing:
            raise ValueError(f"Model types missing from taxonomy: {missing}")
        self.type_names = list(type_names)
        self.type_to_bin = {t: (Bin(type_to_bin[t]) if type_to_bin[t] else None) for t in type_names}
        # M[k, r] = 1 if type k maps to row r (RED, YELLOW, BLUE, WHITE, INVALID)
        self.M = np.zeros((len(type_names), len(ROWS)), dtype=np.float64)
        for k, t in enumerate(type_names):
            b = self.type_to_bin[t]
            self.M[k, ROWS.index(b.value if b else "INVALID")] = 1.0
        # 1 = contains metal, 0 = no metal, nan = unknown (sensor gives no evidence)
        metal_types, metal_unknown = set(metal_types or []), set(metal_unknown or [])
        self.metal = np.array([np.nan if t in metal_unknown else float(t in metal_types)
                               for t in type_names])

    @classmethod
    def from_yaml(cls, path: str | Path, type_names: list[str]) -> "Taxonomy":
        d = yaml.safe_load(Path(path).read_text())
        return cls(type_names, d["types"], d.get("metal_types"), d.get("metal_unknown"))

    def to_rows(self, type_probs: np.ndarray) -> np.ndarray:
        """(K,) or (N, K) type probabilities -> (5,) or (N, 5) over RED, YELLOW, BLUE, WHITE, INVALID."""
        return np.asarray(type_probs) @ self.M

    def apply_metal(self, type_probs: np.ndarray, reading: bool | None,
                    p_detect: float, p_false_alarm: float) -> np.ndarray:
        """Bayes update of type probabilities with the metal sensor reading.

        p_detect      = P(sensor fires | metal item), measured on the rig for the hardest
                        metal items (thin needles far from the coil). Overstating it is unsafe:
                        a "no metal" reading then suppresses sharps more than it should.
        p_false_alarm = P(sensor fires | non-metal item).
        """
        if reading is None:
            return type_probs
        if reading:
            L = np.where(self.metal == 1, p_detect, p_false_alarm)
        else:
            L = np.where(self.metal == 1, 1 - p_detect, 1 - p_false_alarm)
        L = np.where(np.isnan(self.metal), 1.0, L)
        post = np.asarray(type_probs) * L
        return post / post.sum()

    def row_index(self, type_name: str) -> int:
        b = self.type_to_bin[type_name]
        return ROWS.index(b.value if b else "INVALID")


__all__ = ["Taxonomy", "BINS", "ROWS"]
