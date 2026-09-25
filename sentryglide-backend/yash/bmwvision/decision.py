"""Temporal aggregation and the cost-based accept / hold rule.

Rule (Bayes decision with rejection):
    R(k)  = sum_j P(j | x) * C[j, k]        expected cost of routing to bin k
    k*    = argmin_k R(k)
    route to k* if R(k*) < c_hold, otherwise hold (UNKNOWN)
P(j | x) runs over RED, YELLOW, BLUE, WHITE and INVALID. Probabilities must be calibrated
(temperature scaling on item-level outputs) for R to mean anything.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .taxonomy import Taxonomy
from .types import BINS, ROWS, Outcome


# ---------------------------------------------------------------- costs
@dataclass
class CostModel:
    C: np.ndarray        # (5, 4): rows ROWS (true), columns BINS (routed)
    c_hold: float

    @classmethod
    def from_config(cls, dcfg: dict) -> "CostModel":
        cm = dcfg["cost_matrix"]
        C = np.array([cm[r] for r in ROWS], dtype=np.float64)
        if C.shape != (len(ROWS), len(BINS)):
            raise ValueError(f"cost_matrix must be {len(ROWS)} rows x {len(BINS)} columns")
        return cls(C, float(dcfg["c_hold"]))


@dataclass
class Decision:
    outcome: Outcome
    reason: str | None = None
    row_probs: np.ndarray | None = None       # (5,)
    expected_cost: np.ndarray | None = None   # (4,)

    @property
    def p(self) -> float:
        """Probability of the routed bin, or of the most likely row when held."""
        if self.row_probs is None:
            return 0.0
        if self.outcome.value in ROWS:
            return float(self.row_probs[ROWS.index(self.outcome.value)])
        return float(self.row_probs.max())


def bayes_route(row_probs: np.ndarray, cost: CostModel) -> Decision:
    R = np.asarray(row_probs, np.float64) @ cost.C
    k = int(np.argmin(R))
    if R[k] < cost.c_hold:
        return Decision(Outcome(BINS[k].value), None, row_probs, R)
    return Decision(Outcome.UNKNOWN, f"expected_cost_{R[k]:.3g}_above_hold_{cost.c_hold:g}", row_probs, R)


# ---------------------------------------------------------------- aggregation
@dataclass
class TemporalAggregator:
    """Mean of calibrated per-frame type probabilities for one stationary item.

    Frames of a stationary item share viewpoint and errors, so averaging removes transient
    noise only. The consistency check guards against a flickering decision.
    """
    taxonomy: Taxonomy
    min_frames: int = 3
    probs: list = field(default_factory=list)
    ood_scores: list = field(default_factory=list)
    geoms: list = field(default_factory=list)

    def reset(self):
        self.probs, self.ood_scores, self.geoms = [], [], []

    def add(self, type_probs: np.ndarray, ood_score: float | None = None, geom: np.ndarray | None = None):
        self.probs.append(np.asarray(type_probs, np.float64))
        if ood_score is not None:
            self.ood_scores.append(float(ood_score))
        if geom is not None:
            self.geoms.append(geom)

    @property
    def n(self) -> int:
        return len(self.probs)

    def ready(self) -> bool:
        return self.n >= self.min_frames

    def type_probs(self) -> np.ndarray:
        return np.mean(self.probs, axis=0)

    def row_probs(self) -> np.ndarray:
        return self.taxonomy.to_rows(self.type_probs())

    def consistency(self) -> float:
        """Fraction of frames whose own most likely row equals the aggregate's."""
        per = self.taxonomy.to_rows(np.stack(self.probs)).argmax(axis=1)
        return float((per == self.row_probs().argmax()).mean())

    def ood_score(self) -> float | None:
        return float(np.median(self.ood_scores)) if self.ood_scores else None


def decide_item(agg: TemporalAggregator, cost: CostModel, *, min_consistency: float,
                ood_flag: bool = False, metal_reading: bool | None = None,
                metal_model: tuple[float, float] | None = None) -> Decision:
    """Final item decision. metal_model = (p_detect, p_false_alarm) when a sensor is fitted."""
    tp = agg.type_probs()
    if metal_model is not None:
        tp = agg.taxonomy.apply_metal(tp, metal_reading, *metal_model)
    rows = agg.taxonomy.to_rows(tp)
    if agg.n < agg.min_frames:
        return Decision(Outcome.UNKNOWN, "insufficient_frames", rows)
    if ood_flag:
        return Decision(Outcome.UNKNOWN, "out_of_distribution", rows)
    c = agg.consistency()
    if c < min_consistency:
        return Decision(Outcome.UNKNOWN, f"inconsistent_frames_{c:.2f}", rows)
    return bayes_route(rows, cost)
