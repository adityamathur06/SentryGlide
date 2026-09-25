"""Offline calibration of the decision layer from validation outputs.

Input: per-frame classifier outputs on rig data, with a placement id (one drop of one item)
and a physical item id (the object itself). Aggregation happens per placement, exactly as
at runtime; every split is by physical item, so the same object never appears on both sides.

Steps:
  1. fit_temperature on split A (item-level NLL of mean softmax)
  2. fit the OOD threshold on split A embeddings
  3. risk_coverage on split B over a sweep of c_hold; choose the largest coverage whose
     dangerous-error upper bound meets the target
  4. verify once on an untouched test set with everything frozen
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import beta

from .classifier import softmax
from .decision import CostModel
from .taxonomy import Taxonomy
from .types import BINS, ROWS


def clopper_pearson_upper(k: int, n: int, alpha: float = 0.05) -> float:
    """One-sided (1 - alpha) upper bound on a rate with k events in n trials.
    k = 0 gives about 3/n at alpha = 0.05 (the rule of three)."""
    if n == 0:
        return 1.0
    return 1.0 if k >= n else float(beta.ppf(1 - alpha, k + 1, n - k))


def aggregate_placements(frame_logits: np.ndarray, placement_ids: np.ndarray, temperature: float):
    """Mean calibrated softmax per placement. Returns (unique_ids, probs (P, K), inverse index)."""
    uniq, inv = np.unique(placement_ids, return_inverse=True)
    P = softmax(frame_logits, temperature)
    out = np.zeros((len(uniq), P.shape[1]))
    np.add.at(out, inv, P)
    out /= np.bincount(inv)[:, None]
    return uniq, out, inv


def placement_labels(labels: np.ndarray, placement_ids: np.ndarray) -> np.ndarray:
    uniq, inv = np.unique(placement_ids, return_inverse=True)
    out = np.full(len(uniq), -1)
    for i, lab in zip(inv, labels):
        if out[i] not in (-1, lab):
            raise ValueError(f"placement {uniq[i]} has mixed labels")
        out[i] = lab
    return out


def fit_temperature(frame_logits, placement_ids, labels) -> float:
    y = placement_labels(labels, placement_ids)

    def nll(logT):
        _, p, _ = aggregate_placements(frame_logits, placement_ids, float(np.exp(logT)))
        return -np.mean(np.log(p[np.arange(len(y)), y] + 1e-12))

    r = minimize_scalar(nll, bounds=(np.log(0.05), np.log(20.0)), method="bounded")
    return float(np.exp(r.x))


def ece(probs: np.ndarray, y: np.ndarray, n_bins: int = 10) -> float:
    conf, pred = probs.max(1), probs.argmax(1)
    edges = np.linspace(0, 1, n_bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
    return float(e)


def risk_coverage(row_probs: np.ndarray, true_rows: np.ndarray, cost: CostModel,
                  c_hold_values, dangerous: list[tuple[str, str]], alpha: float = 0.05) -> list[dict]:
    """Coverage vs risk for each candidate c_hold. Dangerous rate is over all placements;
    sharps_missed is over placements whose true row is WHITE."""
    R = row_probs @ cost.C
    k = R.argmin(1)
    rmin = R[np.arange(len(R)), k]
    bin_names = [b.value for b in BINS]
    dpairs = {(ROWS.index(t), bin_names.index(r)) for t, r in dangerous}
    is_danger = np.array([(t, kk) in dpairs for t, kk in zip(true_rows, k)])
    white = true_rows == ROWS.index("WHITE")
    n = len(true_rows)
    table = []
    for c in c_hold_values:
        acc = rmin < c
        err = acc & (true_rows != k)
        dang = acc & is_danger
        miss = acc & white & (k != bin_names.index("WHITE"))
        na = int(acc.sum())
        table.append({
            "c_hold": float(c),
            "coverage": na / n if n else 0.0,
            "n_accepted": na,
            "error_rate_accepted": float(err.sum() / na) if na else 0.0,
            "error_upper_accepted": clopper_pearson_upper(int(err.sum()), na, alpha),
            "n_dangerous": int(dang.sum()),
            "dangerous_upper": clopper_pearson_upper(int(dang.sum()), n, alpha),
            "n_sharps": int(white.sum()),
            "n_sharps_missed": int(miss.sum()),
            "sharps_missed_upper": clopper_pearson_upper(int(miss.sum()), int(white.sum()), alpha),
        })
    return table


def select_c_hold(table: list[dict], max_dangerous_upper: float) -> dict | None:
    ok = [r for r in table if r["dangerous_upper"] <= max_dangerous_upper]
    return max(ok, key=lambda r: (r["coverage"], -r["c_hold"])) if ok else None


def split_by_physical(physical_ids: np.ndarray, frac: float = 0.5, seed: int = 0):
    uniq = np.unique(physical_ids)
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    a = set(uniq[: int(round(len(uniq) * frac))].tolist())
    return np.array([p in a for p in physical_ids])


def rows_for(taxonomy: Taxonomy, type_labels: np.ndarray) -> np.ndarray:
    return np.array([taxonomy.row_index(taxonomy.type_names[t]) for t in type_labels])
