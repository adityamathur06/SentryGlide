import numpy as np

from bmwvision.classifier import softmax
from bmwvision.decision import CostModel
from bmwvision.offline import (aggregate_placements, clopper_pearson_upper, fit_temperature,
                               risk_coverage, select_c_hold)


def test_rule_of_three():
    assert abs(clopper_pearson_upper(0, 3000) - 3 / 3000) < 1e-4
    assert clopper_pearson_upper(0, 0) == 1.0


def synth(n_place=600, K=5, T_true=2.5, seed=0):
    """Logits that are overconfident by factor T_true: true probs are softmax(z / T_true)."""
    rng = np.random.default_rng(seed)
    logits, pid, lab = [], [], []
    for p in range(n_place):
        y = rng.integers(K)
        base = rng.normal(0, 1, K)
        base[y] += 2.0
        z = base * T_true
        probs = softmax(z, T_true)
        y = rng.choice(K, p=probs)      # label drawn from the calibrated distribution
        for _ in range(3):
            logits.append(z + rng.normal(0, 0.05, K))
            pid.append(p)
            lab.append(y)
    return np.array(logits), np.array(pid), np.array(lab)


def test_temperature_recovers_overconfidence():
    z, pid, y = synth()
    T = fit_temperature(z, pid, y)
    assert 1.8 < T < 3.4, T
    _, p, _ = aggregate_placements(z, pid, T)
    assert np.allclose(p.sum(1), 1)


def test_risk_coverage_monotone(cfg):
    cost = CostModel.from_config(cfg["decision"])
    rng = np.random.default_rng(1)
    probs = rng.dirichlet(np.ones(5) * 0.3, size=400)
    true = np.array([rng.choice(5, p=p) for p in probs])
    tab = risk_coverage(probs, true, cost, [0.1, 1, 10, 100], cfg["decision"]["dangerous"])
    cov = [r["coverage"] for r in tab]
    assert cov == sorted(cov)
    assert select_c_hold(tab, 1.0)["c_hold"] == 100
    assert select_c_hold(tab, 0.0) is None
