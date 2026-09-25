import numpy as np

from bmwvision.decision import CostModel, TemporalAggregator, bayes_route, decide_item
from bmwvision.types import ROWS, Outcome


def rows(**kw):
    return np.array([kw.get(r, 0.0) for r in ROWS])


def test_confident_white_is_accepted(cfg):
    cost = CostModel.from_config(cfg["decision"])
    d = bayes_route(rows(WHITE=0.96, RED=0.02, BLUE=0.01, YELLOW=0.01), cost)
    assert d.outcome == Outcome.WHITE


def test_ambiguous_is_held(cfg):
    cost = CostModel.from_config(cfg["decision"])
    d = bayes_route(rows(WHITE=0.41, RED=0.32, YELLOW=0.18, BLUE=0.09), cost)
    assert d.outcome == Outcome.UNKNOWN


def test_small_sharps_probability_blocks_red(cfg):
    """RED at 0.99 is still held if 1% could be a sharp: 0.01 x 1000 > c_hold."""
    cost = CostModel.from_config(cfg["decision"])
    assert bayes_route(rows(RED=0.99, WHITE=0.01), cost).outcome == Outcome.UNKNOWN
    assert bayes_route(rows(RED=0.9995, WHITE=0.0005), cost).outcome == Outcome.RED


def test_argmax_red_but_routed_white_when_cheaper(cfg):
    cost = CostModel.from_config(cfg["decision"])
    cost = CostModel(cost.C, c_hold=10.0)
    d = bayes_route(rows(RED=0.6, WHITE=0.4), cost)
    assert d.outcome == Outcome.WHITE      # the safe bin wins despite lower probability


def _agg(tax, type_probs_list):
    a = TemporalAggregator(tax, min_frames=3)
    for p in type_probs_list:
        a.add(np.asarray(p))
    return a


def onehot(tax, name, p=0.99):
    v = np.full(len(tax.type_names), (1 - p) / (len(tax.type_names) - 1))
    v[tax.type_names.index(name)] = p
    return v


def test_metal_evidence(cfg, tax):
    cost = CostModel.from_config(cfg["decision"])
    model = (0.95, 0.01)
    glove = [onehot(tax, "gloves", 0.99)] * 3
    # without sensor, residual sharps mass holds it; a "no metal" reading lets it through
    assert decide_item(_agg(tax, glove), cost, min_consistency=0.8).outcome == Outcome.UNKNOWN
    assert decide_item(_agg(tax, glove), cost, min_consistency=0.8, metal_reading=False,
                       metal_model=model).outcome == Outcome.RED
    # sensor says metal but vision says glove: conflict -> hold
    assert decide_item(_agg(tax, glove), cost, min_consistency=0.8, metal_reading=True,
                       metal_model=model).outcome == Outcome.UNKNOWN


def test_inconsistent_frames_held(cfg, tax):
    cost = CostModel(CostModel.from_config(cfg["decision"]).C, c_hold=1e9)
    a = _agg(tax, [onehot(tax, "gloves"), onehot(tax, "needle"), onehot(tax, "gloves")])
    d = decide_item(a, cost, min_consistency=0.8)
    assert d.outcome == Outcome.UNKNOWN and d.reason.startswith("inconsistent")


def test_ood_and_min_frames(cfg, tax):
    cost = CostModel.from_config(cfg["decision"])
    a = _agg(tax, [onehot(tax, "needle")] * 2)
    assert decide_item(a, cost, min_consistency=0.8).reason == "insufficient_frames"
    a.add(onehot(tax, "needle"))
    assert decide_item(a, cost, min_consistency=0.8, ood_flag=True).reason == "out_of_distribution"
    assert decide_item(a, cost, min_consistency=0.8).outcome == Outcome.WHITE


def test_taxonomy_rows_sum_to_one(tax):
    p = np.random.default_rng(0).dirichlet(np.ones(len(tax.type_names)))
    assert np.isclose(tax.to_rows(p).sum(), 1.0)
