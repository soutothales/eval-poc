"""Deterministic metrics, pricing, bootstrap and gate logic."""

import numpy as np

from eval import constants as c
from eval import deterministic as det
from eval.compare import bootstrap_ci
from eval.gate import evaluate_gate
from eval.pricing import PriceTable


def test_retrieval_metrics():
    assert det.hit_rate(["a", "b"], ["x", "a"]) == 0.5
    assert det.mrr(["a", "b"], ["x", "b", "a"]) == 0.5
    assert det.mrr(["a"], ["x"]) == 0.0
    assert det.hit_rate([], ["a"]) is None


def test_behavior_metrics():
    assert det.behavior_accuracy(c.BEHAVIOR_NOT_FOUND, "NOT_FOUND: no clause") == 1.0
    assert det.behavior_accuracy(c.BEHAVIOR_NOT_FOUND, "It is 5%") == 0.0
    assert det.behavior_accuracy(c.BEHAVIOR_ANSWER, "NOT_FOUND: x") == 0.0
    assert det.behavior_accuracy(c.BEHAVIOR_DECLINE, "no") is None
    assert det.false_refusal(c.BEHAVIOR_ANSWER, "NOT_FOUND: x") == 1.0
    assert det.false_refusal(c.BEHAVIOR_NOT_FOUND, "NOT_FOUND: x") is None


def test_pricing_resolution_and_tiers():
    pt = PriceTable.load()
    assert pt.resolve("claude-haiku-5-5") == "claude-haiku-5-5"
    assert pt.resolve("claude-haiku-5-5-20261001") == "claude-haiku-5-5"
    assert pt.resolve("gpt-x") is None
    short = pt.call_cost("claude-haiku-5-5", 1_000_000, 0)
    long = pt.call_cost("claude-haiku-5-5", 1_000_000 + 0, 0)
    assert short == long  # both above threshold -> long tier
    assert pt.call_cost("claude-haiku-5-5", 50_000, 0) == 50_000 * 0.10 / 1_000_000
    cost, unpriced = pt.usage_cost({"claude-opus-5-5": {"input_tokens": 1000, "output_tokens": 100}, "mystery": {}})
    assert abs(cost - (1000 * 4 + 100 * 20) / 1e6) < 1e-12 and unpriced == ["mystery"]


def test_bootstrap_ci_contains_mean_and_is_reproducible():
    d = np.array([0.1, -0.05, 0.2, 0.0, 0.15, 0.05])
    lo, hi = bootstrap_ci(d, samples=1000, seed=1)
    assert lo <= d.mean() <= hi
    assert (lo, hi) == bootstrap_ci(d, samples=1000, seed=1)


def _q(metric, delta, lo, hi, n=20, scope=c.SCOPE_OVERALL):
    return {c.CMP_METRIC: metric, c.CMP_SCOPE: scope, c.CMP_MEAN_A: 0.8, c.CMP_MEAN_B: 0.8 + delta,
            c.CMP_DELTA: delta, c.CMP_CI_LOW: lo, c.CMP_CI_HIGH: hi, c.CMP_N: n}


TH = {
    c.TH_REQUIRE_SIGNIFICANCE: True,
    c.TH_MIN_CATEGORY_CASES: 3,
    c.TH_QUALITY: {c.M_CORRECTNESS: {c.TH_MAX_DROP: 0.05}, c.M_FALSE_REFUSAL: {c.TH_MAX_DROP: 0.05}},
    c.TH_OPERATIONAL: {c.OP_AGENT_COST_MEAN: {c.TH_MAX_REL_INCREASE: 0.2}},
    c.TH_MAX_ERROR_RATE: 0.05,
}


def _cmp(quality, ops=()):
    return {c.CMP_QUALITY: quality, c.CMP_OPERATIONAL: list(ops)}


def test_gate_requires_significance():
    noisy = _cmp([_q(c.M_CORRECTNESS, -0.10, -0.25, 0.04)])
    real = _cmp([_q(c.M_CORRECTNESS, -0.10, -0.18, -0.02)])
    assert evaluate_gate(noisy, TH)[c.GATE_PASSED]
    assert not evaluate_gate(real, TH)[c.GATE_PASSED]


def test_gate_lower_is_better_and_small_categories():
    worse = _cmp([_q(c.M_FALSE_REFUSAL, +0.10, 0.02, 0.2)])
    assert not evaluate_gate(worse, TH)[c.GATE_PASSED]
    tiny = _cmp([_q(c.M_CORRECTNESS, -0.5, -0.9, -0.1, n=2, scope=c.CATEGORY_AGGREGATE)])
    assert evaluate_gate(tiny, TH)[c.GATE_PASSED]


def test_gate_operational_and_errors():
    ops = [
        {c.CMP_METRIC: c.OP_AGENT_COST_MEAN, c.CMP_VALUE_A: 1.0, c.CMP_VALUE_B: 1.3, c.CMP_REL_CHANGE: 0.3},
        {c.CMP_METRIC: c.OP_ERROR_RATE, c.CMP_VALUE_A: 0.0, c.CMP_VALUE_B: 0.1, c.CMP_REL_CHANGE: None},
    ]
    res = evaluate_gate(_cmp([], ops), TH)
    assert not res[c.GATE_PASSED] and len(res[c.GATE_BREACHES]) == 2
