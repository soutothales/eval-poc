"""Regression gate: evaluate a comparison against thresholds; CI uses the exit code."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import yaml

from eval import constants as c
from eval.compare import compare_runs


def load_thresholds(path: Path = c.THRESHOLDS_PATH) -> Dict[str, Any]:
    """Load gate thresholds.

    Parameters
    ----------
    path: Path
        Thresholds YAML.

    Returns
    -------
    Dict[str, Any]
        Parsed thresholds.
    """
    return yaml.safe_load(Path(path).read_text(encoding=c.TEXT_ENCODING))


def evaluate_gate(comparison: Dict[str, Any], thresholds: Dict[str, Any]) -> Dict[str, Any]:
    """Check a comparison against thresholds.

    A quality breach needs the drop to exceed ``max_drop`` and, when
    ``require_significance`` is set, the bootstrap CI to exclude zero in the
    bad direction. Categories with too few cases are not gated.

    Parameters
    ----------
    comparison: Dict[str, Any]
        Output of :func:`eval.compare.compare_runs`.
    thresholds: Dict[str, Any]
        Parsed thresholds.

    Returns
    -------
    Dict[str, Any]
        ``passed`` flag and the list of breaches.
    """
    breaches: List[Dict[str, Any]] = []
    q_th = thresholds.get(c.TH_QUALITY, {})
    min_n = int(thresholds.get(c.TH_MIN_CATEGORY_CASES, 3))
    need_sig = bool(thresholds.get(c.TH_REQUIRE_SIGNIFICANCE, True))

    for row in comparison[c.CMP_QUALITY]:
        metric, scope = row[c.CMP_METRIC], row[c.CMP_SCOPE]
        if metric not in q_th:
            continue
        if scope != c.SCOPE_OVERALL and row[c.CMP_N] < min_n:
            continue
        lower_better = metric in c.LOWER_IS_BETTER
        drop = row[c.CMP_DELTA] if lower_better else -row[c.CMP_DELTA]
        significant = row[c.CMP_CI_LOW] > 0 if lower_better else row[c.CMP_CI_HIGH] < 0
        max_drop = float(q_th[metric][c.TH_MAX_DROP])
        if drop > max_drop and (significant or not need_sig):
            breaches.append({
                c.GATE_RULE: f"{metric}[{scope}] max_drop={max_drop}",
                c.GATE_DETAIL: (
                    f"{row[c.CMP_MEAN_A]:.3f} -> {row[c.CMP_MEAN_B]:.3f} "
                    f"(delta {row[c.CMP_DELTA]:+.3f}, CI [{row[c.CMP_CI_LOW]:+.3f}, {row[c.CMP_CI_HIGH]:+.3f}], n={row[c.CMP_N]})"
                ),
            })

    o_th = thresholds.get(c.TH_OPERATIONAL, {})
    for row in comparison[c.CMP_OPERATIONAL]:
        metric = row[c.CMP_METRIC]
        rel = row[c.CMP_REL_CHANGE]
        if metric in o_th and rel is not None:
            limit = float(o_th[metric][c.TH_MAX_REL_INCREASE])
            if rel > limit:
                breaches.append({
                    c.GATE_RULE: f"{metric} max_rel_increase={limit:.0%}",
                    c.GATE_DETAIL: f"{row[c.CMP_VALUE_A]} -> {row[c.CMP_VALUE_B]} ({rel:+.1%})",
                })
        if metric == c.OP_ERROR_RATE:
            limit = float(thresholds.get(c.TH_MAX_ERROR_RATE, 0.05))
            if row[c.CMP_VALUE_B] is not None and row[c.CMP_VALUE_B] > limit:
                breaches.append({
                    c.GATE_RULE: f"error_rate <= {limit:.0%}",
                    c.GATE_DETAIL: f"run B error rate {row[c.CMP_VALUE_B]:.1%}",
                })
    return {c.GATE_PASSED: not breaches, c.GATE_BREACHES: breaches}


def gate(run_a: str, run_b: str, thresholds_path: Path = c.THRESHOLDS_PATH) -> Dict[str, Any]:
    """Compare two runs and evaluate the gate.

    Parameters
    ----------
    run_a: str
        Baseline run.
    run_b: str
        Candidate run.
    thresholds_path: Path
        Thresholds YAML.

    Returns
    -------
    Dict[str, Any]
        Gate result plus the comparison warnings.
    """
    th = load_thresholds(thresholds_path)
    comparison = compare_runs(
        run_a, run_b,
        samples=int(th.get(c.TH_BOOTSTRAP_SAMPLES, 2000)),
        confidence=float(th.get(c.TH_CONFIDENCE, 0.95)),
        seed=int(th.get(c.TH_SEED, 7)),
    )
    result = evaluate_gate(comparison, th)
    result[c.CMP_WARNINGS] = comparison[c.CMP_WARNINGS]
    return result
