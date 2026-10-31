"""Paired comparison of two runs on shared cases, with bootstrap confidence intervals."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from eval import constants as c
from eval.io import read_json, run_dir, write_json
from eval.versioning.manifest import load_cases

DEFAULT_SAMPLES = 2000
DEFAULT_CONFIDENCE = 0.95
DEFAULT_SEED = 7


def bootstrap_ci(
    diffs: np.ndarray,
    samples: int = DEFAULT_SAMPLES,
    confidence: float = DEFAULT_CONFIDENCE,
    seed: int = DEFAULT_SEED,
) -> Tuple[float, float]:
    """Percentile bootstrap CI for the mean of paired differences.

    Parameters
    ----------
    diffs: np.ndarray
        Per-case differences (B - A).
    samples: int
        Bootstrap resamples.
    confidence: float
        Confidence level, e.g. 0.95.
    seed: int
        RNG seed, for reproducible reports.

    Returns
    -------
    Tuple[float, float]
        Lower and upper bounds; both equal the mean when n < 2.
    """
    if len(diffs) < 2:
        m = float(diffs.mean()) if len(diffs) else 0.0
        return m, m
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diffs), size=(samples, len(diffs)))
    means = diffs[idx].mean(axis=1)
    alpha = (1 - confidence) / 2
    return float(np.quantile(means, alpha)), float(np.quantile(means, 1 - alpha))


def paired_stat(
    a: pd.Series,
    b: pd.Series,
    samples: int,
    confidence: float,
    seed: int,
) -> Optional[Dict[str, float]]:
    """Mean A, mean B, delta and CI over pairs where both values exist.

    Parameters
    ----------
    a: pd.Series
        Scores of run A, indexed by case_id.
    b: pd.Series
        Scores of run B, same index.
    samples: int
        Bootstrap resamples.
    confidence: float
        Confidence level.
    seed: int
        RNG seed.

    Returns
    -------
    Optional[Dict[str, float]]
        Stats dict, or ``None`` when there are no pairs.
    """
    mask = a.notna() & b.notna()
    if not mask.any():
        return None
    av, bv = a[mask].astype(float).to_numpy(), b[mask].astype(float).to_numpy()
    lo, hi = bootstrap_ci(bv - av, samples, confidence, seed)
    return {
        c.CMP_MEAN_A: float(av.mean()),
        c.CMP_MEAN_B: float(bv.mean()),
        c.CMP_DELTA: float((bv - av).mean()),
        c.CMP_CI_LOW: lo,
        c.CMP_CI_HIGH: hi,
        c.CMP_N: int(mask.sum()),
    }


def _rel(a: Optional[float], b: Optional[float]) -> Optional[float]:
    """Relative change ``(b - a) / a``.

    Parameters
    ----------
    a: Optional[float]
        Baseline value.
    b: Optional[float]
        New value.

    Returns
    -------
    Optional[float]
        Relative change, or ``None`` if undefined.
    """
    if a is None or b is None or a == 0:
        return None
    return (b - a) / a


def _load(run_id: str) -> Tuple[Dict[str, Any], Dict[str, Any], pd.DataFrame]:
    """Load manifest, summary and cases table of a scored run.

    Parameters
    ----------
    run_id: str
        Run ID.

    Returns
    -------
    Tuple[Dict[str, Any], Dict[str, Any], pd.DataFrame]
        Manifest, summary, cases table.

    Raises
    ------
    FileNotFoundError
        If the run has not been scored.
    """
    rdir = run_dir(run_id)
    if not (rdir / c.CASES_TABLE_FILE).exists():
        raise FileNotFoundError(f"run {run_id} is not scored yet; run `score {run_id}` first")
    return (
        read_json(rdir / c.RUN_MANIFEST_FILE),
        read_json(rdir / c.SUMMARY_FILE),
        pd.read_parquet(rdir / c.CASES_TABLE_FILE),
    )


def compare_runs(
    run_a: str,
    run_b: str,
    samples: int = DEFAULT_SAMPLES,
    confidence: float = DEFAULT_CONFIDENCE,
    seed: int = DEFAULT_SEED,
    write: bool = True,
) -> Dict[str, Any]:
    """Compare run B against baseline run A on their shared case IDs.

    Parameters
    ----------
    run_a: str
        Baseline run.
    run_b: str
        Candidate run.
    samples: int
        Bootstrap resamples.
    confidence: float
        Confidence level.
    seed: int
        RNG seed.
    write: bool
        Write the result to ``reports/``.

    Returns
    -------
    Dict[str, Any]
        Comparison: warnings, quality deltas (overall and per category),
        operational changes and per-case diffs.
    """
    ma, sa, da = _load(run_a)
    mb, sb, db = _load(run_b)
    warnings: List[str] = []

    ha, hb = ma[c.RM_DATASET][c.RM_DATASET_HASH], mb[c.RM_DATASET][c.RM_DATASET_HASH]
    if ha != hb:
        warnings.append("dataset hashes differ: comparing only the intersection of case_ids")
    ja, jb = ma.get(c.RM_JUDGE) or {}, mb.get(c.RM_JUDGE) or {}
    if ja.get(c.RM_CRITERIA_HASH) != jb.get(c.RM_CRITERIA_HASH):
        warnings.append("judge configuration differs: LLM-judged scores are not directly comparable")
    if ma[c.RM_AGENT][c.RM_CORPUS_VERSION] != mb[c.RM_AGENT][c.RM_CORPUS_VERSION]:
        warnings.append("corpus versions differ: retrieval metrics reflect the corpus change")
    if ma.get(c.RM_PRICE_TABLE_VERSION) != mb.get(c.RM_PRICE_TABLE_VERSION):
        warnings.append("price table versions differ: re-score both runs to compare cost")

    da = da.set_index(c.CT_CASE_ID)
    db = db.set_index(c.CT_CASE_ID)
    shared = sorted(set(da.index) & set(db.index))
    da, db = da.loc[shared], db.loc[shared]
    categories = da[c.CT_CATEGORY]

    quality: List[Dict[str, Any]] = []
    for metric in c.ALL_METRICS:
        if metric not in da.columns or metric not in db.columns:
            continue
        stat = paired_stat(da[metric], db[metric], samples, confidence, seed)
        if stat:
            quality.append({c.CMP_METRIC: metric, c.CMP_SCOPE: c.SCOPE_OVERALL, **stat})
        for cat in c.CATEGORIES:
            idx = categories[categories == cat].index
            if len(idx) == 0:
                continue
            stat = paired_stat(da.loc[idx, metric], db.loc[idx, metric], samples, confidence, seed)
            if stat:
                quality.append({c.CMP_METRIC: metric, c.CMP_SCOPE: cat, **stat})

    ok = ~(da[c.CT_ERROR] | db[c.CT_ERROR])
    oa, ob = sa[c.SUM_OPERATIONAL], sb[c.SUM_OPERATIONAL]

    def op(name: str, a: Optional[float], b: Optional[float]) -> Dict[str, Any]:
        return {c.CMP_METRIC: name, c.CMP_VALUE_A: a, c.CMP_VALUE_B: b, c.CMP_REL_CHANGE: _rel(a, b)}

    def pct(df: pd.DataFrame, q: float) -> Optional[float]:
        vals = df.loc[ok, c.CT_LATENCY_S].dropna()
        return float(np.percentile(vals, q)) if len(vals) else None

    operational = [
        op(c.OP_AGENT_COST_MEAN, float(da.loc[ok, c.CT_AGENT_COST].mean()), float(db.loc[ok, c.CT_AGENT_COST].mean())),
        op(c.OP_LATENCY_P50, pct(da, 50), pct(db, 50)),
        op(c.OP_LATENCY_P95, pct(da, 95), pct(db, 95)),
        op(c.OP_TOKENS_IN, float(da.loc[ok, c.CT_INPUT_TOKENS].mean()), float(db.loc[ok, c.CT_INPUT_TOKENS].mean())),
        op(c.OP_TOKENS_OUT, float(da.loc[ok, c.CT_OUTPUT_TOKENS].mean()), float(db.loc[ok, c.CT_OUTPUT_TOKENS].mean())),
        op(c.OP_COST_PER_CORRECT, oa.get(c.SUM_COST_PER_CORRECT), ob.get(c.SUM_COST_PER_CORRECT)),
        op(c.OP_ERROR_RATE, sa[c.SUM_ERROR_RATE], sb[c.SUM_ERROR_RATE]),
    ]

    questions = {k.case_id: k.question for k in load_cases()}
    case_diffs = []
    for cid in shared:
        rec: Dict[str, Any] = {
            c.CT_CASE_ID: cid,
            c.CT_CATEGORY: da.loc[cid, c.CT_CATEGORY],
            c.CMP_QUESTION: questions.get(cid, ""),
            c.CMP_DELTAS: {},
        }
        for metric in c.ALL_METRICS:
            if metric in da.columns and metric in db.columns:
                va, vb = da.loc[cid, metric], db.loc[cid, metric]
                if pd.notna(va) and pd.notna(vb):
                    rec[c.CMP_DELTAS][metric] = float(vb) - float(va)
        reason_col = c.M_CORRECTNESS + c.CT_REASON_SUFFIX
        if reason_col in da.columns and reason_col in db.columns:
            rec[c.CMP_REASON_A] = da.loc[cid, reason_col]
            rec[c.CMP_REASON_B] = db.loc[cid, reason_col]
        case_diffs.append(rec)
    key = c.M_CORRECTNESS if c.M_CORRECTNESS in da.columns else c.M_BEHAVIOR_ACCURACY
    case_diffs.sort(key=lambda r: r[c.CMP_DELTAS].get(key, 0.0))

    result = {
        c.CMP_RUN_A: run_a,
        c.CMP_RUN_B: run_b,
        c.CMP_DATASET_MATCH: ha == hb,
        c.CMP_SHARED_CASES: len(shared),
        c.CMP_WARNINGS: warnings,
        c.CMP_QUALITY: quality,
        c.CMP_OPERATIONAL: operational,
        c.CMP_CASE_DIFFS: case_diffs,
        c.SUM_CURRENCY: oa.get(c.SUM_CURRENCY),
    }
    if write:
        c.REPORTS_DIR.mkdir(exist_ok=True)
        write_json(c.REPORTS_DIR / c.COMPARE_FILE_TEMPLATE.format(a=run_a, b=run_b), result)
    return result
