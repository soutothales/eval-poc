"""Per-run aggregates (quality, cost, latency) and the per-case table."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from eval import constants as c
from eval.io import latest_by_key, read_jsonl, run_dir, write_json
from eval.pricing import PriceTable

N_JUDGE_ERRORS = "n_judge_errors"
CORRECTNESS_PROXY = "correct_answer_metric"


def _stats(values: List[float]) -> Dict[str, Optional[float]]:
    """Mean, p50 and p95 of a list.

    Parameters
    ----------
    values: List[float]
        Values.

    Returns
    -------
    Dict[str, Optional[float]]
        ``mean``, ``p50``, ``p95`` (``None`` when empty).
    """
    if not values:
        return {c.SUM_MEAN: None, c.SUM_P50: None, c.SUM_P95: None}
    arr = np.asarray(values, dtype=float)
    return {
        c.SUM_MEAN: float(arr.mean()),
        c.SUM_P50: float(np.percentile(arr, 50)),
        c.SUM_P95: float(np.percentile(arr, 95)),
    }


def build_cases_table(run_id: str, prices: PriceTable) -> Tuple[pd.DataFrame, Dict[str, Dict[str, int]]]:
    """One row per case (repeat 0) with scores, reasons, tokens, cost and latency.

    Parameters
    ----------
    run_id: str
        Run ID.
    prices: PriceTable
        Price table used for agent cost.

    Returns
    -------
    Tuple[pd.DataFrame, Dict[str, Dict[str, int]]]
        Cases table, and judge token totals per case_id.
    """
    rdir = run_dir(run_id)
    outputs = latest_by_key(read_jsonl(rdir / c.OUTPUTS_FILE), [c.OUT_CASE_ID, c.OUT_REPEAT_IDX])
    scores = latest_by_key(read_jsonl(rdir / c.SCORES_FILE), [c.SC_CASE_ID, c.SC_REPEAT_IDX, c.SC_METRIC])
    judge_tokens: Dict[str, Dict[str, int]] = {}
    by_case: Dict[str, Dict[str, Any]] = {}

    for o in outputs:
        if o[c.OUT_REPEAT_IDX] != 0:
            continue
        usage = o.get(c.OUT_USAGE) or {}
        cost, _ = prices.usage_cost(usage)
        by_case[o[c.OUT_CASE_ID]] = {
            c.CT_RUN_ID: run_id,
            c.CT_CASE_ID: o[c.OUT_CASE_ID],
            c.CT_CATEGORY: o[c.OUT_CATEGORY],
            c.CT_LATENCY_S: o.get(c.OUT_LATENCY_S),
            c.CT_INPUT_TOKENS: sum(u.get(c.USAGE_INPUT_TOKENS, 0) for u in usage.values()),
            c.CT_OUTPUT_TOKENS: sum(u.get(c.USAGE_OUTPUT_TOKENS, 0) for u in usage.values()),
            c.CT_AGENT_COST: cost,
            c.CT_JUDGE_COST: 0.0,
            c.CT_ERROR: bool(o.get(c.OUT_ERROR)),
        }
    for s in scores:
        if s[c.SC_REPEAT_IDX] != 0 or s[c.SC_CASE_ID] not in by_case:
            continue
        row = by_case[s[c.SC_CASE_ID]]
        row[s[c.SC_METRIC]] = s[c.SC_SCORE]
        row[s[c.SC_METRIC] + c.CT_REASON_SUFFIX] = s.get(c.SC_REASON) or s.get(c.SC_ERROR)
        jt = judge_tokens.setdefault(s[c.SC_CASE_ID], {c.USAGE_INPUT_TOKENS: 0, c.USAGE_OUTPUT_TOKENS: 0})
        jt[c.USAGE_INPUT_TOKENS] += int(s.get(c.SC_JUDGE_INPUT_TOKENS) or 0)
        jt[c.USAGE_OUTPUT_TOKENS] += int(s.get(c.SC_JUDGE_OUTPUT_TOKENS) or 0)
    return pd.DataFrame(list(by_case.values())), judge_tokens


def summarize(run_id: str, prices: PriceTable) -> Dict[str, Any]:
    """Aggregate a scored run and write ``summary.json`` and ``cases.parquet``.

    Parameters
    ----------
    run_id: str
        Run ID.
    prices: PriceTable
        Price table.

    Returns
    -------
    Dict[str, Any]
        Summary dict.
    """
    from eval.io import read_json  # local import keeps module import cheap

    rdir = run_dir(run_id)
    manifest = read_json(rdir / c.RUN_MANIFEST_FILE)
    judge_model = (manifest.get(c.RM_JUDGE) or {}).get(c.JCFG_MODEL_ID)
    df, judge_tokens = build_cases_table(run_id, prices)
    for cid, jt in judge_tokens.items():
        if judge_model and prices.resolve(judge_model):
            cost = prices.call_cost(prices.resolve(judge_model), jt[c.USAGE_INPUT_TOKENS], jt[c.USAGE_OUTPUT_TOKENS])
            df.loc[df[c.CT_CASE_ID] == cid, c.CT_JUDGE_COST] = cost

    scores = latest_by_key(read_jsonl(rdir / c.SCORES_FILE), [c.SC_CASE_ID, c.SC_REPEAT_IDX, c.SC_METRIC])
    s0 = [s for s in scores if s[c.SC_REPEAT_IDX] == 0]

    quality: Dict[str, Any] = {}
    n_judge_errors = 0
    for metric in c.ALL_METRICS:
        rows = [s for s in s0 if s[c.SC_METRIC] == metric]
        n_judge_errors += sum(1 for s in rows if s[c.SC_SCORE] is None)
        valid = [s for s in rows if s[c.SC_SCORE] is not None]
        if not valid:
            continue

        def agg(items: List[Dict[str, Any]]) -> Dict[str, Any]:
            vals = [float(i[c.SC_SCORE]) for i in items]
            succ = [i[c.SC_SUCCESS] for i in items if i[c.SC_SUCCESS] is not None]
            return {
                c.SUM_MEAN: float(np.mean(vals)) if vals else None,
                c.SUM_PASS_RATE: float(np.mean(succ)) if succ else None,
                c.SUM_N: len(vals),
            }

        cats = sorted({s[c.SC_CATEGORY] for s in valid})
        quality[metric] = {
            c.SUM_OVERALL: agg(valid),
            c.SUM_BY_CATEGORY: {k: agg([s for s in valid if s[c.SC_CATEGORY] == k]) for k in cats},
        }

    n = len(df)
    n_err = int(df[c.CT_ERROR].sum()) if n else 0
    ok = df[~df[c.CT_ERROR]] if n else df
    agent_total = float(df[c.CT_AGENT_COST].sum()) if n else 0.0

    proxy = c.M_CORRECTNESS if c.M_CORRECTNESS in df.columns else c.M_BEHAVIOR_ACCURACY
    if proxy == c.M_CORRECTNESS:
        th = (manifest.get(c.RM_JUDGE) or {}).get(c.JCFG_THRESHOLDS, {}).get(c.M_CORRECTNESS, 0.5)
        n_correct = int((df[proxy].fillna(0) >= th).sum()) if n else 0
    else:
        n_correct = int((df[proxy].fillna(0) >= 1.0).sum()) if proxy in df.columns else 0

    unpriced: List[str] = []
    for o in latest_by_key(read_jsonl(rdir / c.OUTPUTS_FILE), [c.OUT_CASE_ID, c.OUT_REPEAT_IDX]):
        unpriced += prices.usage_cost(o.get(c.OUT_USAGE) or {})[1]

    operational = {
        c.SUM_LATENCY: _stats(ok[c.CT_LATENCY_S].dropna().tolist() if n else []),
        c.SUM_AGENT_COST: _stats(ok[c.CT_AGENT_COST].tolist() if n else []),
        c.SUM_TOKENS_IN: _stats(ok[c.CT_INPUT_TOKENS].tolist() if n else [])[c.SUM_MEAN],
        c.SUM_TOKENS_OUT: _stats(ok[c.CT_OUTPUT_TOKENS].tolist() if n else [])[c.SUM_MEAN],
        c.SUM_AGENT_COST_TOTAL: agent_total,
        c.SUM_JUDGE_COST_TOTAL: float(df[c.CT_JUDGE_COST].sum()) if n else 0.0,
        c.SUM_COST_PER_CORRECT: (agent_total / n_correct) if n_correct else None,
        CORRECTNESS_PROXY: proxy,
        c.SUM_CURRENCY: prices.currency,
        c.SUM_UNPRICED_MODELS: sorted(set(unpriced)),
    }
    summary = {
        c.SUM_RUN_ID: run_id,
        c.SUM_N_CASES: n,
        c.SUM_N_ERRORS: n_err,
        c.SUM_ERROR_RATE: (n_err / n) if n else 0.0,
        N_JUDGE_ERRORS: n_judge_errors,
        c.SUM_QUALITY: quality,
        c.SUM_OPERATIONAL: operational,
    }
    write_json(rdir / c.SUMMARY_FILE, summary)
    if n:
        df.to_parquet(rdir / c.CASES_TABLE_FILE, index=False)
    return summary
