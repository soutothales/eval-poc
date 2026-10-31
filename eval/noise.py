"""Noise measurement: judge variance (re-scoring) and agent variance (repeats)."""

from __future__ import annotations

import asyncio
import random
from typing import Any, Dict, List

import numpy as np

from eval import constants as c
from eval.io import latest_by_key, read_jsonl, run_dir, write_json
from eval.judge import load_judge_config
from eval.score import _measure, applicable_llm_metrics
from eval.versioning.manifest import load_cases

NOISE_FILE = "noise.json"
JUDGE_NOISE = "judge_noise"
AGENT_NOISE = "agent_noise"
MEAN_STD = "mean_std"
MAX_RANGE = "max_range"
N_CASES = "n_cases"
TIMES = "times"


def _spread(groups: Dict[Any, List[float]]) -> Dict[str, float]:
    """Mean per-group standard deviation and max per-group range.

    Parameters
    ----------
    groups: Dict[Any, List[float]]
        Repeated measurements per item.

    Returns
    -------
    Dict[str, float]
        ``mean_std``, ``max_range``, ``n_cases``.
    """
    groups = {k: v for k, v in groups.items() if len(v) >= 2}
    if not groups:
        return {MEAN_STD: 0.0, MAX_RANGE: 0.0, N_CASES: 0}
    stds = [float(np.std(v, ddof=1)) for v in groups.values()]
    ranges = [max(v) - min(v) for v in groups.values()]
    return {MEAN_STD: float(np.mean(stds)), MAX_RANGE: float(max(ranges)), N_CASES: len(groups)}


async def judge_noise(run_id: str, times: int = 3, sample: int = 8, seed: int = 7) -> Dict[str, Any]:
    """Re-score a sample of outputs several times with the same judge config.

    Nothing is written to ``scores.jsonl``; the result goes to ``noise.json``.
    Also reports agent variance from scored repeats, if the run has them.

    Parameters
    ----------
    run_id: str
        Run to sample from.
    times: int
        Judgments per (case, metric).
    sample: int
        Number of answerable, error-free cases to sample.
    seed: int
        Sampling seed.

    Returns
    -------
    Dict[str, Any]
        Per-metric judge and agent spread.
    """
    rdir = run_dir(run_id)
    cfg = load_judge_config()
    cases = {k.case_id: k for k in load_cases()}
    outputs = [
        o for o in latest_by_key(read_jsonl(rdir / c.OUTPUTS_FILE), [c.OUT_CASE_ID, c.OUT_REPEAT_IDX])
        if o[c.OUT_REPEAT_IDX] == 0 and not o.get(c.OUT_ERROR)
        and cases.get(o[c.OUT_CASE_ID]) and cases[o[c.OUT_CASE_ID]].expected_behavior == c.BEHAVIOR_ANSWER
    ]
    random.Random(seed).shuffle(outputs)
    outputs = outputs[:sample]
    sem = asyncio.Semaphore(int(cfg[c.JCFG_CONCURRENCY]))

    jobs = []
    for o in outputs:
        case = cases[o[c.OUT_CASE_ID]]
        for m in applicable_llm_metrics(case, bool(o.get(c.OUT_RETRIEVAL_CONTEXT))):
            for _ in range(times):
                jobs.append(_measure(m, case, o, cfg, sem))
    rows = await asyncio.gather(*jobs)

    judge_groups: Dict[str, Dict[str, List[float]]] = {}
    for r in rows:
        if r[c.SC_SCORE] is not None:
            judge_groups.setdefault(r[c.SC_METRIC], {}).setdefault(r[c.SC_CASE_ID], []).append(r[c.SC_SCORE])

    agent_groups: Dict[str, Dict[str, List[float]]] = {}
    for s in latest_by_key(read_jsonl(rdir / c.SCORES_FILE), [c.SC_CASE_ID, c.SC_REPEAT_IDX, c.SC_METRIC]):
        if s[c.SC_SCORE] is not None:
            agent_groups.setdefault(s[c.SC_METRIC], {}).setdefault(s[c.SC_CASE_ID], []).append(s[c.SC_SCORE])

    result = {
        c.SUM_RUN_ID: run_id,
        TIMES: times,
        JUDGE_NOISE: {m: _spread(g) for m, g in sorted(judge_groups.items())},
        AGENT_NOISE: {m: _spread(g) for m, g in sorted(agent_groups.items())},
    }
    write_json(rdir / NOISE_FILE, result)
    return result
