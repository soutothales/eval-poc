"""Phase 2 — score: run deterministic and DeepEval metrics on stored outputs."""

from __future__ import annotations

import asyncio
import random
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import anthropic
from deepeval.metrics import (
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric,
    FaithfulnessMetric,
    GEval,
)
from deepeval.test_case import LLMTestCase, SingleTurnParams

from eval import constants as c
from eval import deterministic as det
from eval.io import append_jsonl, latest_by_key, read_json, read_jsonl, run_dir, write_json
from eval.judge import ClaudeJudge, load_judge_config
from eval.pricing import PriceTable
from eval.run_manifest import judge_section
from eval.summary import summarize
from eval.versioning.manifest import load_cases
from eval.versioning.schema import Case

JUDGE_RETRYABLE = (anthropic.RateLimitError, anthropic.APIConnectionError, anthropic.InternalServerError)
AGENT_ERROR = "agent_error"
NO_RETRIEVAL = "no passages retrieved"


def applicable_llm_metrics(case: Case, has_context: bool) -> List[str]:
    """LLM metrics that make sense for a case.

    Parameters
    ----------
    case: Case
        Golden case.
    has_context: bool
        Whether the agent retrieved any passage.

    Returns
    -------
    List[str]
        Metric names.
    """
    if case.expected_behavior != c.BEHAVIOR_ANSWER:
        return [c.M_CORRECTNESS]
    metrics = [c.M_ANSWER_RELEVANCY, c.M_CORRECTNESS]
    metrics += [c.M_CONTEXTUAL_PRECISION, c.M_CONTEXTUAL_RECALL, c.M_CONTEXTUAL_RELEVANCY]
    if has_context:
        metrics.append(c.M_FAITHFULNESS)
    return metrics


def build_metric(name: str, judge: ClaudeJudge, cfg: Dict[str, Any]):
    """Instantiate a DeepEval metric bound to a judge.

    Parameters
    ----------
    name: str
        Metric name constant.
    judge: ClaudeJudge
        Judge instance (one per case and metric, for token attribution).
    cfg: Dict[str, Any]
        Judge config.

    Returns
    -------
    BaseMetric
        Configured metric with ``include_reason`` where supported.
    """
    th = cfg[c.JCFG_THRESHOLDS][name]
    common = {"threshold": th, "model": judge, "async_mode": True}
    if name == c.M_ANSWER_RELEVANCY:
        return AnswerRelevancyMetric(include_reason=True, **common)
    if name == c.M_FAITHFULNESS:
        return FaithfulnessMetric(include_reason=True, **common)
    if name == c.M_CONTEXTUAL_PRECISION:
        return ContextualPrecisionMetric(include_reason=True, **common)
    if name == c.M_CONTEXTUAL_RECALL:
        return ContextualRecallMetric(include_reason=True, **common)
    if name == c.M_CONTEXTUAL_RELEVANCY:
        return ContextualRelevancyMetric(include_reason=True, **common)
    if name == c.M_CORRECTNESS:
        return GEval(
            name=c.GEVAL_CORRECTNESS_NAME,
            evaluation_steps=cfg[c.JCFG_CORRECTNESS_STEPS],
            evaluation_params=[
                SingleTurnParams.INPUT,
                SingleTurnParams.ACTUAL_OUTPUT,
                SingleTurnParams.EXPECTED_OUTPUT,
            ],
            **common,
        )
    raise ValueError(f"unknown LLM metric {name}")


def _row(out: Dict[str, Any], metric: str, score: Optional[float], **extra: Any) -> Dict[str, Any]:
    """Build a score row.

    Parameters
    ----------
    out: Dict[str, Any]
        Output row the score belongs to.
    metric: str
        Metric name.
    score: Optional[float]
        Score, or ``None`` if not computed.
    **extra: Any
        Additional score-row fields.

    Returns
    -------
    Dict[str, Any]
        Score row.
    """
    row = {
        c.SC_RUN_ID: out[c.OUT_RUN_ID],
        c.SC_CASE_ID: out[c.OUT_CASE_ID],
        c.SC_REPEAT_IDX: out[c.OUT_REPEAT_IDX],
        c.SC_CATEGORY: out[c.OUT_CATEGORY],
        c.SC_METRIC: metric,
        c.SC_SCORE: score,
        c.SC_SUCCESS: None,
        c.SC_THRESHOLD: None,
        c.SC_REASON: None,
        c.SC_ERROR: None,
        c.SC_JUDGE_INPUT_TOKENS: 0,
        c.SC_JUDGE_OUTPUT_TOKENS: 0,
    }
    row.update(extra)
    return row


def deterministic_rows(case: Case, out: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Compute deterministic metrics for one output row.

    Agent errors score 0 on answer-behaviour metrics so failures never
    inflate averages.

    Parameters
    ----------
    case: Case
        Golden case.
    out: Dict[str, Any]
        Output row.

    Returns
    -------
    List[Dict[str, Any]]
        Score rows (metrics that apply to the case only).
    """
    expected = [s.passage_id for s in case.expected_sources]
    retrieved = out.get(c.OUT_RETRIEVED_IDS) or []
    answer = out.get(c.OUT_ANSWER) or ""
    err = AGENT_ERROR if out.get(c.OUT_ERROR) else None
    values = {
        c.M_HIT_RATE: det.hit_rate(expected, retrieved),
        c.M_MRR: det.mrr(expected, retrieved),
        c.M_BEHAVIOR_ACCURACY: det.behavior_accuracy(case.expected_behavior, answer),
        c.M_FALSE_REFUSAL: det.false_refusal(case.expected_behavior, answer),
    }
    if err and values[c.M_FALSE_REFUSAL] is not None:
        values[c.M_FALSE_REFUSAL] = 0.0  # an error is not a refusal; it is counted in error_rate
    return [_row(out, m, v, error=err) for m, v in values.items() if v is not None]


async def _measure(
    metric_name: str,
    case: Case,
    out: Dict[str, Any],
    cfg: Dict[str, Any],
    sem: asyncio.Semaphore,
    max_attempts: int = 4,
) -> Dict[str, Any]:
    """Run one LLM metric with its own judge instance and retries.

    Parameters
    ----------
    metric_name: str
        Metric name.
    case: Case
        Golden case.
    out: Dict[str, Any]
        Output row.
    cfg: Dict[str, Any]
        Judge config.
    sem: asyncio.Semaphore
        Limits concurrent judge calls.
    max_attempts: int
        Attempts on retryable API errors.

    Returns
    -------
    Dict[str, Any]
        Score row; judge failures have ``score=None`` and an error.
    """
    context = out.get(c.OUT_RETRIEVAL_CONTEXT) or []
    th = cfg[c.JCFG_THRESHOLDS][metric_name]
    if not context and metric_name in (c.M_CONTEXTUAL_PRECISION, c.M_CONTEXTUAL_RECALL, c.M_CONTEXTUAL_RELEVANCY):
        return _row(out, metric_name, 0.0, success=False, threshold=th, reason=NO_RETRIEVAL)

    test_case = LLMTestCase(
        input=case.question,
        actual_output=out.get(c.OUT_ANSWER) or "",
        expected_output=case.expected_output,
        retrieval_context=context or None,
    )
    attempt = 0
    async with sem:
        while True:
            attempt += 1
            judge = ClaudeJudge.from_config(cfg)
            metric = build_metric(metric_name, judge, cfg)
            try:
                await metric.a_measure(test_case, _show_indicator=False)
                return _row(
                    out, metric_name, float(metric.score),
                    success=bool(metric.success), threshold=th, reason=metric.reason,
                    judge_input_tokens=judge.input_tokens, judge_output_tokens=judge.output_tokens,
                )
            except JUDGE_RETRYABLE as exc:
                if attempt >= max_attempts:
                    return _row(out, metric_name, None, threshold=th, error=f"judge: {type(exc).__name__}: {exc}")
                await asyncio.sleep(min(2 ** attempt, 30) + random.uniform(0, 1))
            except Exception as exc:  # noqa: BLE001 — recorded, never dropped
                return _row(
                    out, metric_name, None, threshold=th, error=f"judge: {type(exc).__name__}: {exc}",
                    judge_input_tokens=judge.input_tokens, judge_output_tokens=judge.output_tokens,
                )


def agent_error_rows(case: Case, out: Dict[str, Any], cfg: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Score rows for an output that errored: applicable LLM metrics score 0.

    Parameters
    ----------
    case: Case
        Golden case.
    out: Dict[str, Any]
        Errored output row.
    cfg: Optional[Dict[str, Any]]
        Judge config (for thresholds), or ``None``.

    Returns
    -------
    List[Dict[str, Any]]
        Zero-score rows tagged ``agent_error``.
    """
    rows = []
    for m in applicable_llm_metrics(case, has_context=True):
        th = cfg[c.JCFG_THRESHOLDS][m] if cfg else None
        rows.append(_row(out, m, 0.0, success=False, threshold=th, error=AGENT_ERROR))
    return rows


async def score_run(
    run_id: str,
    deterministic_only: bool = False,
    judge_config_path=c.JUDGE_CONFIG_PATH,
    rescore: bool = False,
    progress: Callable[[str], None] = print,
) -> Dict[str, Any]:
    """Score a run and write scores, summary and the cases table.

    Parameters
    ----------
    run_id: str
        Run to score.
    deterministic_only: bool
        Skip LLM-judged metrics (no API calls).
    judge_config_path: Path
        Judge config YAML.
    rescore: bool
        Discard existing scores and score everything again.
    progress: Callable[[str], None]
        Progress printer.

    Returns
    -------
    Dict[str, Any]
        Run summary.
    """
    rdir = run_dir(run_id)
    cfg = None if deterministic_only else load_judge_config(judge_config_path)
    cases = {k.case_id: k for k in load_cases()}
    outputs = latest_by_key(read_jsonl(rdir / c.OUTPUTS_FILE), [c.OUT_CASE_ID, c.OUT_REPEAT_IDX])
    scores_path = rdir / c.SCORES_FILE

    manifest = read_json(rdir / c.RUN_MANIFEST_FILE)
    new_judge = judge_section(cfg, deterministic_only)
    old_judge = manifest.get(c.RM_JUDGE)
    if not rescore and old_judge and old_judge != new_judge and scores_path.exists():
        raise ValueError("judge config changed since this run was scored; pass --rescore")
    if rescore and scores_path.exists():
        scores_path.unlink()

    done: Set[Tuple[str, int, str]] = {
        (r[c.SC_CASE_ID], r[c.SC_REPEAT_IDX], r[c.SC_METRIC])
        for r in read_jsonl(scores_path)
        if not (r.get(c.SC_ERROR) or "").startswith("judge:")
    }
    sem = asyncio.Semaphore(int(cfg[c.JCFG_CONCURRENCY]) if cfg else 1)

    async def score_one(out: Dict[str, Any]) -> List[Dict[str, Any]]:
        case = cases.get(out[c.OUT_CASE_ID])
        if case is None:
            return []
        rows = deterministic_rows(case, out)
        if cfg is not None:
            if out.get(c.OUT_ERROR):
                rows += agent_error_rows(case, out, cfg)
            else:
                names = applicable_llm_metrics(case, bool(out.get(c.OUT_RETRIEVAL_CONTEXT)))
                todo = [m for m in names if (case.case_id, out[c.OUT_REPEAT_IDX], m) not in done]
                rows += await asyncio.gather(*(_measure(m, case, out, cfg, sem) for m in todo))
        key = (out[c.OUT_CASE_ID], out[c.OUT_REPEAT_IDX])
        return [r for r in rows if (key[0], key[1], r[c.SC_METRIC]) not in done]

    tasks = [asyncio.create_task(score_one(o)) for o in outputs]
    n = 0
    for fut in asyncio.as_completed(tasks):
        rows = await fut
        append_jsonl(scores_path, rows)
        n += 1
        if n % 5 == 0 or n == len(tasks):
            progress(f"[score] {n}/{len(tasks)} outputs scored")

    prices = PriceTable.load()
    manifest[c.RM_JUDGE] = new_judge
    manifest[c.RM_PRICE_TABLE_VERSION] = prices.version
    write_json(rdir / c.RUN_MANIFEST_FILE, manifest)

    summary = summarize(run_id, prices)
    progress(f"[score] summary written to {rdir / c.SUMMARY_FILE}")
    return summary
