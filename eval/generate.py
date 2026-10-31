"""Phase 1 — generate: call the agent for every case and persist raw outputs."""

from __future__ import annotations

import asyncio
import datetime as dt
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import anthropic

from agent import constants as ac
from agent.graph import AgentConfig
from eval import constants as c
from eval.adapters.agent_adapter import AgentAdapter, AgentResult
from eval.io import append_jsonl, read_json, read_jsonl, write_json
from eval.run_manifest import build_run_manifest, new_run_id
from eval.versioning.manifest import load_cases, load_core
from eval.versioning.schema import Case

RETRYABLE = (
    anthropic.RateLimitError,
    anthropic.APIConnectionError,
    anthropic.InternalServerError,
    asyncio.TimeoutError,
)
FATAL = (anthropic.AuthenticationError, anthropic.PermissionDeniedError, anthropic.NotFoundError)


class FatalRunError(RuntimeError):
    """Raised when an error will affect every case (bad key, unknown model)."""


def select_work(
    cases: List[Case],
    core_ids: List[str],
    repeats: int,
    limit: Optional[int],
    core_only: bool,
) -> List[Tuple[Case, int]]:
    """Choose (case, repeat_idx) pairs to run.

    Every selected case runs once (repeat 0). Core cases run ``repeats``
    times in total, to measure agent nondeterminism.

    Parameters
    ----------
    cases: List[Case]
        All cases.
    core_ids: List[str]
        Core regression subset.
    repeats: int
        Total runs per core case (1 = no repeats).
    limit: Optional[int]
        Keep only the first N active cases.
    core_only: bool
        Run only the core subset.

    Returns
    -------
    List[Tuple[Case, int]]
        Work items.
    """
    active = [k for k in cases if k.is_active]
    if core_only:
        active = [k for k in active if k.case_id in core_ids]
    if limit:
        active = active[:limit]
    work = [(k, 0) for k in active]
    for k in active:
        if k.case_id in core_ids:
            work.extend((k, r) for r in range(1, max(repeats, 1)))
    return work


async def _run_one(
    adapter: AgentAdapter,
    case: Case,
    repeat_idx: int,
    run_id: str,
    sem: asyncio.Semaphore,
    timeout_s: float,
    max_attempts: int,
) -> Dict[str, Any]:
    """Run one work item with timeout and exponential backoff.

    Parameters
    ----------
    adapter: AgentAdapter
        Agent adapter.
    case: Case
        Case to run.
    repeat_idx: int
        Repeat index.
    run_id: str
        Run ID.
    sem: asyncio.Semaphore
        Concurrency limiter.
    timeout_s: float
        Per-attempt timeout.
    max_attempts: int
        Maximum attempts for retryable errors.

    Returns
    -------
    Dict[str, Any]
        Output row; errors are recorded, never dropped.

    Raises
    ------
    FatalRunError
        On errors that would affect every case.
    """
    attempt = 0
    result: AgentResult
    async with sem:
        while True:
            attempt += 1
            try:
                result = await asyncio.wait_for(adapter.ainvoke(case.question), timeout=timeout_s)
                break
            except FATAL as exc:
                raise FatalRunError(f"{type(exc).__name__}: {exc}") from exc
            except RETRYABLE as exc:
                if attempt >= max_attempts:
                    result = AgentResult(error=f"{type(exc).__name__}: {exc}")
                    break
                await asyncio.sleep(min(2 ** attempt, 30) + random.uniform(0, 1))
            except Exception as exc:  # noqa: BLE001 — recorded as an error row
                result = AgentResult(error=f"{type(exc).__name__}: {exc}")
                break
    row = {
        c.OUT_RUN_ID: run_id,
        c.OUT_CASE_ID: case.case_id,
        c.OUT_REPEAT_IDX: repeat_idx,
        c.OUT_CATEGORY: case.category,
        c.OUT_ATTEMPTS: attempt,
        c.OUT_TIMESTAMP: dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    row.update(result.to_dict())
    return row


async def generate(
    config_path: Path,
    repeats: int = 1,
    limit: Optional[int] = None,
    core_only: bool = False,
    concurrency: int = 4,
    resume_run_id: Optional[str] = None,
    retry_errors: bool = False,
    max_attempts: int = 5,
) -> str:
    """Run the generate phase and return the run ID.

    Outputs are appended to ``outputs.jsonl`` after each case, so a crash
    resumes with ``resume_run_id`` instead of restarting.

    Parameters
    ----------
    config_path: Path
        Agent config YAML.
    repeats: int
        Total runs per core case.
    limit: Optional[int]
        Only the first N active cases.
    core_only: bool
        Only the core regression subset.
    concurrency: int
        Parallel agent calls.
    resume_run_id: Optional[str]
        Existing run to resume.
    retry_errors: bool
        When resuming, re-run cases whose last row is an error.
    max_attempts: int
        Attempts per case for retryable errors.

    Returns
    -------
    str
        Run ID.
    """
    config = AgentConfig.from_yaml(config_path)
    options = {
        c.RM_REPEATS: repeats,
        c.RM_LIMIT: limit,
        c.RM_CORE_ONLY: core_only,
        c.RM_CONCURRENCY: concurrency,
    }
    if resume_run_id:
        run_id = resume_run_id
        rdir = c.RUNS_DIR / run_id
        manifest = read_json(rdir / c.RUN_MANIFEST_FILE)
        if manifest[c.RM_AGENT][ac.CFG_MODEL_ID] != config.model_id:
            raise ValueError("resume config does not match the run's agent config")
    else:
        run_id = new_run_id(config.name)
        rdir = c.RUNS_DIR / run_id
        rdir.mkdir(parents=True, exist_ok=False)
        write_json(rdir / c.RUN_MANIFEST_FILE, build_run_manifest(run_id, config, options))

    out_path = rdir / c.OUTPUTS_FILE
    done: Set[Tuple[str, int]] = set()
    last: Dict[Tuple[str, int], Dict[str, Any]] = {}
    for row in read_jsonl(out_path):
        last[(row[c.OUT_CASE_ID], row[c.OUT_REPEAT_IDX])] = row
    for key, row in last.items():
        if not (retry_errors and row.get(c.OUT_ERROR)):
            done.add(key)

    work = [
        (k, r)
        for k, r in select_work(load_cases(), load_core().case_ids, repeats, limit, core_only)
        if (k.case_id, r) not in done
    ]
    adapter = AgentAdapter(config)
    sem = asyncio.Semaphore(concurrency)
    total = len(work)
    print(f"[generate] run {run_id}: {total} items ({len(done)} already done), concurrency {concurrency}")

    tasks = [
        asyncio.create_task(_run_one(adapter, k, r, run_id, sem, config.timeout_s, max_attempts))
        for k, r in work
    ]
    n_done = n_err = 0
    try:
        for fut in asyncio.as_completed(tasks):
            row = await fut
            append_jsonl(out_path, [row])
            n_done += 1
            n_err += bool(row.get(c.OUT_ERROR))
            if n_done % 5 == 0 or n_done == total:
                print(f"[generate] {n_done}/{total} done, {n_err} errors")
    except FatalRunError:
        for t in tasks:
            t.cancel()
        raise
    print(f"[generate] finished run {run_id}: {n_err} error rows recorded")
    return run_id
