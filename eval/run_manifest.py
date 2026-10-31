"""Run manifest (fingerprint): everything needed to reproduce or compare a run."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import uuid
from importlib import metadata
from typing import Any, Dict, Optional

from agent.graph import AgentConfig, prompt_hash
from agent.retriever import corpus_version
from eval import constants as c
from eval import gitutil
from eval.versioning.manifest import load_cases, load_core, load_manifest
from eval.versioning.hashing import core_hash, dataset_hash

# Packages whose versions affect results.
_PINNED_PACKAGES = (
    "anthropic",
    "langchain-anthropic",
    "langchain-core",
    "langgraph",
    "deepeval",
    "rank-bm25",
    "snowballstemmer",
)


def dependency_hash() -> str:
    """Hash the installed versions of result-relevant packages.

    Returns
    -------
    str
        ``sha256:<16 hex>`` over ``name==version`` lines.
    """
    lines = []
    for pkg in _PINNED_PACKAGES:
        try:
            lines.append(f"{pkg}=={metadata.version(pkg)}")
        except metadata.PackageNotFoundError:
            lines.append(f"{pkg}==missing")
    return "sha256:" + hashlib.sha256("\n".join(lines).encode(c.TEXT_ENCODING)).hexdigest()[:16]


def new_run_id(config_name: str) -> str:
    """Create a sortable run ID.

    Parameters
    ----------
    config_name: str
        Agent config name.

    Returns
    -------
    str
        ``<UTC timestamp>-<config name>-<4 hex>``; the suffix keeps runs
        started in the same second distinct.
    """
    stamp = dt.datetime.now(dt.timezone.utc).strftime(c.RUN_ID_TIME_FORMAT)
    return f"{stamp}-{config_name}-{uuid.uuid4().hex[:4]}"


def build_run_manifest(run_id: str, config: AgentConfig, options: Dict[str, Any]) -> Dict[str, Any]:
    """Build the generate-phase run manifest.

    The judge section and price table version are added by the score phase.

    Parameters
    ----------
    run_id: str
        Run ID.
    config: AgentConfig
        Agent configuration.
    options: Dict[str, Any]
        Run options (repeats, limit, core_only, concurrency).

    Returns
    -------
    Dict[str, Any]
        Manifest dict.
    """
    cases = load_cases()
    core = load_core()
    dmanifest = load_manifest()
    current_hash = dataset_hash(cases)
    agent = config.to_dict()
    agent[c.RM_PROMPT_HASH] = prompt_hash(config.prompt_file)
    agent[c.RM_CORPUS_VERSION] = corpus_version()
    return {
        c.RM_RUN_ID: run_id,
        c.RM_TIMESTAMP: dt.datetime.now(dt.timezone.utc).isoformat(),
        c.RM_GIT_COMMIT: gitutil.head_commit(),
        c.RM_DATASET: {
            c.RM_DATASET_VERSION: dmanifest.version if dmanifest and dmanifest.dataset_hash == current_hash else "unreleased",
            c.RM_DATASET_HASH: current_hash,
            c.RM_CORE_HASH: core_hash(core.case_ids, cases),
        },
        c.RM_AGENT: agent,
        c.RM_JUDGE: None,
        c.RM_PRICE_TABLE_VERSION: None,
        c.RM_DEPENDENCY_HASH: dependency_hash(),
        c.RM_RUN_OPTIONS: options,
    }


def criteria_hash(judge_cfg: Dict[str, Any]) -> str:
    """Hash everything about the judge that changes scores.

    Parameters
    ----------
    judge_cfg: Dict[str, Any]
        Parsed judge config.

    Returns
    -------
    str
        ``sha256:<16 hex>`` over model, temperature, thresholds and steps.
    """
    payload = {
        c.JCFG_MODEL_ID: judge_cfg[c.JCFG_MODEL_ID],
        c.JCFG_TEMPERATURE: judge_cfg.get(c.JCFG_TEMPERATURE),
        c.JCFG_THRESHOLDS: judge_cfg[c.JCFG_THRESHOLDS],
        c.JCFG_CORRECTNESS_STEPS: judge_cfg[c.JCFG_CORRECTNESS_STEPS],
    }
    blob = json.dumps(payload, sort_keys=True).encode(c.TEXT_ENCODING)
    return "sha256:" + hashlib.sha256(blob).hexdigest()[:16]


def judge_section(judge_cfg: Optional[Dict[str, Any]], deterministic_only: bool) -> Dict[str, Any]:
    """Build the judge section of the run manifest.

    Parameters
    ----------
    judge_cfg: Optional[Dict[str, Any]]
        Parsed judge config, or ``None`` in deterministic-only mode.
    deterministic_only: bool
        Whether LLM metrics were skipped.

    Returns
    -------
    Dict[str, Any]
        Judge fingerprint.
    """
    if deterministic_only or judge_cfg is None:
        return {"mode": "deterministic_only"}
    return {
        c.JCFG_MODEL_ID: judge_cfg[c.JCFG_MODEL_ID],
        c.JCFG_TEMPERATURE: judge_cfg.get(c.JCFG_TEMPERATURE),
        c.JCFG_THRESHOLDS: judge_cfg[c.JCFG_THRESHOLDS],
        c.RM_CRITERIA_HASH: criteria_hash(judge_cfg),
    }
