"""Deterministic metrics: retrieval hit rate, MRR, behaviour accuracy, false refusals."""

from __future__ import annotations

from typing import List, Optional

from agent import constants as ac
from eval import constants as c


def is_not_found(answer: str) -> bool:
    """Whether the agent declared the information missing.

    Parameters
    ----------
    answer: str
        Agent answer.

    Returns
    -------
    bool
        True when the answer starts with, or contains, the not-found marker.
    """
    head = answer.strip()
    return head.startswith(ac.NOT_FOUND_MARKER) or f"{ac.NOT_FOUND_MARKER}:" in head


def hit_rate(expected: List[str], retrieved: List[str]) -> Optional[float]:
    """Fraction of expected passages present in the retrieved set (recall@k).

    Parameters
    ----------
    expected: List[str]
        Expected passage IDs.
    retrieved: List[str]
        Retrieved passage IDs, rank order.

    Returns
    -------
    Optional[float]
        Value in [0, 1], or ``None`` when nothing is expected.
    """
    if not expected:
        return None
    return len(set(expected) & set(retrieved)) / len(set(expected))


def mrr(expected: List[str], retrieved: List[str]) -> Optional[float]:
    """Reciprocal rank of the first expected passage.

    Parameters
    ----------
    expected: List[str]
        Expected passage IDs.
    retrieved: List[str]
        Retrieved passage IDs, rank order.

    Returns
    -------
    Optional[float]
        ``1/rank`` or 0, or ``None`` when nothing is expected.
    """
    if not expected:
        return None
    wanted = set(expected)
    for rank, pid in enumerate(retrieved, start=1):
        if pid in wanted:
            return 1.0 / rank
    return 0.0


def behavior_accuracy(expected_behavior: str, answer: str) -> Optional[float]:
    """Whether the agent answered vs said not-found as expected.

    Decline cases are left to the LLM correctness metric.

    Parameters
    ----------
    expected_behavior: str
        ``answer``, ``not_found`` or ``decline``.
    answer: str
        Agent answer.

    Returns
    -------
    Optional[float]
        1.0 correct, 0.0 wrong, ``None`` for decline cases.
    """
    if expected_behavior == c.BEHAVIOR_NOT_FOUND:
        return float(is_not_found(answer))
    if expected_behavior == c.BEHAVIOR_ANSWER:
        return float(bool(answer.strip()) and not is_not_found(answer))
    return None


def false_refusal(expected_behavior: str, answer: str) -> Optional[float]:
    """Whether the agent said not-found on an answerable question (lower is better).

    Parameters
    ----------
    expected_behavior: str
        Expected behaviour.
    answer: str
        Agent answer.

    Returns
    -------
    Optional[float]
        1.0 for a false refusal, 0.0 otherwise, ``None`` for non-answer cases.
    """
    if expected_behavior != c.BEHAVIOR_ANSWER:
        return None
    return float(is_not_found(answer))
