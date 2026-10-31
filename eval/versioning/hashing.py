"""Normalization and hashing for cases, the dataset and the core subset."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from typing import Any, Iterable, List

from eval import constants as c
from eval.versioning.schema import Case

_WS = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """Normalize text for hashing: Unicode NFC, trim, collapse whitespace.

    Case is preserved because in contracts it can carry meaning.

    Parameters
    ----------
    text: str
        Raw text.

    Returns
    -------
    str
        Normalized text.
    """
    return _WS.sub(" ", unicodedata.normalize(c.UNICODE_NORMAL_FORM, text)).strip()


def canonical_json(obj: Any) -> str:
    """Serialize to canonical JSON (sorted keys, fixed separators, UTF-8).

    Parameters
    ----------
    obj: Any
        JSON-serializable object.

    Returns
    -------
    str
        Canonical JSON string.
    """
    return json.dumps(obj, sort_keys=True, separators=c.JSON_SEPARATORS, ensure_ascii=False)


def sha256_hex(payload: str) -> str:
    """Return the hex SHA-256 of a string.

    Parameters
    ----------
    payload: str
        Input string.

    Returns
    -------
    str
        Hex digest.
    """
    return hashlib.sha256(payload.encode(c.TEXT_ENCODING)).hexdigest()


def content_hash(case: Case) -> str:
    """Hash the semantic fields of a case.

    Invariant to key order, whitespace and source order; changes when the
    question, expected output, sources or expected behaviour change.

    Parameters
    ----------
    case: Case
        Case to hash.

    Returns
    -------
    str
        ``sha256:<hex>``.
    """
    sources = sorted(
        [normalize_text(s.doc_id), normalize_text(s.passage_id)] for s in case.expected_sources
    )
    payload = {
        c.CASE_QUESTION: normalize_text(case.question),
        c.CASE_EXPECTED_OUTPUT: normalize_text(case.expected_output),
        c.CASE_EXPECTED_SOURCES: sources,
        c.CASE_EXPECTED_BEHAVIOR: case.expected_behavior,
    }
    return f"{c.HASH_ALGO}:{sha256_hex(canonical_json(payload))}"


def dataset_hash(cases: Iterable[Case]) -> str:
    """Hash the active cases as sorted (case_id, content_hash) pairs.

    Including ``case_id`` means a renumbering is detected.

    Parameters
    ----------
    cases: Iterable[Case]
        All cases; deprecated ones are ignored.

    Returns
    -------
    str
        ``sha256:<hex>``.
    """
    pairs = sorted((k.case_id, content_hash(k)) for k in cases if k.is_active)
    return f"{c.HASH_ALGO}:{sha256_hex(canonical_json(pairs))}"


def core_hash(core_ids: List[str], cases: Iterable[Case]) -> str:
    """Hash the core regression subset (IDs plus their content hashes).

    Parameters
    ----------
    core_ids: List[str]
        Core subset case IDs.
    cases: Iterable[Case]
        All cases.

    Returns
    -------
    str
        ``sha256:<hex>``.
    """
    by_id = {k.case_id: k for k in cases}
    pairs = sorted(
        (cid, content_hash(by_id[cid]) if cid in by_id else None) for cid in core_ids
    )
    return f"{c.HASH_ALGO}:{sha256_hex(canonical_json(pairs))}"
