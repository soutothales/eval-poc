"""Load and save cases, the core subset and the dataset manifest."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import List, Optional

from eval import constants as c
from eval.versioning.hashing import content_hash, core_hash, dataset_hash
from eval.versioning.schema import (
    Case,
    ChangelogEntry,
    CoreRegression,
    Counts,
    DatasetManifest,
)


def parse_cases(text: str) -> List[Case]:
    """Parse JSONL text into cases.

    Parameters
    ----------
    text: str
        JSONL content, one case per line.

    Returns
    -------
    List[Case]
        Parsed cases in file order.
    """
    return [Case.model_validate_json(line) for line in text.splitlines() if line.strip()]


def load_cases(path: Path = c.CASES_PATH) -> List[Case]:
    """Load cases from a JSONL file.

    Parameters
    ----------
    path: Path
        Cases file.

    Returns
    -------
    List[Case]
        Cases in file order.
    """
    return parse_cases(Path(path).read_text(encoding=c.TEXT_ENCODING))


def save_cases(cases: List[Case], path: Path = c.CASES_PATH) -> None:
    """Write cases as JSONL, one canonical line per case.

    Parameters
    ----------
    cases: List[Case]
        Cases to write, in order.
    path: Path
        Destination file.
    """
    lines = [json.dumps(k.model_dump(mode="json"), ensure_ascii=False) for k in cases]
    Path(path).write_text("\n".join(lines) + "\n", encoding=c.TEXT_ENCODING)


def rehash_cases(cases: List[Case]) -> List[Case]:
    """Return copies of the cases with ``content_hash`` recomputed.

    Parameters
    ----------
    cases: List[Case]
        Input cases.

    Returns
    -------
    List[Case]
        Cases with fresh hashes.
    """
    return [k.model_copy(update={c.CASE_CONTENT_HASH: content_hash(k)}) for k in cases]


def load_core(path: Path = c.CORE_REGRESSION_PATH) -> CoreRegression:
    """Load the core regression subset.

    Parameters
    ----------
    path: Path
        ``core_regression.json``.

    Returns
    -------
    CoreRegression
        Core subset.
    """
    return CoreRegression.model_validate_json(Path(path).read_text(encoding=c.TEXT_ENCODING))


def load_manifest(path: Path = c.DATASET_MANIFEST_PATH) -> Optional[DatasetManifest]:
    """Load the dataset manifest if it exists.

    Parameters
    ----------
    path: Path
        ``manifest.json``.

    Returns
    -------
    Optional[DatasetManifest]
        Manifest, or ``None`` if the file does not exist.
    """
    if not Path(path).exists():
        return None
    return DatasetManifest.model_validate_json(Path(path).read_text(encoding=c.TEXT_ENCODING))


def save_manifest(manifest: DatasetManifest, path: Path = c.DATASET_MANIFEST_PATH) -> None:
    """Write the dataset manifest as indented JSON.

    Parameters
    ----------
    manifest: DatasetManifest
        Manifest to write.
    path: Path
        Destination file.
    """
    Path(path).write_text(manifest.model_dump_json(indent=2) + "\n", encoding=c.TEXT_ENCODING)


def build_manifest(
    cases: List[Case],
    core: CoreRegression,
    version: str,
    corpus_version: str,
    changelog: List[ChangelogEntry],
) -> DatasetManifest:
    """Compute a manifest from the current cases.

    Parameters
    ----------
    cases: List[Case]
        All cases.
    core: CoreRegression
        Core regression subset.
    version: str
        Semantic version to record.
    corpus_version: str
        Corpus fingerprint.
    changelog: List[ChangelogEntry]
        Full changelog including the new entry.

    Returns
    -------
    DatasetManifest
        New manifest.
    """
    active = [k for k in cases if k.is_active]
    counts = Counts(
        by_category=dict(sorted(Counter(k.category for k in active).items())),
        by_status=dict(sorted(Counter(k.status for k in cases).items())),
    )
    return DatasetManifest(
        schema_version=c.CURRENT_SCHEMA_VERSION,
        version=version,
        dataset_hash=dataset_hash(cases),
        corpus_version=corpus_version,
        core_regression_hash=core_hash(core.case_ids, cases),
        counts=counts,
        changelog=changelog,
    )
