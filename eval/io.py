"""File helpers for run directories (JSON, JSONL, run paths)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

from eval import constants as c


def run_dir(run_id: str) -> Path:
    """Directory of a run.

    Parameters
    ----------
    run_id: str
        Run ID.

    Returns
    -------
    Path
        ``runs/<run_id>``.

    Raises
    ------
    FileNotFoundError
        If the run does not exist.
    """
    path = c.RUNS_DIR / run_id
    if not path.exists():
        raise FileNotFoundError(f"run not found: {path}")
    return path


def read_json(path: Path) -> Any:
    """Read a JSON file.

    Parameters
    ----------
    path: Path
        File path.

    Returns
    -------
    Any
        Parsed JSON.
    """
    return json.loads(Path(path).read_text(encoding=c.TEXT_ENCODING))


def write_json(path: Path, obj: Any) -> None:
    """Write indented JSON.

    Parameters
    ----------
    path: Path
        Destination.
    obj: Any
        JSON-serializable object.
    """
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str) + "\n", encoding=c.TEXT_ENCODING)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Read a JSONL file; missing file yields an empty list.

    Parameters
    ----------
    path: Path
        File path.

    Returns
    -------
    List[Dict[str, Any]]
        Rows in file order.
    """
    path = Path(path)
    if not path.exists():
        return []
    with path.open(encoding=c.TEXT_ENCODING) as fh:
        return [json.loads(line) for line in fh if line.strip()]


def append_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    """Append rows to a JSONL file and flush (checkpoint semantics).

    Parameters
    ----------
    path: Path
        File path.
    rows: Iterable[Dict[str, Any]]
        Rows to append.
    """
    with Path(path).open("a", encoding=c.TEXT_ENCODING) as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        fh.flush()


def latest_by_key(rows: Iterable[Dict[str, Any]], keys: Sequence[str]) -> List[Dict[str, Any]]:
    """Deduplicate rows by key, keeping the last occurrence (resume/retry).

    Parameters
    ----------
    rows: Iterable[Dict[str, Any]]
        Rows in append order.
    keys: Sequence[str]
        Field names forming the key.

    Returns
    -------
    List[Dict[str, Any]]
        Deduplicated rows in first-seen key order.
    """
    out: Dict[tuple, Dict[str, Any]] = {}
    for row in rows:
        out[tuple(row.get(k) for k in keys)] = row
    return list(out.values())
