"""Small git helpers. Every function degrades gracefully outside a git repo."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import List, Optional

from eval import constants as c


def _git(*args: str) -> Optional[str]:
    """Run a git command at the project root.

    Parameters
    ----------
    *args: str
        git arguments.

    Returns
    -------
    Optional[str]
        stdout, or ``None`` if git fails or is unavailable.
    """
    try:
        out = subprocess.run(
            ["git", *args],
            cwd=c.ROOT_DIR,
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return out.stdout


def is_repo() -> bool:
    """Whether the project root is inside a git work tree.

    Returns
    -------
    bool
        True inside a repo.
    """
    out = _git("rev-parse", "--is-inside-work-tree")
    return bool(out and out.strip() == "true")


def head_commit() -> str:
    """Current commit SHA, with ``-dirty`` when the tree has changes.

    Returns
    -------
    str
        Short SHA, or ``unknown`` outside a repo.
    """
    sha = _git("rev-parse", "--short", "HEAD")
    if not sha:
        return c.GIT_UNKNOWN
    dirty = _git("status", "--porcelain")
    return sha.strip() + ("-dirty" if dirty and dirty.strip() else "")


def dataset_tags() -> List[str]:
    """Dataset release tags sorted by semantic version, oldest first.

    Returns
    -------
    List[str]
        Tags like ``dataset-v1.2.0``.
    """
    out = _git("tag", "--list", f"{c.DATASET_TAG_PREFIX}*")
    if not out:
        return []

    def key(tag: str):
        parts = tag[len(c.DATASET_TAG_PREFIX):].split(".")
        return tuple(int(p) if p.isdigit() else 0 for p in parts)

    return sorted((t.strip() for t in out.splitlines() if t.strip()), key=key)


def show_file(rev: str, path: Path) -> Optional[str]:
    """Content of a file at a revision.

    Parameters
    ----------
    rev: str
        Commit or tag.
    path: Path
        Absolute path inside the repo.

    Returns
    -------
    Optional[str]
        File content, or ``None`` if unavailable.
    """
    rel = Path(path).resolve().relative_to(c.ROOT_DIR).as_posix()
    return _git("show", f"{rev}:{rel}")


def commit_and_tag(paths: List[Path], message: str, tag: str) -> bool:
    """Stage paths, commit and create an annotated tag.

    Parameters
    ----------
    paths: List[Path]
        Files to stage.
    message: str
        Commit and tag message.
    tag: str
        Tag name.

    Returns
    -------
    bool
        True on success.
    """
    rels = [Path(p).resolve().relative_to(c.ROOT_DIR).as_posix() for p in paths]
    if _git("add", *rels) is None:
        return False
    if _git("commit", "-m", message) is None:
        return False
    return _git("tag", "-a", tag, "-m", message) is not None
