"""Diff two dataset versions and infer the minimum semver bump."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, Iterable, List, Optional

from eval import constants as c
from eval.versioning.hashing import content_hash
from eval.versioning.schema import Case

# Fields that are neither hashed nor lineage bookkeeping; changes are patch-level.
_LABEL_FIELDS = (c.CASE_CATEGORY,)


@dataclass
class DatasetDiff:
    """Differences between an old and a new case list.

    Parameters
    ----------
    added: List[str]
        New active case IDs.
    removed: List[str]
        IDs missing from the new list (append-only violation).
    redefined: List[str]
        Active cases whose content hash changed without a typo-fix override.
    typo_fixed: List[str]
        Active cases whose content hash changed under a typo-fix override.
    deprecated: List[str]
        Cases active before and deprecated now.
    metadata_changed: List[str]
        Cases where only label fields changed.
    reasons: List[str]
        Human-readable reasons for the inferred bump.
    """

    added: List[str] = field(default_factory=list)
    removed: List[str] = field(default_factory=list)
    redefined: List[str] = field(default_factory=list)
    typo_fixed: List[str] = field(default_factory=list)
    deprecated: List[str] = field(default_factory=list)
    metadata_changed: List[str] = field(default_factory=list)
    reasons: List[str] = field(default_factory=list)
    inferred_bump: str = c.BUMP_NONE

    def to_dict(self) -> Dict[str, object]:
        """Return the diff as a plain dict.

        Returns
        -------
        Dict[str, object]
            Field name to value.
        """
        return asdict(self)


def bump_max(a: str, b: str) -> str:
    """Return the larger of two bump levels.

    Parameters
    ----------
    a: str
        Bump level.
    b: str
        Bump level.

    Returns
    -------
    str
        The larger level by ``none < patch < minor < major``.
    """
    return a if c.BUMP_ORDER.index(a) >= c.BUMP_ORDER.index(b) else b


def next_version(version: str, bump: str) -> str:
    """Apply a bump to a semantic version.

    Parameters
    ----------
    version: str
        Current version ``X.Y.Z``.
    bump: str
        ``patch``, ``minor`` or ``major``.

    Returns
    -------
    str
        Bumped version.
    """
    major, minor, patch = (int(p) for p in version.split("."))
    if bump == c.BUMP_MAJOR:
        return f"{major + 1}.0.0"
    if bump == c.BUMP_MINOR:
        return f"{major}.{minor + 1}.0"
    if bump == c.BUMP_PATCH:
        return f"{major}.{minor}.{patch + 1}"
    return version


def diff_cases(
    old: Iterable[Case],
    new: Iterable[Case],
    typo_fixes: Optional[Iterable[str]] = None,
    old_core: Optional[List[str]] = None,
    new_core: Optional[List[str]] = None,
    schema_changed: bool = False,
) -> DatasetDiff:
    """Compare two case lists and infer the minimum required bump.

    Rules: removal, deprecation of an active case, an unflagged content change,
    a core-subset change or a schema change is major; new cases are minor;
    typo fixes and label-only changes are patch.

    Parameters
    ----------
    old: Iterable[Case]
        Cases at the previous release.
    new: Iterable[Case]
        Current cases.
    typo_fixes: Optional[Iterable[str]]
        case_ids declared as in-place typo fixes.
    old_core: Optional[List[str]]
        Core subset IDs at the previous release.
    new_core: Optional[List[str]]
        Current core subset IDs.
    schema_changed: bool
        Whether the case schema version changed.

    Returns
    -------
    DatasetDiff
        Differences and the inferred bump.
    """
    typo = set(typo_fixes or [])
    old_by = {k.case_id: k for k in old}
    new_by = {k.case_id: k for k in new}
    d = DatasetDiff()

    for cid, o in old_by.items():
        n = new_by.get(cid)
        if n is None:
            d.removed.append(cid)
            continue
        if o.is_active and not n.is_active:
            d.deprecated.append(cid)
            continue
        if not (o.is_active and n.is_active):
            continue
        if content_hash(o) != content_hash(n):
            (d.typo_fixed if cid in typo else d.redefined).append(cid)
        elif any(getattr(o, f) != getattr(n, f) for f in _LABEL_FIELDS):
            d.metadata_changed.append(cid)

    d.added = [cid for cid, n in new_by.items() if cid not in old_by and n.is_active]

    bump = c.BUMP_NONE
    if d.removed:
        bump = bump_max(bump, c.BUMP_MAJOR)
        d.reasons.append(f"removed cases: {d.removed}")
    if d.deprecated:
        bump = bump_max(bump, c.BUMP_MAJOR)
        d.reasons.append(f"active cases deprecated: {d.deprecated}")
    if d.redefined:
        bump = bump_max(bump, c.BUMP_MAJOR)
        d.reasons.append(f"cases redefined without --typo-fix: {d.redefined}")
    if schema_changed:
        bump = bump_max(bump, c.BUMP_MAJOR)
        d.reasons.append("case schema version changed")
    if old_core is not None and new_core is not None and sorted(old_core) != sorted(new_core):
        bump = bump_max(bump, c.BUMP_MAJOR)
        d.reasons.append("core regression subset changed")
    if d.added:
        bump = bump_max(bump, c.BUMP_MINOR)
        d.reasons.append(f"new cases: {d.added}")
    if d.typo_fixed or d.metadata_changed:
        bump = bump_max(bump, c.BUMP_PATCH)
        d.reasons.append(f"typo/label fixes: {d.typo_fixed + d.metadata_changed}")
    d.inferred_bump = bump
    return d
