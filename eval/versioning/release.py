"""Dataset release flow: validate -> diff -> check bump -> manifest -> git tag."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

from agent.retriever import corpus_index, corpus_version
from eval import constants as c
from eval import gitutil
from eval.versioning.diff import DatasetDiff, bump_max, diff_cases, next_version
from eval.versioning.hashing import content_hash
from eval.versioning.manifest import (
    build_manifest,
    load_cases,
    load_core,
    load_manifest,
    parse_cases,
    rehash_cases,
    save_cases,
    save_manifest,
)
from eval.versioning.schema import Case, ChangelogEntry, CoreRegression, DatasetManifest
from eval.versioning.validate import Issue, validate

INITIAL_VERSION = "1.0.0"
UNRELEASED = "unreleased"


@dataclass
class PreviousRelease:
    """Snapshot of the last tagged dataset release.

    Parameters
    ----------
    tag: str
        Git tag.
    cases: List[Case]
        Cases at that tag.
    core: Optional[CoreRegression]
        Core subset at that tag.
    manifest: Optional[DatasetManifest]
        Manifest at that tag.
    """

    tag: str
    cases: List[Case]
    core: Optional[CoreRegression]
    manifest: Optional[DatasetManifest]


@dataclass
class ReleaseResult:
    """Outcome of a release attempt.

    Parameters
    ----------
    ok: bool
        Whether the release was written.
    version: Optional[str]
        New version on success.
    diff: Optional[DatasetDiff]
        Diff against the previous release.
    issues: List[Issue]
        Validation issues.
    messages: List[str]
        Human-readable notes and errors.
    """

    ok: bool
    version: Optional[str] = None
    diff: Optional[DatasetDiff] = None
    issues: List[Issue] = field(default_factory=list)
    messages: List[str] = field(default_factory=list)


def previous_release() -> Optional[PreviousRelease]:
    """Load the most recent tagged release from git.

    Returns
    -------
    Optional[PreviousRelease]
        The release, or ``None`` if there is no tag or no git repo.
    """
    tags = gitutil.dataset_tags()
    if not tags:
        return None
    tag = tags[-1]
    cases_text = gitutil.show_file(tag, c.CASES_PATH)
    if cases_text is None:
        return None
    core_text = gitutil.show_file(tag, c.CORE_REGRESSION_PATH)
    manifest_text = gitutil.show_file(tag, c.DATASET_MANIFEST_PATH)
    return PreviousRelease(
        tag=tag,
        cases=parse_cases(cases_text),
        core=CoreRegression.model_validate_json(core_text) if core_text else None,
        manifest=DatasetManifest.model_validate_json(manifest_text) if manifest_text else None,
    )


def _apply_typo_fixes(cases: List[Case], prev: Optional[PreviousRelease], typo_ids: Sequence[str]) -> List[Case]:
    """Bump ``revision`` for declared typo fixes whose content actually changed.

    Parameters
    ----------
    cases: List[Case]
        Current cases.
    prev: Optional[PreviousRelease]
        Previous release.
    typo_ids: Sequence[str]
        Declared typo-fix case IDs.

    Returns
    -------
    List[Case]
        Cases with revisions bumped where needed.
    """
    if not prev or not typo_ids:
        return cases
    prev_by = {k.case_id: k for k in prev.cases}
    out = []
    for k in cases:
        p = prev_by.get(k.case_id)
        if k.case_id in typo_ids and p and content_hash(p) != content_hash(k) and k.revision <= p.revision:
            k = k.model_copy(update={c.CASE_REVISION: p.revision + 1})
        out.append(k)
    return out


def _fill_lineage(cases: List[Case], prev: Optional[PreviousRelease], version: str) -> List[Case]:
    """Fill ``added_in`` for new cases and ``deprecated_in`` for newly deprecated ones.

    Parameters
    ----------
    cases: List[Case]
        Current cases.
    prev: Optional[PreviousRelease]
        Previous release.
    version: str
        Version being released.

    Returns
    -------
    List[Case]
        Cases with lineage fields filled.
    """
    prev_ids = {k.case_id for k in prev.cases} if prev else set()
    out = []
    for k in cases:
        update = {}
        if k.case_id not in prev_ids and k.added_in in (UNRELEASED, ""):
            update[c.CASE_ADDED_IN] = version
        if not k.is_active and not k.deprecated_in:
            update[c.CASE_DEPRECATED_IN] = version
        out.append(k.model_copy(update=update) if update else k)
    return out


def release(
    bump: str,
    what: str,
    why: str,
    author: str,
    typo_fixes: Sequence[str] = (),
    use_git: bool = True,
    today: Optional[str] = None,
) -> ReleaseResult:
    """Validate the working dataset and write a new release.

    Parameters
    ----------
    bump: str
        Declared bump: ``patch``, ``minor`` or ``major``. Must be at least the
        inferred bump. Ignored for the first release, which is ``1.0.0``.
    what: str
        Changelog: what changed.
    why: str
        Changelog: why.
    author: str
        Changelog: who.
    typo_fixes: Sequence[str]
        case_ids edited in place whose meaning is unchanged.
    use_git: bool
        Commit and tag when inside a git repo.
    today: Optional[str]
        Release date override (tests).

    Returns
    -------
    ReleaseResult
        Outcome, diff and issues.
    """
    result = ReleaseResult(ok=False)
    prev = previous_release()
    manifest = load_manifest()
    core = load_core()
    cases = load_cases()

    base_version = prev.manifest.version if prev and prev.manifest else (manifest.version if manifest else None)
    first_release = base_version is None

    cases = _apply_typo_fixes(cases, prev, typo_fixes)
    revision_typos = []
    if prev:
        prev_by = {k.case_id: k for k in prev.cases}
        revision_typos = [k.case_id for k in cases if k.case_id in prev_by and k.revision > prev_by[k.case_id].revision]

    d = diff_cases(
        prev.cases if prev else [],
        cases,
        typo_fixes=set(typo_fixes) | set(revision_typos),
        old_core=prev.core.case_ids if prev and prev.core else None,
        new_core=core.case_ids,
        schema_changed=bool(prev and prev.manifest and prev.manifest.schema_version != c.CURRENT_SCHEMA_VERSION),
    )
    result.diff = d

    if first_release:
        new_version = INITIAL_VERSION
        bump = c.BUMP_MAJOR
    else:
        if prev is None:
            result.messages.append(
                "A manifest exists but no git tag was found; the diff is computed against an empty "
                "dataset. Commit and tag releases so append-only checks work."
            )
        if bump_max(bump, d.inferred_bump) != bump:
            result.messages.append(f"declared bump '{bump}' is below inferred '{d.inferred_bump}': {d.reasons}")
            return result
        if d.inferred_bump == c.BUMP_NONE and bump == c.BUMP_NONE:
            result.messages.append("nothing to release")
            return result
        new_version = next_version(base_version, bump)

    cases = rehash_cases(_fill_lineage(cases, prev, new_version))
    issues = validate(
        cases,
        core,
        corpus_index(),
        manifest=None,
        previous=prev.cases if prev else None,
        previous_core=prev.core if prev else None,
        check_manifest=False,
    )
    result.issues = issues
    if any(i.level == c.LEVEL_ERROR for i in issues):
        result.messages.append("validation failed; nothing written")
        return result
    if int(new_version.split(".")[0]) != core.major:
        result.messages.append(
            f"new version {new_version} is major {new_version.split('.')[0]} but core_regression.json "
            f"has major {core.major}; update the core subset for the new major first"
        )
        return result

    entry = ChangelogEntry(
        version=new_version,
        date=today or dt.date.today().strftime(c.DATE_FORMAT),
        author=author,
        what=what,
        why=why,
        bump=bump,
        typo_fixes=sorted(set(typo_fixes) | set(revision_typos)),
    )
    changelog = (manifest.changelog if manifest else []) + [entry]
    new_manifest = build_manifest(cases, core, new_version, corpus_version(), changelog)
    save_cases(cases)
    save_manifest(new_manifest)
    result.ok = True
    result.version = new_version

    if use_git and gitutil.is_repo():
        tag = f"{c.DATASET_TAG_PREFIX}{new_version}"
        ok = gitutil.commit_and_tag(
            [c.CASES_PATH, c.DATASET_MANIFEST_PATH, c.CORE_REGRESSION_PATH],
            f"dataset: release {new_version} — {what}",
            tag,
        )
        result.messages.append(f"tagged {tag}" if ok else "git commit/tag failed; commit manually")
    else:
        result.messages.append("not a git repo (or --no-git): files written, no tag created")
    return result
