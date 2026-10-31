"""Dataset validator: the checks CI runs on every change to the golden set."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional

from pydantic import ValidationError

from eval import constants as c
from eval.versioning.hashing import content_hash, core_hash, dataset_hash
from eval.versioning.schema import Case, CoreRegression, DatasetManifest


@dataclass(frozen=True)
class Issue:
    """One validation finding.

    Parameters
    ----------
    check: str
        Check name.
    level: str
        ``error``, ``warning`` or ``info``.
    message: str
        Description.
    """

    check: str
    level: str
    message: str

    def to_dict(self) -> Dict[str, str]:
        """Return the issue as a dict.

        Returns
        -------
        Dict[str, str]
            Keys ``check``, ``level``, ``message``.
        """
        return {c.ISSUE_CHECK: self.check, c.ISSUE_LEVEL: self.level, c.ISSUE_MESSAGE: self.message}


def parse_with_issues(text: str) -> tuple[List[Case], List[Issue]]:
    """Parse JSONL cases, reporting schema errors per line instead of raising.

    Parameters
    ----------
    text: str
        JSONL content.

    Returns
    -------
    tuple[List[Case], List[Issue]]
        Valid cases and schema issues.
    """
    cases: List[Case] = []
    issues: List[Issue] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            cases.append(Case.model_validate_json(line))
        except ValidationError as exc:
            first = exc.errors()[0]
            loc = ".".join(str(p) for p in first["loc"])
            issues.append(Issue(c.CHECK_SCHEMA, c.LEVEL_ERROR, f"line {lineno}: {loc}: {first['msg']}"))
    return cases, issues


def validate(
    cases: List[Case],
    core: CoreRegression,
    corpus_index: Dict[str, List[str]],
    manifest: Optional[DatasetManifest] = None,
    previous: Optional[List[Case]] = None,
    previous_core: Optional[CoreRegression] = None,
    allow_unreleased: bool = False,
    corpus_version: Optional[str] = None,
) -> List[Issue]:
    """Run all dataset checks.

    Parameters
    ----------
    cases: List[Case]
        Current cases (already schema-valid).
    core: CoreRegression
        Current core regression subset.
    corpus_index: Dict[str, List[str]]
        ``doc_id -> passage_ids`` of the pinned corpus.
    manifest: Optional[DatasetManifest]
        Current manifest, if any.
    previous: Optional[List[Case]]
        Cases at the last release tag, if available.
    previous_core: Optional[CoreRegression]
        Core subset at the last release tag, if available.
    allow_unreleased: bool
        Downgrade "dataset changed since manifest" to a warning (local dev).
    corpus_version: Optional[str]
        Current corpus fingerprint, compared with the manifest's.

    Returns
    -------
    List[Issue]
        All findings; any ``error`` should fail CI.
    """
    issues: List[Issue] = []
    by_id: Dict[str, Case] = {}

    # 2. unique case_id
    for cid, n in Counter(k.case_id for k in cases).items():
        if n > 1:
            issues.append(Issue(c.CHECK_UNIQUE_ID, c.LEVEL_ERROR, f"{cid} appears {n} times"))
    for k in cases:
        by_id.setdefault(k.case_id, k)

    # 5a. stored content hashes match
    for k in cases:
        computed = content_hash(k)
        if k.content_hash is None:
            issues.append(Issue(c.CHECK_HASHES, c.LEVEL_ERROR, f"{k.case_id}: content_hash missing (run `dataset rehash`)"))
        elif k.content_hash != computed:
            issues.append(Issue(c.CHECK_HASHES, c.LEVEL_ERROR, f"{k.case_id}: stored content_hash is stale (run `dataset rehash`)"))

    # 3. duplicate content among active cases
    groups: Dict[str, List[str]] = defaultdict(list)
    for k in cases:
        if k.is_active:
            groups[content_hash(k)].append(k.case_id)
    for ids in groups.values():
        if len(ids) > 1:
            issues.append(Issue(c.CHECK_DUPLICATE_CONTENT, c.LEVEL_ERROR, f"identical active cases: {ids}"))

    # 4. lineage: supersedes points to an existing, deprecated case; no cycles
    for k in cases:
        if k.status == c.STATUS_DEPRECATED and not k.deprecated_in:
            issues.append(Issue(c.CHECK_LINEAGE, c.LEVEL_ERROR, f"{k.case_id}: deprecated without deprecated_in"))
        if k.supersedes is None:
            continue
        target = by_id.get(k.supersedes)
        if target is None:
            issues.append(Issue(c.CHECK_LINEAGE, c.LEVEL_ERROR, f"{k.case_id}: supersedes unknown {k.supersedes}"))
        elif target.is_active:
            issues.append(Issue(c.CHECK_LINEAGE, c.LEVEL_ERROR, f"{k.case_id}: supersedes {k.supersedes}, which is still active"))
    for k in cases:
        seen, cur = set(), k
        while cur.supersedes and cur.supersedes in by_id:
            if cur.case_id in seen:
                issues.append(Issue(c.CHECK_LINEAGE, c.LEVEL_ERROR, f"supersedes cycle involving {k.case_id}"))
                break
            seen.add(cur.case_id)
            cur = by_id[cur.supersedes]

    # Behaviour/sources consistency
    for k in cases:
        if k.expected_behavior == c.BEHAVIOR_ANSWER and not k.expected_sources:
            issues.append(Issue(c.CHECK_SOURCES, c.LEVEL_ERROR, f"{k.case_id}: answer case without expected_sources"))

    # 8. source references exist in the pinned corpus
    for k in cases:
        for s in k.expected_sources:
            if s.passage_id not in corpus_index.get(s.doc_id, []):
                issues.append(Issue(c.CHECK_SOURCES, c.LEVEL_ERROR, f"{k.case_id}: {s.doc_id}/{s.passage_id} not in corpus"))

    # 7. core subset
    for cid in core.case_ids:
        if cid not in by_id:
            issues.append(Issue(c.CHECK_CORE, c.LEVEL_ERROR, f"core case {cid} does not exist"))
        elif not by_id[cid].is_active:
            issues.append(Issue(c.CHECK_CORE, c.LEVEL_ERROR, f"core case {cid} is deprecated"))

    # 2b. append-only and in-place edits vs the last release
    if previous is not None:
        for p in previous:
            cur = by_id.get(p.case_id)
            if cur is None:
                issues.append(Issue(c.CHECK_APPEND_ONLY, c.LEVEL_ERROR, f"{p.case_id} was removed; deprecate it instead"))
            elif p.is_active and cur.is_active and content_hash(p) != content_hash(cur):
                if cur.revision > p.revision:
                    issues.append(Issue(c.CHECK_APPEND_ONLY, c.LEVEL_INFO, f"{p.case_id}: typo fix (revision {p.revision}->{cur.revision})"))
                else:
                    issues.append(Issue(
                        c.CHECK_APPEND_ONLY, c.LEVEL_ERROR,
                        f"{p.case_id} changed in place; deprecate it and add a superseding case, "
                        "or release with --typo-fix if meaning is unchanged",
                    ))
        if manifest and previous_core and core.major == previous_core.major:
            prev_ids = {k.case_id: k for k in previous}
            for cid in previous_core.case_ids:
                if cid in prev_ids and cid in by_id and content_hash(prev_ids[cid]) != content_hash(by_id[cid]):
                    issues.append(Issue(c.CHECK_CORE, c.LEVEL_ERROR, f"core case {cid} changed within major {core.major}"))
    else:
        issues.append(Issue(c.CHECK_APPEND_ONLY, c.LEVEL_INFO, "no previous release found; append-only check skipped"))

    # 5b. manifest is in sync with the cases
    if manifest is None:
        issues.append(Issue(c.CHECK_MANIFEST, c.LEVEL_WARNING, "no manifest.json yet; run `dataset release`"))
    else:
        level = c.LEVEL_WARNING if allow_unreleased else c.LEVEL_ERROR
        if manifest.dataset_hash != dataset_hash(cases):
            issues.append(Issue(c.CHECK_MANIFEST, level, "cases changed since the manifest was built; run `dataset release`"))
        if manifest.core_regression_hash != core_hash(core.case_ids, cases):
            issues.append(Issue(c.CHECK_MANIFEST, level, "core subset changed since the manifest was built"))
        if corpus_version is not None and manifest.corpus_version != corpus_version:
            issues.append(Issue(
                c.CHECK_MANIFEST, level,
                f"corpus changed ({manifest.corpus_version} -> {corpus_version}); "
                "re-check expected sources and release a new dataset version",
            ))
        if int(manifest.version.split(".")[0]) != core.major:
            issues.append(Issue(c.CHECK_CORE, c.LEVEL_ERROR, f"core subset major {core.major} != dataset major {manifest.version}"))
    return issues
