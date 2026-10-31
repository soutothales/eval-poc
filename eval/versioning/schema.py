"""Pydantic models for golden-dataset cases and the dataset manifest."""

from __future__ import annotations

from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

from eval import constants as c


class Category(str, Enum):
    """Question category; regressions are broken down by it."""

    LOOKUP = c.CATEGORY_LOOKUP
    MULTI_PASSAGE = c.CATEGORY_MULTI_PASSAGE
    AGGREGATE = c.CATEGORY_AGGREGATE
    UNANSWERABLE = c.CATEGORY_UNANSWERABLE
    ADVERSARIAL = c.CATEGORY_ADVERSARIAL


class Behavior(str, Enum):
    """Expected agent behaviour for a case."""

    ANSWER = c.BEHAVIOR_ANSWER
    NOT_FOUND = c.BEHAVIOR_NOT_FOUND
    DECLINE = c.BEHAVIOR_DECLINE


class Status(str, Enum):
    """Lifecycle status. Deprecated cases stay in the file for lineage."""

    ACTIVE = c.STATUS_ACTIVE
    DEPRECATED = c.STATUS_DEPRECATED


class Bump(str, Enum):
    """Semantic-version bump level."""

    NONE = c.BUMP_NONE
    PATCH = c.BUMP_PATCH
    MINOR = c.BUMP_MINOR
    MAJOR = c.BUMP_MAJOR


class SourceRef(BaseModel):
    """Reference to a passage that supports the expected answer.

    Parameters
    ----------
    doc_id: str
        Contract identifier.
    passage_id: str
        Passage identifier inside that contract.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    doc_id: str
    passage_id: str


class Case(BaseModel):
    """One golden-dataset case.

    Hashed (semantic) fields: question, expected_output, expected_sources,
    expected_behavior. Lineage and status fields are not hashed.

    Parameters
    ----------
    case_id: str
        Permanent ID, ``QA-0001`` style, never reused.
    question: str
        User question.
    expected_output: str
        Reference answer.
    expected_sources: List[SourceRef]
        Passages that support the answer; empty for not-found/decline cases.
    expected_behavior: Behavior
        ``answer``, ``not_found`` or ``decline``.
    category: Category
        Question category.
    status: Status
        ``active`` or ``deprecated``.
    added_in: str
        Dataset version that introduced the case.
    supersedes: Optional[str]
        case_id this case replaces.
    deprecated_in: Optional[str]
        Dataset version that deprecated the case.
    revision: int
        Incremented on in-place typo fixes.
    content_hash: Optional[str]
        Hash of the semantic fields; filled by ``dataset rehash``.
    """

    model_config = ConfigDict(extra="forbid", use_enum_values=True)

    case_id: str = Field(pattern=c.CASE_ID_PATTERN)
    question: str = Field(min_length=1)
    expected_output: str = Field(min_length=1)
    expected_sources: List[SourceRef] = Field(default_factory=list)
    expected_behavior: Behavior = Behavior.ANSWER
    category: Category
    status: Status = Status.ACTIVE
    added_in: str
    supersedes: Optional[str] = None
    deprecated_in: Optional[str] = None
    revision: int = Field(default=1, ge=1)
    content_hash: Optional[str] = None

    @field_validator("expected_sources")
    @classmethod
    def _no_duplicate_sources(cls, v: List[SourceRef]) -> List[SourceRef]:
        if len(set(v)) != len(v):
            raise ValueError("duplicate entries in expected_sources")
        return v

    @property
    def is_active(self) -> bool:
        """Whether the case is active.

        Returns
        -------
        bool
            True when status is ``active``.
        """
        return self.status == c.STATUS_ACTIVE


class ChangelogEntry(BaseModel):
    """One release entry in the dataset changelog.

    Parameters
    ----------
    version: str
        Released semantic version.
    date: str
        Release date, ``YYYY-MM-DD``.
    author: str
        Who released it.
    what: str
        What changed.
    why: str
        Why it changed.
    bump: Bump
        Declared bump level.
    typo_fixes: List[str]
        case_ids edited in place as typo fixes.
    """

    model_config = ConfigDict(extra="forbid", use_enum_values=True)

    version: str
    date: str
    author: str
    what: str
    why: str
    bump: Bump
    typo_fixes: List[str] = Field(default_factory=list)


class Counts(BaseModel):
    """Case counts stored in the manifest.

    Parameters
    ----------
    by_category: Dict[str, int]
        Active cases per category.
    by_status: Dict[str, int]
        Cases per status.
    """

    by_category: Dict[str, int]
    by_status: Dict[str, int]


class DatasetManifest(BaseModel):
    """Dataset release manifest (``datasets/manifest.json``).

    Parameters
    ----------
    schema_version: str
        Case schema version; a change forces a major bump.
    version: str
        Semantic version of the dataset.
    dataset_hash: str
        Hash of sorted (case_id, content_hash) pairs of active cases.
    corpus_version: str
        Fingerprint of the corpus the sources refer to.
    core_regression_hash: str
        Hash of the core regression subset.
    counts: Counts
        Case counts.
    changelog: List[ChangelogEntry]
        Release history, newest last.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: str
    version: str
    dataset_hash: str
    corpus_version: str
    core_regression_hash: str
    counts: Counts
    changelog: List[ChangelogEntry] = Field(default_factory=list)


class CoreRegression(BaseModel):
    """Frozen core regression subset for one major version.

    Parameters
    ----------
    major: int
        Major dataset version the subset belongs to.
    case_ids: List[str]
        Case IDs in the subset.
    """

    major: int
    case_ids: List[str]
