"""Hashing, diff/bump inference and validator checks."""

from agent.retriever import corpus_index, load_passages
from eval import constants as c
from eval.versioning.diff import diff_cases, next_version
from eval.versioning.hashing import content_hash, dataset_hash
from eval.versioning.manifest import load_cases, load_core, rehash_cases
from eval.versioning.schema import Case, CoreRegression, SourceRef
from eval.versioning.validate import validate


_P0 = load_passages()[0]  # any real passage of the current corpus


def make(cid="QA-0001", q="What is the fee?", out="USD 10", sources=None, **kw):
    srcs = sources if sources is not None else [SourceRef(doc_id=_P0.doc_id, passage_id=_P0.passage_id)]
    return Case(case_id=cid, question=q, expected_output=out, expected_sources=srcs,
                category=kw.pop("category", c.CATEGORY_LOOKUP), added_in="1.0.0", **kw)


def test_hash_invariant_to_whitespace_and_source_order():
    s1 = [SourceRef(doc_id="A", passage_id="A-1"), SourceRef(doc_id="B", passage_id="B-1")]
    a = make(q="What  is the\nfee? ", sources=s1)
    b = make(q="What is the fee?", sources=list(reversed(s1)))
    assert content_hash(a) == content_hash(b)


def test_hash_changes_on_semantic_fields_only():
    base = make()
    assert content_hash(base) != content_hash(make(out="USD 11"))
    assert content_hash(base) != content_hash(make(expected_behavior=c.BEHAVIOR_NOT_FOUND))
    assert content_hash(base) == content_hash(make(category=c.CATEGORY_AGGREGATE))
    assert content_hash(base) == content_hash(make(status=c.STATUS_DEPRECATED, deprecated_in="1.1.0"))


def test_dataset_hash_ignores_deprecated_and_detects_renumbering():
    a, b = make("QA-0001"), make("QA-0002", q="Other?")
    dep = make("QA-0003", q="Old?", status=c.STATUS_DEPRECATED, deprecated_in="1.1.0")
    assert dataset_hash([a, b]) == dataset_hash([a, b, dep])
    assert dataset_hash([a]) != dataset_hash([make("QA-0009")])


def test_bump_inference():
    old = [make("QA-0001"), make("QA-0002", q="Two?")]
    assert diff_cases(old, old).inferred_bump == c.BUMP_NONE
    assert diff_cases(old, old + [make("QA-0003", q="Three?")]).inferred_bump == c.BUMP_MINOR
    edited = [make("QA-0001", out="USD 99"), old[1]]
    assert diff_cases(old, edited).inferred_bump == c.BUMP_MAJOR
    assert diff_cases(old, edited, typo_fixes=["QA-0001"]).inferred_bump == c.BUMP_PATCH
    relabeled = [make("QA-0001", category=c.CATEGORY_AGGREGATE), old[1]]
    assert diff_cases(old, relabeled).inferred_bump == c.BUMP_PATCH
    assert diff_cases(old, old[:1]).inferred_bump == c.BUMP_MAJOR
    dep = [old[0], old[1].model_copy(update={"status": c.STATUS_DEPRECATED, "deprecated_in": "2.0.0"})]
    assert diff_cases(old, dep).inferred_bump == c.BUMP_MAJOR
    assert diff_cases(old, old, old_core=["QA-0001"], new_core=["QA-0002"]).inferred_bump == c.BUMP_MAJOR


def test_next_version():
    assert next_version("1.2.3", c.BUMP_PATCH) == "1.2.4"
    assert next_version("1.2.3", c.BUMP_MINOR) == "1.3.0"
    assert next_version("1.2.3", c.BUMP_MAJOR) == "2.0.0"


def _errors(issues):
    return [i for i in issues if i.level == c.LEVEL_ERROR]


def test_repo_dataset_is_valid():
    issues = validate(load_cases(), load_core(), corpus_index(), check_manifest=False)
    assert _errors(issues) == []


def test_validator_catches_problems():
    core = CoreRegression(major=1, case_ids=["QA-0001"])
    idx = corpus_index()
    a = make("QA-0001")
    dup = make("QA-0002")  # same content as QA-0001
    bad_src = make("QA-0003", q="X?", sources=[SourceRef(doc_id=_P0.doc_id, passage_id=_P0.doc_id + "-9999")])
    orphan = make("QA-0004", q="Y?", supersedes="QA-0001")  # supersedes an active case
    cases = rehash_cases([a, dup, bad_src, orphan])
    checks = {i.check for i in _errors(validate(cases, core, idx, check_manifest=False))}
    assert {c.CHECK_DUPLICATE_CONTENT, c.CHECK_SOURCES, c.CHECK_LINEAGE} <= checks


def test_validator_append_only_and_in_place_edit():
    core = CoreRegression(major=1, case_ids=[])
    prev = rehash_cases([make("QA-0001"), make("QA-0002", q="Two?")])
    edited = rehash_cases([make("QA-0001", out="USD 99")])
    msgs = [i.message for i in _errors(validate(edited, core, corpus_index(), previous=prev, check_manifest=False))]
    assert any("QA-0002 was removed" in m for m in msgs)
    assert any("QA-0001 changed in place" in m for m in msgs)
    typo = rehash_cases([make("QA-0001", out="USD 99", revision=2), prev[1]])
    assert _errors(validate(typo, core, corpus_index(), previous=prev, check_manifest=False)) == []
