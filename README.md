# eval-poc — continuous evaluation and drift detection for a RAG agent

A local proof of concept of an LLMOps evaluation pipeline: a versioned golden
dataset, a two-phase runner (generate → score), a run fingerprint, paired
comparisons with bootstrap confidence intervals, a CI-ready regression gate and
an HTML report. The agent under test is a small LangGraph contracts assistant
running on Claude; the judge is Claude too, through DeepEval.

Everything runs on your machine. The only external dependency is the Claude API
(your key). There is also a fully offline mode for trying the pipeline at zero cost.

```
datasets/ (versioned) ──► run (phase 1) ──► agent under test (pinned config)
                              │                 └─► answer, retrieved passages, tokens, latency
                              ▼
                     runs/<id>/outputs.jsonl ──► score (phase 2: DeepEval + deterministic)
                                                      └─► scores.jsonl, summary.json, cases.parquet
                                                              │
                              compare / gate / report / history (DuckDB) ◄┘
```

The corpus is **CUAD v1** (Contract Understanding Atticus Dataset): 510 real
commercial contracts from public SEC filings, split into 27,564 passages, with
CUAD's lawyer labels mapped to passage IDs (CC BY 4.0, see
`agent/corpus/NOTICE.md`). Rebuild it with
`python -m agent.ingest_cuad path/to/CUADv1.json`. Dataset v1.0.0 used a small
fictional corpus; its 30 cases are kept in the file as deprecated lineage.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env            # then put your key in ANTHROPIC_API_KEY
make test                       # 17 tests, no API calls

python -m eval.cli ask "What is the liability cap under the Acme MSA in dollars?"
```

Try the whole pipeline with no key and no cost:

```bash
python -m eval.cli run --config configs/offline.yaml            # prints RUN_A
python -m eval.cli score RUN_A --deterministic-only
python -m eval.cli run --config configs/offline_low_topk.yaml   # prints RUN_B
python -m eval.cli score RUN_B --deterministic-only
python -m eval.cli gate RUN_A RUN_B                              # fails: retrieval regressed
python -m eval.cli report RUN_A RUN_B                            # reports/report_*.html
```

## The demo for the article: four runs

| # | Command | What it shows |
|---|---|---|
| 1 | `run --config configs/baseline.yaml --repeats 3` then `score` | Baseline (Haiku). Core cases run 3× to measure agent variance. |
| 2 | Same config again, then `compare RUN1 RUN2` and `judge-noise RUN1` | **Noise floor.** Deltas between two identical configs, plus judge variance on re-scoring the same outputs. Thresholds go above this. |
| 3 | `run --config configs/low_topk.yaml`, `score`, `gate RUN1 RUN3` | **Induced regression.** top_k=1 starves retrieval; the gate fails, and the per-category view shows multi-passage and aggregate collapsing. |
| 4 | `run --config configs/model_b.yaml`, `score`, `report RUN1 RUN4` | **Model decision.** Sonnet vs Haiku on the same index and prompt: quality, latency and cost per category, plus cost per correct answer. |

`python -m eval.cli history` prints the trend across all runs (DuckDB over
`runs/*/cases.parquet`, the local stand-in for S3 + Athena).

A useful decision rule to state in the write-up: *switch models only if
correctness is within 1 point (CI overlapping zero) and cost per correct answer
drops by at least 15%*, or the inverse for a quality upgrade.

## Commands

| Command | Purpose |
|---|---|
| `dataset validate [--allow-unreleased]` | All dataset checks (run this in CI) |
| `dataset rehash` | Recompute `content_hash` after editing cases |
| `dataset diff [--typo-fix IDS]` | Diff vs last release tag, inferred semver bump |
| `dataset release --bump … --what … --why …` | Validate, bump, write manifest, commit and tag `dataset-vX.Y.Z` |
| `run --config C [--repeats N] [--limit N] [--core-only] [--resume ID] [--retry-errors]` | Phase 1 |
| `score ID [--deterministic-only] [--rescore]` | Phase 2 |
| `compare A B [--categories]` | Paired deltas with 95% bootstrap CI, cost and latency |
| `gate A B` | Exit 1 if B regresses beyond `configs/thresholds.yaml` |
| `report A B` | Static HTML report with case-level diffs and judge reasons |
| `judge-noise ID [--times 3] [--sample 8]` | Judge variance (re-scoring) and agent variance (repeats) |
| `history` | Trend table across runs |
| `ask "question" [--config C]` | One-off question to the agent |

The `Makefile` wraps these (`make run CONFIG=…`, `make gate A=… B=…`).

## Golden dataset versioning

`datasets/cases.jsonl` holds 100 active cases (v2.0.0) in five categories:
40 lookup, 20 multi-passage, 10 aggregate (two contracts compared), 20
unanswerable (CUAD labels the clause as absent) and 10 adversarial (false
premises about governing law, and prompt injections attached to real
questions). They are generated from CUAD's labels by
`python -m scripts.build_cuad_golden` (templates in
`scripts/cuad_templates.yaml`), so expected sources are exact and expected
outputs quote the labelled contract language. `core_regression.json` is a
frozen 12-case subset for the current major version. `manifest.json` records the
semantic version, dataset hash, corpus version, core-subset hash, counts and
changelog. Releases are git tags.

- **Identity:** `case_id` is permanent. `content_hash` covers only the semantic
  fields (question, expected output, expected sources, expected behaviour), so
  relabeling or deprecating a case does not change it.
- **Dataset hash:** hash of sorted `(case_id, content_hash)` pairs of active
  cases. Runs record it; `compare` warns and intersects case IDs when it differs.
- **Corpus pinning:** `corpus_version` is a hash of every passage. If the corpus
  changes, `validate` fails until expected sources are re-checked and a new
  dataset version is released.
- **Semver:** new cases → minor; typo/label fixes → patch; removal, deprecation,
  redefinition, core-subset or schema change → major.

Everyday workflows:

```bash
# Add a case: append a line with "added_in": "unreleased" and no content_hash
python -m eval.cli dataset rehash
python -m eval.cli dataset release --bump minor --what "add QA-0031" --why "cover penalty clauses"

# Fix a typo without changing meaning
python -m eval.cli dataset release --bump patch --typo-fix QA-0007 --what "typo" --why "spelling"

# Change meaning: never edit in place. Set the old case to "status": "deprecated",
# add a new case with "supersedes": "QA-00XX", then release a major version
# (and refresh core_regression.json with "major": 2 if a core case is affected).
```

Editing a released case in place without `--typo-fix` fails validation.

## What gets measured

| Metric | Type | Applies to |
|---|---|---|
| answer_relevancy, faithfulness, contextual_precision / recall / relevancy | DeepEval, Claude judge | Answerable cases |
| correctness | DeepEval GEval with explicit evaluation steps | All cases, including not-found and decline |
| hit_rate_at_k, mrr | Deterministic, vs labelled passages | Cases with expected sources |
| behavior_accuracy | Deterministic: answered vs `NOT_FOUND` as expected | Answer and not-found cases |
| false_refusal | Deterministic, lower is better | Answerable cases |
| tokens, latency, agent cost, judge cost, cost per correct answer | Operational | All runs |

Agent errors are recorded as rows, never dropped: they score 0 on applicable
metrics and count toward `error_rate`, which the gate checks. Judge failures are
reported separately (`n_judge_errors`) and excluded from means.

Cost is computed at scoring time from token counts and the versioned
`pricing/price_table.json` (base Anthropic list prices, no caching or batch
discounts). The run manifest stores the price-table version, so old runs can be
re-costed.

## Run fingerprint

Every `runs/<id>/manifest.json` records the git commit, dataset version and
hash, core-subset hash, agent config (model, top_k, prompt hash, corpus
version), judge config (model, thresholds, criteria hash), price-table version,
dependency hash and run options. `compare` warns when judge criteria, corpus or
price table differ between the two runs.

## Configuration

- `configs/*.yaml`: agent configs. `temperature: null` omits the parameter;
  newer Claude models may reject sampling parameters, so only set it if your
  model accepts it.
- `configs/judge.yaml`: judge model (Opus by default, different from the Haiku
  and Sonnet agents), per-metric thresholds and the GEval evaluation steps.
  Changing any of these changes the criteria hash.
- `configs/thresholds.yaml`: gate rules. A quality breach requires the drop to
  exceed `max_drop` **and** the bootstrap CI to exclude zero.

## Cost

The agent runs are cheap: Haiku at roughly 1.5k input tokens per question is a
fraction of a cent per run. The judge dominates: DeepEval makes several calls
per metric, and an answerable case gets six LLM metrics. With an Opus judge, a
full scored run costs on the order of a few dollars, and scoring repeats
multiplies that. While iterating, use `--core-only`, `--limit`, `--repeats 1`,
or switch the judge to Sonnet. `summary.json` reports the exact judge cost.

## Honest caveats (worth stating in the article)

- **Same provider for agent and judge.** The judge is a different, larger
  model, but the same family. A production setup should calibrate the judge
  against about 30–50 human-labelled cases, ideally with a judge from another
  family.
- **No temperature control** on current models: judge determinism comes from
  explicit evaluation steps and unambiguous rubrics, so measure judge noise
  before trusting small deltas.
- **100 cases** is enough to show the concept, not to detect small
  regressions. Per-category CIs are wide; that is visible in the report.
- **Lexical retrieval** is deliberate: no embedding download, fully
  deterministic. It is two-stage BM25 with stemming (pick the contract named in
  the question, then rank its passages). Offline it reaches about 0.62 passage
  hit rate@5 overall and 0.35 on aggregate questions, which makes the
  per-category story clearer.
- **Generated expected outputs** quote CUAD's labelled spans verbatim. They are
  exact but long; the correctness judge compares facts, not wording.

## From POC to production

| Concern | This POC | Production on AWS |
|---|---|---|
| Dataset | git + manifest + tags | same, plus DVC for the corpus snapshot |
| Agent | in-process LangGraph | in-process for eval + smoke test against AgentCore |
| Results store | Parquet + DuckDB | S3 Parquet + Athena |
| Triggers | CLI / `make` | CI on change + EventBridge schedule |
| Alerts | gate exit code | SNS / Slack with a link to the report |
| Online eval | — | sampled production traces, reference-free metrics |

## Layout

```
agent/        constants, llm (Claude factory + offline fake), retriever (two-stage BM25), graph, ingest_cuad, corpus/, prompts/
eval/
  constants.py
  versioning/  schema, hashing, manifest, diff, validate, release
  adapters/    AgentResult adapter (per-invocation usage callback)
  generate.py  phase 1: asyncio + semaphore, backoff, timeout, checkpoint/resume
  judge.py     DeepEval judge wrapper on Claude with token accounting
  score.py     phase 2
  deterministic.py, pricing.py, summary.py, compare.py, gate.py, report.py, noise.py, store.py
  run_manifest.py, gitutil.py, io.py, cli.py
scripts/      build_cuad_golden.py, cuad_templates.yaml
configs/  datasets/  pricing/  runs/  reports/  tests/
```
