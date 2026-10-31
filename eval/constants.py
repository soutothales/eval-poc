"""String constants for the evaluation pipeline (keys, labels, paths, enums)."""

from pathlib import Path

# --- Paths -----------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
DATASETS_DIR = ROOT_DIR / "datasets"
CASES_PATH = DATASETS_DIR / "cases.jsonl"
DATASET_MANIFEST_PATH = DATASETS_DIR / "manifest.json"
CORE_REGRESSION_PATH = DATASETS_DIR / "core_regression.json"
CONFIGS_DIR = ROOT_DIR / "configs"
JUDGE_CONFIG_PATH = CONFIGS_DIR / "judge.yaml"
THRESHOLDS_PATH = CONFIGS_DIR / "thresholds.yaml"
PRICE_TABLE_PATH = ROOT_DIR / "pricing" / "price_table.json"
RUNS_DIR = ROOT_DIR / "runs"
REPORTS_DIR = ROOT_DIR / "reports"

# --- Run directory files ---------------------------------------------------
RUN_MANIFEST_FILE = "manifest.json"
OUTPUTS_FILE = "outputs.jsonl"
SCORES_FILE = "scores.jsonl"
SUMMARY_FILE = "summary.json"
CASES_TABLE_FILE = "cases.parquet"
COMPARE_FILE_TEMPLATE = "compare_{a}__{b}.json"
REPORT_FILE_TEMPLATE = "report_{a}__{b}.html"
LOCK_FILE = "uv.lock"
REQUIREMENTS_FILE = "requirements.txt"

# --- Git -------------------------------------------------------------------
DATASET_TAG_PREFIX = "dataset-v"
GIT_UNKNOWN = "unknown"

# --- Encoding / formats ----------------------------------------------------
TEXT_ENCODING = "utf-8"
HASH_ALGO = "sha256"
UNICODE_NORMAL_FORM = "NFC"
RUN_ID_TIME_FORMAT = "%Y%m%dT%H%M%S"
DATE_FORMAT = "%Y-%m-%d"
JSON_SEPARATORS = (",", ":")

# --- Case fields -----------------------------------------------------------
CASE_ID = "case_id"
CASE_QUESTION = "question"
CASE_EXPECTED_OUTPUT = "expected_output"
CASE_EXPECTED_SOURCES = "expected_sources"
CASE_EXPECTED_BEHAVIOR = "expected_behavior"
CASE_CATEGORY = "category"
CASE_STATUS = "status"
CASE_ADDED_IN = "added_in"
CASE_SUPERSEDES = "supersedes"
CASE_DEPRECATED_IN = "deprecated_in"
CASE_REVISION = "revision"
CASE_CONTENT_HASH = "content_hash"
CASE_ID_PATTERN = r"^QA-\d{4}$"

SOURCE_DOC_ID = "doc_id"
SOURCE_PASSAGE_ID = "passage_id"

# --- Case enums ------------------------------------------------------------
STATUS_ACTIVE = "active"
STATUS_DEPRECATED = "deprecated"

CATEGORY_LOOKUP = "lookup"
CATEGORY_MULTI_PASSAGE = "multi_passage"
CATEGORY_AGGREGATE = "aggregate"
CATEGORY_UNANSWERABLE = "unanswerable"
CATEGORY_ADVERSARIAL = "adversarial"
CATEGORIES = (
    CATEGORY_LOOKUP,
    CATEGORY_MULTI_PASSAGE,
    CATEGORY_AGGREGATE,
    CATEGORY_UNANSWERABLE,
    CATEGORY_ADVERSARIAL,
)

BEHAVIOR_ANSWER = "answer"
BEHAVIOR_NOT_FOUND = "not_found"
BEHAVIOR_DECLINE = "decline"

# --- Dataset manifest fields -----------------------------------------------
MANIFEST_SCHEMA_VERSION = "schema_version"
MANIFEST_VERSION = "version"
MANIFEST_DATASET_HASH = "dataset_hash"
MANIFEST_CORPUS_VERSION = "corpus_version"
MANIFEST_CORE_HASH = "core_regression_hash"
MANIFEST_COUNTS = "counts"
MANIFEST_BY_CATEGORY = "by_category"
MANIFEST_BY_STATUS = "by_status"
MANIFEST_CHANGELOG = "changelog"
CURRENT_SCHEMA_VERSION = "1"

CHANGELOG_VERSION = "version"
CHANGELOG_DATE = "date"
CHANGELOG_AUTHOR = "author"
CHANGELOG_WHAT = "what"
CHANGELOG_WHY = "why"
CHANGELOG_BUMP = "bump"
CHANGELOG_TYPO_FIXES = "typo_fixes"

CORE_CASE_IDS = "case_ids"
CORE_MAJOR = "major"

# --- Semver bumps ----------------------------------------------------------
BUMP_NONE = "none"
BUMP_PATCH = "patch"
BUMP_MINOR = "minor"
BUMP_MAJOR = "major"
BUMP_ORDER = (BUMP_NONE, BUMP_PATCH, BUMP_MINOR, BUMP_MAJOR)

# --- Diff keys -------------------------------------------------------------
DIFF_ADDED = "added"
DIFF_REMOVED = "removed"
DIFF_REDEFINED = "redefined"
DIFF_TYPO_FIXED = "typo_fixed"
DIFF_DEPRECATED = "deprecated"
DIFF_METADATA_CHANGED = "metadata_changed"
DIFF_INFERRED_BUMP = "inferred_bump"

# --- Validation issue fields -----------------------------------------------
ISSUE_CHECK = "check"
ISSUE_LEVEL = "level"
ISSUE_MESSAGE = "message"
LEVEL_ERROR = "error"
LEVEL_WARNING = "warning"
LEVEL_INFO = "info"

CHECK_SCHEMA = "schema"
CHECK_UNIQUE_ID = "unique_case_id"
CHECK_APPEND_ONLY = "append_only"
CHECK_DUPLICATE_CONTENT = "duplicate_content"
CHECK_LINEAGE = "lineage"
CHECK_HASHES = "hashes"
CHECK_CORE = "core_regression"
CHECK_SOURCES = "source_refs"
CHECK_MANIFEST = "manifest"

# --- Agent result / outputs row fields ------------------------------------
OUT_RUN_ID = "run_id"
OUT_CASE_ID = "case_id"
OUT_REPEAT_IDX = "repeat_idx"
OUT_CATEGORY = "category"
OUT_ANSWER = "answer"
OUT_RETRIEVAL_CONTEXT = "retrieval_context"
OUT_RETRIEVED_IDS = "retrieved_ids"
OUT_USAGE = "usage"
OUT_LATENCY_S = "latency_s"
OUT_ERROR = "error"
OUT_ATTEMPTS = "attempts"
OUT_TIMESTAMP = "timestamp"

USAGE_INPUT_TOKENS = "input_tokens"
USAGE_OUTPUT_TOKENS = "output_tokens"
USAGE_TOTAL_TOKENS = "total_tokens"
USAGE_LLM_CALLS = "llm_calls"

# --- Run manifest fields ---------------------------------------------------
RM_RUN_ID = "run_id"
RM_TIMESTAMP = "timestamp"
RM_GIT_COMMIT = "git_commit"
RM_DATASET = "dataset"
RM_DATASET_VERSION = "version"
RM_DATASET_HASH = "dataset_hash"
RM_CORE_HASH = "core_regression_hash"
RM_CORPUS_VERSION = "corpus_version"
RM_AGENT = "agent"
RM_PROMPT_HASH = "prompt_hash"
RM_JUDGE = "judge"
RM_CRITERIA_HASH = "criteria_hash"
RM_PRICE_TABLE_VERSION = "price_table_version"
RM_DEPENDENCY_HASH = "dependency_hash"
RM_RUN_OPTIONS = "run_options"
RM_REPEATS = "repeats"
RM_LIMIT = "limit"
RM_CORE_ONLY = "core_only"
RM_CONCURRENCY = "concurrency"

# --- Judge config keys -----------------------------------------------------
JCFG_PROVIDER = "provider"
JCFG_MODEL_ID = "model_id"
JCFG_TEMPERATURE = "temperature"
JCFG_MAX_TOKENS = "max_tokens"
JCFG_CONCURRENCY = "concurrency"
JCFG_THRESHOLDS = "thresholds"
JCFG_CORRECTNESS_STEPS = "correctness_evaluation_steps"
JCFG_TIMEOUT_S = "timeout_s"

# --- Metric names ----------------------------------------------------------
M_ANSWER_RELEVANCY = "answer_relevancy"
M_FAITHFULNESS = "faithfulness"
M_CONTEXTUAL_PRECISION = "contextual_precision"
M_CONTEXTUAL_RECALL = "contextual_recall"
M_CONTEXTUAL_RELEVANCY = "contextual_relevancy"
M_CORRECTNESS = "correctness"
M_HIT_RATE = "hit_rate_at_k"
M_MRR = "mrr"
M_BEHAVIOR_ACCURACY = "behavior_accuracy"
M_FALSE_REFUSAL = "false_refusal"

LLM_METRICS = (
    M_ANSWER_RELEVANCY,
    M_FAITHFULNESS,
    M_CONTEXTUAL_PRECISION,
    M_CONTEXTUAL_RECALL,
    M_CONTEXTUAL_RELEVANCY,
    M_CORRECTNESS,
)
DETERMINISTIC_METRICS = (M_HIT_RATE, M_MRR, M_BEHAVIOR_ACCURACY, M_FALSE_REFUSAL)
ALL_METRICS = LLM_METRICS + DETERMINISTIC_METRICS
# Metrics where a lower value is better.
LOWER_IS_BETTER = (M_FALSE_REFUSAL,)
GEVAL_CORRECTNESS_NAME = "Correctness"

# --- Score row fields ------------------------------------------------------
SC_RUN_ID = "run_id"
SC_CASE_ID = "case_id"
SC_REPEAT_IDX = "repeat_idx"
SC_CATEGORY = "category"
SC_METRIC = "metric"
SC_SCORE = "score"
SC_SUCCESS = "success"
SC_THRESHOLD = "threshold"
SC_REASON = "reason"
SC_ERROR = "error"
SC_JUDGE_INPUT_TOKENS = "judge_input_tokens"
SC_JUDGE_OUTPUT_TOKENS = "judge_output_tokens"

# --- Cases table (one row per case, repeat 0) ------------------------------
CT_CASE_ID = "case_id"
CT_CATEGORY = "category"
CT_LATENCY_S = "latency_s"
CT_INPUT_TOKENS = "input_tokens"
CT_OUTPUT_TOKENS = "output_tokens"
CT_AGENT_COST = "agent_cost"
CT_JUDGE_COST = "judge_cost"
CT_ERROR = "has_error"
CT_RUN_ID = "run_id"
CT_SCORE_SUFFIX = ""
CT_REASON_SUFFIX = "__reason"

# --- Summary fields --------------------------------------------------------
SUM_RUN_ID = "run_id"
SUM_N_CASES = "n_cases"
SUM_N_ERRORS = "n_errors"
SUM_ERROR_RATE = "error_rate"
SUM_QUALITY = "quality"
SUM_OPERATIONAL = "operational"
SUM_BY_CATEGORY = "by_category"
SUM_OVERALL = "overall"
SUM_MEAN = "mean"
SUM_PASS_RATE = "pass_rate"
SUM_N = "n"
SUM_P50 = "p50"
SUM_P95 = "p95"
SUM_LATENCY = "latency_s"
SUM_AGENT_COST = "agent_cost_per_question"
SUM_AGENT_COST_TOTAL = "agent_cost_total"
SUM_JUDGE_COST_TOTAL = "judge_cost_total"
SUM_COST_PER_CORRECT = "cost_per_correct_answer"
SUM_TOKENS_IN = "input_tokens_per_question"
SUM_TOKENS_OUT = "output_tokens_per_question"
SUM_CURRENCY = "currency"
SUM_UNPRICED_MODELS = "unpriced_models"

# --- Price table fields ----------------------------------------------------
PT_VERSION = "version"
PT_CURRENCY = "currency"
PT_UNIT_TOKENS = "unit_tokens"
PT_MODELS = "models"
PT_ALIASES = "aliases"
PT_INPUT = "input"
PT_OUTPUT = "output"
PT_LONG_THRESHOLD = "long_context_threshold"
PT_INPUT_LONG = "input_long"
PT_OUTPUT_LONG = "output_long"

# --- Thresholds config keys ------------------------------------------------
TH_QUALITY = "quality"
TH_MAX_DROP = "max_drop"
TH_OPERATIONAL = "operational"
TH_MAX_REL_INCREASE = "max_rel_increase"
TH_MAX_ERROR_RATE = "max_error_rate"
TH_MIN_CATEGORY_CASES = "min_category_cases"
TH_REQUIRE_SIGNIFICANCE = "require_significance"
TH_BOOTSTRAP_SAMPLES = "bootstrap_samples"
TH_CONFIDENCE = "confidence"
TH_SEED = "seed"

# --- Compare output fields -------------------------------------------------
CMP_RUN_A = "run_a"
CMP_RUN_B = "run_b"
CMP_DATASET_MATCH = "dataset_hash_match"
CMP_SHARED_CASES = "n_shared_cases"
CMP_WARNINGS = "warnings"
CMP_QUALITY = "quality"
CMP_OPERATIONAL = "operational"
CMP_CASE_DIFFS = "case_diffs"
CMP_DELTAS = "deltas"
CMP_METRIC = "metric"
CMP_SCOPE = "scope"
CMP_MEAN_A = "mean_a"
CMP_MEAN_B = "mean_b"
CMP_DELTA = "delta"
CMP_CI_LOW = "ci_low"
CMP_CI_HIGH = "ci_high"
CMP_N = "n"
CMP_VALUE_A = "value_a"
CMP_VALUE_B = "value_b"
CMP_REL_CHANGE = "rel_change"
CMP_REASON_A = "reason_a"
CMP_REASON_B = "reason_b"
CMP_QUESTION = "question"
SCOPE_OVERALL = "overall"

# --- Gate output fields ----------------------------------------------------
GATE_PASSED = "passed"
GATE_BREACHES = "breaches"
GATE_RULE = "rule"
GATE_DETAIL = "detail"

# --- Operational metric names (compare/gate) -------------------------------
OP_AGENT_COST_MEAN = "agent_cost_mean"
OP_LATENCY_P50 = "latency_p50"
OP_LATENCY_P95 = "latency_p95"
OP_COST_PER_CORRECT = "cost_per_correct_answer"
OP_ERROR_RATE = "error_rate"
OP_TOKENS_IN = "input_tokens_mean"
OP_TOKENS_OUT = "output_tokens_mean"

# --- CLI -------------------------------------------------------------------
CMD_DATASET = "dataset"
CMD_VALIDATE = "validate"
CMD_REHASH = "rehash"
CMD_DIFF = "diff"
CMD_RELEASE = "release"
CMD_RUN = "run"
CMD_SCORE = "score"
CMD_COMPARE = "compare"
CMD_GATE = "gate"
CMD_REPORT = "report"
CMD_HISTORY = "history"
CMD_JUDGE_NOISE = "judge-noise"
CMD_ASK = "ask"
