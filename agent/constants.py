"""String constants for the contracts agent (keys, labels, env vars, paths)."""

from pathlib import Path

# --- Paths -----------------------------------------------------------------
AGENT_DIR = Path(__file__).resolve().parent
CORPUS_PATH = AGENT_DIR / "corpus" / "passages.jsonl"
PROMPTS_DIR = AGENT_DIR / "prompts"

# --- Environment -----------------------------------------------------------
ENV_ANTHROPIC_API_KEY = "ANTHROPIC_API_KEY"
ENV_FILE_NAME = ".env"

# --- Providers -------------------------------------------------------------
PROVIDER_ANTHROPIC = "anthropic"
PROVIDER_FAKE = "fake"

# --- Corpus passage fields -------------------------------------------------
PASSAGE_DOC_ID = "doc_id"
PASSAGE_ID = "passage_id"
PASSAGE_TITLE = "title"
PASSAGE_SECTION = "section"
PASSAGE_TEXT = "text"
PASSAGE_SOURCE_FILE = "source_file"

# --- Agent config keys -----------------------------------------------------
CFG_NAME = "name"
CFG_PROVIDER = "provider"
CFG_MODEL_ID = "model_id"
CFG_TEMPERATURE = "temperature"
CFG_MAX_TOKENS = "max_tokens"
CFG_TOP_K = "top_k"
CFG_PROMPT_FILE = "prompt_file"
CFG_TIMEOUT_S = "timeout_s"

# --- Graph state keys ------------------------------------------------------
STATE_QUESTION = "question"
STATE_PASSAGES = "passages"
STATE_ANSWER = "answer"

# --- Graph node names ------------------------------------------------------
NODE_RETRIEVE = "retrieve"
NODE_ANSWER = "answer"

# --- Prompt template placeholders ------------------------------------------
PROMPT_VAR_CONTEXT = "context"
PROMPT_VAR_QUESTION = "question"
PROMPT_VAR_NOT_FOUND = "not_found"

# --- Agent behaviour markers -----------------------------------------------
NOT_FOUND_MARKER = "NOT_FOUND"
CONTEXT_PASSAGE_TEMPLATE = "[{passage_id}] {title} — {section}\n{text}"
CONTEXT_SEPARATOR = "\n\n"
TEXT_ENCODING = "utf-8"
TOKEN_PATTERN = r"[a-z0-9]+"
STEMMER_LANGUAGE = "english"

# --- CUAD ingestion --------------------------------------------------------
CUAD_ANNOTATIONS_PATH = AGENT_DIR / "corpus" / "cuad_annotations.jsonl"
CUAD_DOC_PREFIX = "CUAD"
CUAD_DOC_ID_TEMPLATE = "{prefix}-{index:04d}"
CUAD_PASSAGE_ID_TEMPLATE = "{doc_id}-{index:04d}"
CUAD_DEFAULT_SECTION = "Preamble"
CUAD_CLAUSE_SEPARATOR = "__"

# SQuAD-style keys of CUADv1.json
SQUAD_DATA = "data"
SQUAD_TITLE = "title"
SQUAD_PARAGRAPHS = "paragraphs"
SQUAD_CONTEXT = "context"
SQUAD_QAS = "qas"
SQUAD_ID = "id"
SQUAD_QUESTION = "question"
SQUAD_ANSWERS = "answers"
SQUAD_ANSWER_TEXT = "text"
SQUAD_ANSWER_START = "answer_start"
SQUAD_IS_IMPOSSIBLE = "is_impossible"

# Annotation sidecar fields
ANN_DOC_ID = "doc_id"
ANN_CLAUSE = "clause"
ANN_QUESTION = "question"
ANN_HAS_ANSWER = "has_answer"
ANN_ANSWERS = "answers"
ANN_TEXT = "text"
ANN_PASSAGE_IDS = "passage_ids"
