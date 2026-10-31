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
