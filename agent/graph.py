"""Minimal LangGraph contracts agent: retrieve -> answer."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict

import yaml
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from agent import constants as c
from agent.llm import build_chat_model
from agent.retriever import BM25Retriever, Passage, default_retriever

HUMAN_TEMPLATE = "Contract passages:\n\n{context}\n\nQuestion: {question}"


@dataclass(frozen=True)
class AgentConfig:
    """Pinned configuration of the agent under test.

    Parameters
    ----------
    name: str
        Short config name used in run IDs, e.g. ``baseline``.
    provider: str
        ``anthropic`` or ``fake``.
    model_id: str
        Model ID passed to the provider.
    temperature: Optional[float]
        Sampling temperature, or ``None`` for the model default.
    max_tokens: int
        Maximum output tokens.
    top_k: int
        Number of passages to retrieve.
    prompt_file: str
        File name under ``agent/prompts``.
    timeout_s: float
        Per-call HTTP timeout.
    """

    name: str
    provider: str
    model_id: str
    temperature: Optional[float]
    max_tokens: int
    top_k: int
    prompt_file: str
    timeout_s: float = 60.0

    @classmethod
    def from_yaml(cls, path: Path) -> "AgentConfig":
        """Load a config from YAML.

        Parameters
        ----------
        path: Path
            YAML file path.

        Returns
        -------
        AgentConfig
            Parsed config.
        """
        raw = yaml.safe_load(Path(path).read_text(encoding=c.TEXT_ENCODING))
        return cls(
            name=raw[c.CFG_NAME],
            provider=raw[c.CFG_PROVIDER],
            model_id=raw[c.CFG_MODEL_ID],
            temperature=raw.get(c.CFG_TEMPERATURE),
            max_tokens=int(raw[c.CFG_MAX_TOKENS]),
            top_k=int(raw[c.CFG_TOP_K]),
            prompt_file=raw[c.CFG_PROMPT_FILE],
            timeout_s=float(raw.get(c.CFG_TIMEOUT_S, 60.0)),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Return the config as a plain dict.

        Returns
        -------
        Dict[str, Any]
            Field name to value.
        """
        return asdict(self)


def load_system_prompt(prompt_file: str) -> str:
    """Read and fill the system prompt template.

    Parameters
    ----------
    prompt_file: str
        File name under ``agent/prompts``.

    Returns
    -------
    str
        System prompt with the not-found marker substituted.
    """
    template = (c.PROMPTS_DIR / prompt_file).read_text(encoding=c.TEXT_ENCODING)
    return template.format(**{c.PROMPT_VAR_NOT_FOUND: c.NOT_FOUND_MARKER})


def prompt_hash(prompt_file: str) -> str:
    """Hash the full prompt (system template plus human template).

    Parameters
    ----------
    prompt_file: str
        File name under ``agent/prompts``.

    Returns
    -------
    str
        ``sha256:<first 16 hex chars>``.
    """
    payload = load_system_prompt(prompt_file) + HUMAN_TEMPLATE
    return "sha256:" + hashlib.sha256(payload.encode(c.TEXT_ENCODING)).hexdigest()[:16]


class AgentState(TypedDict, total=False):
    """Graph state.

    Parameters
    ----------
    question: str
        User question.
    passages: List[Passage]
        Retrieved passages in rank order.
    answer: str
        Final answer.
    """

    question: str
    passages: List[Passage]
    answer: str


def build_graph(config: AgentConfig, retriever: Optional[BM25Retriever] = None):
    """Compile the retrieve -> answer graph for a config.

    Parameters
    ----------
    config: AgentConfig
        Pinned agent configuration.
    retriever: Optional[BM25Retriever]
        Retriever to use; defaults to the cached corpus retriever.

    Returns
    -------
    CompiledStateGraph
        Graph accepting ``{"question": str}``; final state carries
        ``passages`` and ``answer``.
    """
    retriever = retriever or default_retriever()
    llm = build_chat_model(
        provider=config.provider,
        model_id=config.model_id,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        timeout_s=config.timeout_s,
    )
    system_prompt = load_system_prompt(config.prompt_file)

    def retrieve(state: AgentState) -> Dict[str, Any]:
        return {c.STATE_PASSAGES: retriever.retrieve(state[c.STATE_QUESTION], config.top_k)}

    async def answer(state: AgentState) -> Dict[str, Any]:
        context = c.CONTEXT_SEPARATOR.join(p.render() for p in state[c.STATE_PASSAGES])
        human = HUMAN_TEMPLATE.format(
            **{c.PROMPT_VAR_CONTEXT: context or "(none)", c.PROMPT_VAR_QUESTION: state[c.STATE_QUESTION]}
        )
        response = await llm.ainvoke([SystemMessage(system_prompt), HumanMessage(human)])
        return {c.STATE_ANSWER: _text_of(response.content)}

    graph = StateGraph(AgentState)
    graph.add_node(c.NODE_RETRIEVE, retrieve)
    graph.add_node(c.NODE_ANSWER, answer)
    graph.add_edge(START, c.NODE_RETRIEVE)
    graph.add_edge(c.NODE_RETRIEVE, c.NODE_ANSWER)
    graph.add_edge(c.NODE_ANSWER, END)
    return graph.compile()


def _text_of(content: Any) -> str:
    """Flatten a message content (str or content blocks) into text.

    Parameters
    ----------
    content: Any
        ``AIMessage.content``.

    Returns
    -------
    str
        Concatenated text.
    """
    if isinstance(content, str):
        return content.strip()
    parts = [b.get("text", "") for b in content if isinstance(b, dict)]
    return "".join(parts).strip()
