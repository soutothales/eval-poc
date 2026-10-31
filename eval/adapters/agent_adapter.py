"""Agent adapter: one interface returning a fixed result shape per question."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from langchain_core.callbacks import UsageMetadataCallbackHandler

from agent import constants as ac
from agent.graph import AgentConfig, build_graph
from eval import constants as c


@dataclass
class AgentResult:
    """Fixed result shape consumed by the scoring phase.

    Parameters
    ----------
    answer: str
        Final answer text ("" on error).
    retrieval_context: List[str]
        Retrieved chunks in rank order, as shown to the model.
    retrieved_ids: List[str]
        Passage IDs in rank order (for hit rate and MRR).
    usage: Dict[str, Dict[str, int]]
        Token usage keyed by model name: input, output, total tokens, calls.
    latency_s: float
        End-to-end wall-clock latency.
    error: Optional[str]
        Error type and message, or ``None``.
    """

    answer: str = ""
    retrieval_context: List[str] = field(default_factory=list)
    retrieved_ids: List[str] = field(default_factory=list)
    usage: Dict[str, Dict[str, int]] = field(default_factory=dict)
    latency_s: float = 0.0
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Return the result as a plain dict.

        Returns
        -------
        Dict[str, Any]
            Field name to value.
        """
        return asdict(self)


def _normalize_usage(raw: Dict[str, Any]) -> Dict[str, Dict[str, int]]:
    """Convert LangChain ``usage_metadata`` into the stored usage shape.

    Parameters
    ----------
    raw: Dict[str, Any]
        ``UsageMetadataCallbackHandler.usage_metadata`` (model -> usage).

    Returns
    -------
    Dict[str, Dict[str, int]]
        model -> {input_tokens, output_tokens, total_tokens}.
    """
    out: Dict[str, Dict[str, int]] = {}
    for model, u in raw.items():
        out[model] = {
            c.USAGE_INPUT_TOKENS: int(u.get(c.USAGE_INPUT_TOKENS, 0)),
            c.USAGE_OUTPUT_TOKENS: int(u.get(c.USAGE_OUTPUT_TOKENS, 0)),
            c.USAGE_TOTAL_TOKENS: int(u.get(c.USAGE_TOTAL_TOKENS, 0)),
        }
    return out


class AgentAdapter:
    """Wraps the compiled graph behind ``ainvoke(question) -> AgentResult``.

    Parameters
    ----------
    config: AgentConfig
        Pinned agent configuration.
    """

    def __init__(self, config: AgentConfig) -> None:
        self.config = config
        self._graph = build_graph(config)

    async def ainvoke(self, question: str) -> AgentResult:
        """Run one question; exceptions propagate so the runner can retry.

        A fresh usage callback per invocation keeps concurrent questions from
        mixing token counts.

        Parameters
        ----------
        question: str
            User question.

        Returns
        -------
        AgentResult
            Answer, retrieved chunks, usage and latency.
        """
        handler = UsageMetadataCallbackHandler()
        start = time.perf_counter()
        state = await self._graph.ainvoke(
            {ac.STATE_QUESTION: question}, config={"callbacks": [handler]}
        )
        latency = time.perf_counter() - start
        passages = state.get(ac.STATE_PASSAGES, [])
        usage = _normalize_usage(handler.usage_metadata)
        if not usage and self.config.provider == ac.PROVIDER_FAKE:
            usage = {self.config.model_id: {c.USAGE_INPUT_TOKENS: 0, c.USAGE_OUTPUT_TOKENS: 0, c.USAGE_TOTAL_TOKENS: 0}}
        return AgentResult(
            answer=state.get(ac.STATE_ANSWER, ""),
            retrieval_context=[p.render() for p in passages],
            retrieved_ids=[p.passage_id for p in passages],
            usage=usage,
            latency_s=round(latency, 4),
        )
