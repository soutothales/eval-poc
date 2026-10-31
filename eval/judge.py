"""Claude judge for DeepEval, with its own token accounting."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Type

import yaml
from deepeval.models import DeepEvalBaseLLM
from pydantic import BaseModel

from agent.llm import build_chat_model
from eval import constants as c


def load_judge_config(path: Path = c.JUDGE_CONFIG_PATH) -> Dict[str, Any]:
    """Load the judge config YAML.

    Parameters
    ----------
    path: Path
        Judge config file.

    Returns
    -------
    Dict[str, Any]
        Parsed config.
    """
    return yaml.safe_load(Path(path).read_text(encoding=c.TEXT_ENCODING))


class ClaudeJudge(DeepEvalBaseLLM):
    """DeepEval judge backed by a Claude model through LangChain.

    Structured outputs use tool calling (``with_structured_output``). Token
    usage is accumulated on the instance so judge cost can be attributed to a
    case; create one instance per case for per-case accounting.

    Parameters
    ----------
    model_id: str
        Claude model ID.
    temperature: Optional[float]
        Sampling temperature, or ``None`` for the model default.
    max_tokens: int
        Maximum output tokens per judge call.
    timeout_s: float
        HTTP timeout per call.
    provider: str
        Provider name passed to the chat model factory.
    """

    def __init__(
        self,
        model_id: str,
        temperature: Optional[float] = None,
        max_tokens: int = 2000,
        timeout_s: float = 120.0,
        provider: str = "anthropic",
    ) -> None:
        self._model_id = model_id
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._timeout_s = timeout_s
        self._provider = provider
        self.input_tokens = 0
        self.output_tokens = 0
        self.calls = 0
        super().__init__(model_id)

    @classmethod
    def from_config(cls, cfg: Dict[str, Any]) -> "ClaudeJudge":
        """Build a judge from the parsed judge config.

        Parameters
        ----------
        cfg: Dict[str, Any]
            Judge config.

        Returns
        -------
        ClaudeJudge
            New judge instance.
        """
        return cls(
            model_id=cfg[c.JCFG_MODEL_ID],
            temperature=cfg.get(c.JCFG_TEMPERATURE),
            max_tokens=int(cfg.get(c.JCFG_MAX_TOKENS, 2000)),
            timeout_s=float(cfg.get(c.JCFG_TIMEOUT_S, 120)),
            provider=cfg.get(c.JCFG_PROVIDER, "anthropic"),
        )

    def load_model(self, *args: Any, **kwargs: Any):
        """Create the underlying LangChain chat model.

        Returns
        -------
        BaseChatModel
            Chat model.
        """
        return build_chat_model(
            provider=self._provider,
            model_id=self._model_id,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
            timeout_s=self._timeout_s,
        )

    def _account(self, message: Any) -> None:
        """Add a response's token usage to the running totals.

        Parameters
        ----------
        message: Any
            ``AIMessage`` returned by the model.
        """
        usage = getattr(message, "usage_metadata", None) or {}
        self.input_tokens += int(usage.get(c.USAGE_INPUT_TOKENS, 0))
        self.output_tokens += int(usage.get(c.USAGE_OUTPUT_TOKENS, 0))
        self.calls += 1

    @staticmethod
    def _text(message: Any) -> str:
        content = getattr(message, "content", "")
        if isinstance(content, str):
            return content
        return "".join(b.get("text", "") for b in content if isinstance(b, dict))

    def generate(self, prompt: str, schema: Optional[Type[BaseModel]] = None, **kwargs: Any) -> Any:
        """Synchronous judge call.

        Parameters
        ----------
        prompt: str
            Prompt built by the DeepEval metric.
        schema: Optional[Type[BaseModel]]
            Expected output schema.

        Returns
        -------
        Any
            A schema instance when parsing succeeds, else raw text.
        """
        if schema is None:
            msg = self.model.invoke(prompt)
            self._account(msg)
            return self._text(msg)
        out = self.model.with_structured_output(schema, include_raw=True).invoke(prompt)
        self._account(out["raw"])
        return out["parsed"] if out.get("parsed") is not None else self._text(out["raw"])

    async def a_generate(self, prompt: str, schema: Optional[Type[BaseModel]] = None, **kwargs: Any) -> Any:
        """Asynchronous judge call.

        Parameters
        ----------
        prompt: str
            Prompt built by the DeepEval metric.
        schema: Optional[Type[BaseModel]]
            Expected output schema.

        Returns
        -------
        Any
            A schema instance when parsing succeeds, else raw text.
        """
        if schema is None:
            msg = await self.model.ainvoke(prompt)
            self._account(msg)
            return self._text(msg)
        out = await self.model.with_structured_output(schema, include_raw=True).ainvoke(prompt)
        self._account(out["raw"])
        return out["parsed"] if out.get("parsed") is not None else self._text(out["raw"])

    def get_model_name(self, *args: Any, **kwargs: Any) -> str:
        """Model name shown by DeepEval.

        Returns
        -------
        str
            Claude model ID.
        """
        return self._model_id
