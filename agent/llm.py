"""Chat model factory: LLM Provider API, or an offline fake.

The API key is read from the ``LLM_PROVIDER_KEY`` environment variable, which
is loaded from a ``.env`` file at the project root if present.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, List, Optional

from dotenv import load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from agent import constants as c

_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_ROOT / c.ENV_FILE_NAME)


class MissingApiKeyError(RuntimeError):
    """Raised when an LLM provider model is requested without an API key."""


def get_api_key() -> str:
    """Return the LLM provider API key from the environment.

    Returns
    -------
    str
        The API key.

    Raises
    ------
    MissingApiKeyError
        If ``LLM_PROVIDER_KEY`` is not set.
    """
    key = os.environ.get(c.ENV_LLM_PROVIDER_KEY, "").strip()
    if not key:
        raise MissingApiKeyError(
            f"{c.ENV_LLM_PROVIDER_KEY} is not set. Copy .env.example to .env "
            "and add your LLM provider API key, or use configs/offline.yaml."
        )
    return key


class FakeExtractiveChatModel(BaseChatModel):
    """Offline chat model that answers with the first sentence of passage 1.

    It looks for the first ``[PASSAGE-ID] ...`` block in the last message and
    returns its first sentence with a citation. When no passage block exists it
    returns the not-found marker. Useful to exercise the pipeline with no API
    key and no cost; quality scores are not meaningful.
    """

    model_name: str = "fake-extractive"

    @property
    def _llm_type(self) -> str:
        return c.PROVIDER_FAKE

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        content = str(messages[-1].content)
        match = re.search(r"\[([A-Z0-9-]+)\][^\n]*\n([^\n]+)", content)
        if match:
            passage_id, text = match.group(1), match.group(2)
            sentence = re.split(r"(?<=[.!?])\s", text, maxsplit=1)[0]
            answer = f"{sentence} [{passage_id}]"
        else:
            answer = f"{c.NOT_FOUND_MARKER}: no relevant passage was retrieved."
        n_in = len(content.split())
        n_out = len(answer.split())
        message = AIMessage(
            content=answer,
            usage_metadata={
                "input_tokens": n_in,
                "output_tokens": n_out,
                "total_tokens": n_in + n_out,
            },
            response_metadata={"model_name": self.model_name},
        )
        return ChatResult(generations=[ChatGeneration(message=message)])


def build_chat_model(
    provider: str,
    model_id: str,
    temperature: Optional[float] = None,
    max_tokens: int = 1024,
    timeout_s: float = 60.0,
) -> BaseChatModel:
    """Build a chat model for the agent or the judge.

    Parameters
    ----------
    provider: str
        ``anthropic`` for Claude via the Anthropic API, or ``fake`` for the
        offline extractive model.
    model_id: str
        Claude model ID, e.g. ``claude-haiku-5-5``.
    temperature: Optional[float]
        Sampling temperature. ``None`` omits the parameter, which is required
        for models that reject sampling parameters.
    max_tokens: int
        Maximum output tokens per call.
    timeout_s: float
        HTTP timeout per call, in seconds.

    Returns
    -------
    BaseChatModel
        A LangChain chat model. SDK retries are disabled; the eval runner owns
        retry and backoff so attempts are counted.

    Raises
    ------
    MissingApiKeyError
        If an Anthropic model is requested and no API key is configured.
    ValueError
        If the provider is unknown.
    """
    if provider == c.PROVIDER_FAKE:
        return FakeExtractiveChatModel(model_name=model_id)
    if provider != c.PROVIDER_ANTHROPIC:
        raise ValueError(f"Unknown provider: {provider!r}")

    from langchain_anthropic import ChatAnthropic

    kwargs: dict = {
        "model": model_id,
        "api_key": get_api_key(),
        "max_tokens": max_tokens,
        "timeout": timeout_s,
        "max_retries": 0,
    }
    if temperature is not None:
        kwargs[c.CFG_TEMPERATURE] = temperature
    return ChatAnthropic(**kwargs)
