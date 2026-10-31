"""Versioned price table: cost is computed from tokens at scoring time."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from eval import constants as c
from eval.io import read_json


class PriceTable:
    """Token prices per model, loaded from ``pricing/price_table.json``.

    Parameters
    ----------
    data: Dict[str, Any]
        Parsed price table.
    """

    def __init__(self, data: Dict[str, Any]) -> None:
        self.version: str = data[c.PT_VERSION]
        self.currency: str = data[c.PT_CURRENCY]
        self.unit: int = int(data[c.PT_UNIT_TOKENS])
        self.models: Dict[str, Dict[str, float]] = data[c.PT_MODELS]
        self.aliases: Dict[str, str] = data.get(c.PT_ALIASES, {})

    @classmethod
    def load(cls, path: Path = c.PRICE_TABLE_PATH) -> "PriceTable":
        """Load the price table file.

        Parameters
        ----------
        path: Path
            Price table JSON.

        Returns
        -------
        PriceTable
            Loaded table.
        """
        return cls(read_json(path))

    def resolve(self, reported_name: str) -> Optional[str]:
        """Map a model name reported by the SDK to a price-table key.

        Tries an exact match, then the alias map, then the longest key that
        prefixes the reported name (e.g. a dated snapshot ID).

        Parameters
        ----------
        reported_name: str
            Model name from usage metadata.

        Returns
        -------
        Optional[str]
            Price-table key, or ``None`` if unpriced.
        """
        if reported_name in self.models:
            return reported_name
        if reported_name in self.aliases:
            return self.aliases[reported_name]
        prefixes = [k for k in self.models if reported_name.startswith(k)]
        return max(prefixes, key=len) if prefixes else None

    def call_cost(self, model_key: str, input_tokens: int, output_tokens: int) -> float:
        """Cost of a call (or of aggregated calls of similar size).

        Parameters
        ----------
        model_key: str
            Price-table key.
        input_tokens: int
            Input tokens.
        output_tokens: int
            Output tokens.

        Returns
        -------
        float
            Cost in the table currency. Long-context tiers are applied on the
            given input size, which is exact for single-call usage.
        """
        p = self.models[model_key]
        long = c.PT_LONG_THRESHOLD in p and input_tokens > p[c.PT_LONG_THRESHOLD]
        pin = p[c.PT_INPUT_LONG] if long else p[c.PT_INPUT]
        pout = p[c.PT_OUTPUT_LONG] if long else p[c.PT_OUTPUT]
        return (input_tokens * pin + output_tokens * pout) / self.unit

    def usage_cost(self, usage: Dict[str, Dict[str, int]]) -> Tuple[float, List[str]]:
        """Cost of a usage dict keyed by reported model name.

        Parameters
        ----------
        usage: Dict[str, Dict[str, int]]
            model -> {input_tokens, output_tokens, ...}.

        Returns
        -------
        Tuple[float, List[str]]
            Total cost and the reported names that could not be priced.
        """
        total, unpriced = 0.0, []
        for name, u in (usage or {}).items():
            key = self.resolve(name)
            if key is None:
                unpriced.append(name)
                continue
            total += self.call_cost(key, u.get(c.USAGE_INPUT_TOKENS, 0), u.get(c.USAGE_OUTPUT_TOKENS, 0))
        return total, unpriced
