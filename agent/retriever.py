"""BM25 retriever over the contract passages, plus corpus fingerprinting."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, List

from rank_bm25 import BM25Okapi

from agent import constants as c

# Common words that add noise to lexical scoring.
_STOPWORDS = frozenset(
    "a an and are as at be by does do for from has have how in is it of on or "
    "the to under what when which who will with this that there their any".split()
)


@dataclass(frozen=True)
class Passage:
    """One retrievable contract passage.

    Parameters
    ----------
    doc_id: str
        Contract identifier, e.g. ``MSA-ACME``.
    passage_id: str
        Stable passage identifier, e.g. ``MSA-ACME-03``.
    title: str
        Contract title.
    section: str
        Section heading.
    text: str
        Passage body.
    """

    doc_id: str
    passage_id: str
    title: str
    section: str
    text: str

    def render(self) -> str:
        """Render the passage as a context block for the prompt.

        Returns
        -------
        str
            ``[ID] title — section`` followed by the text.
        """
        return c.CONTEXT_PASSAGE_TEMPLATE.format(
            passage_id=self.passage_id,
            title=self.title,
            section=self.section,
            text=self.text,
        )


def tokenize(text: str) -> List[str]:
    """Lowercase and split text into alphanumeric tokens without stopwords.

    Parameters
    ----------
    text: str
        Input text.

    Returns
    -------
    List[str]
        Tokens.
    """
    return [t for t in re.findall(c.TOKEN_PATTERN, text.lower()) if t not in _STOPWORDS]


def load_passages(path: Path = c.CORPUS_PATH) -> List[Passage]:
    """Load passages from a JSONL corpus file.

    Parameters
    ----------
    path: Path
        Corpus file path.

    Returns
    -------
    List[Passage]
        Passages in file order.
    """
    passages: List[Passage] = []
    with path.open(encoding=c.TEXT_ENCODING) as fh:
        for line in fh:
            if line.strip():
                row = json.loads(line)
                passages.append(
                    Passage(
                        doc_id=row[c.PASSAGE_DOC_ID],
                        passage_id=row[c.PASSAGE_ID],
                        title=row[c.PASSAGE_TITLE],
                        section=row[c.PASSAGE_SECTION],
                        text=row[c.PASSAGE_TEXT],
                    )
                )
    return passages


def corpus_index(path: Path = c.CORPUS_PATH) -> Dict[str, List[str]]:
    """Map each doc_id to its passage_ids, for source-reference validation.

    Parameters
    ----------
    path: Path
        Corpus file path.

    Returns
    -------
    Dict[str, List[str]]
        ``doc_id -> [passage_id, ...]``.
    """
    index: Dict[str, List[str]] = {}
    for p in load_passages(path):
        index.setdefault(p.doc_id, []).append(p.passage_id)
    return index


def corpus_version(path: Path = c.CORPUS_PATH) -> str:
    """Fingerprint the corpus as a hash of sorted (passage_id, text hash) pairs.

    It changes exactly when a retrieval target is added, removed or edited.

    Parameters
    ----------
    path: Path
        Corpus file path.

    Returns
    -------
    str
        ``sha256:<first 16 hex chars>``.
    """
    pairs = sorted(
        (p.passage_id, hashlib.sha256(p.text.encode(c.TEXT_ENCODING)).hexdigest())
        for p in load_passages(path)
    )
    digest = hashlib.sha256(json.dumps(pairs).encode(c.TEXT_ENCODING)).hexdigest()
    return f"sha256:{digest[:16]}"


class BM25Retriever:
    """Lexical retriever over contract passages.

    Parameters
    ----------
    passages: List[Passage]
        Passages to index.
    """

    def __init__(self, passages: List[Passage]) -> None:
        self._passages = passages
        corpus_tokens = [tokenize(f"{p.title} {p.section} {p.text}") for p in passages]
        self._bm25 = BM25Okapi(corpus_tokens)

    def retrieve(self, query: str, top_k: int) -> List[Passage]:
        """Return the top-k passages with a positive BM25 score.

        Parameters
        ----------
        query: str
            User question.
        top_k: int
            Maximum number of passages.

        Returns
        -------
        List[Passage]
            Passages in rank order; may be shorter than ``top_k``.
        """
        scores = self._bm25.get_scores(tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [self._passages[i] for i in ranked[:top_k] if scores[i] > 0]


@lru_cache(maxsize=1)
def default_retriever() -> BM25Retriever:
    """Return a cached retriever over the default corpus.

    Returns
    -------
    BM25Retriever
        Retriever built from ``agent/corpus/passages.jsonl``.
    """
    return BM25Retriever(load_passages())
