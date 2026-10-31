"""BM25 retriever over the contract passages, plus corpus fingerprinting."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, List

import numpy as np

import snowballstemmer
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


_STEMMER = snowballstemmer.stemmer(c.STEMMER_LANGUAGE)


def tokenize(text: str) -> List[str]:
    """Lowercase, split into alphanumeric tokens, drop stopwords and stem.

    Stemming makes "governs" match "governed" and "laws" match "law".

    Parameters
    ----------
    text: str
        Input text.

    Returns
    -------
    List[str]
        Stemmed tokens.
    """
    return [_stem(t) for t in re.findall(c.TOKEN_PATTERN, text.lower()) if t not in _STOPWORDS]


@lru_cache(maxsize=None)
def _stem(word: str) -> str:
    """Memoized stemming; the vocabulary is far smaller than the token stream.

    Parameters
    ----------
    word: str
        Lower-case token.

    Returns
    -------
    str
        Stem.
    """
    return _STEMMER.stemWord(word)


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
    """Two-stage lexical retriever: pick the contract(s), then the passages.

    Stage 1 scores contract titles against the question (BM25 over one title per
    contract) and keeps the best contract plus any within ``DOC_SCORE_RATIO`` of
    it, up to ``MAX_DOCS``. Stage 2 ranks the passages of those contracts with
    BM25 over section and text, using the question minus the words that only
    name the contract. With several contracts (comparison questions), results
    are interleaved so each contract is represented. If no title matches, it
    falls back to plain BM25 over all passages.

    Parameters
    ----------
    passages: List[Passage]
        Passages to index.
    """

    DOC_SCORE_RATIO = 0.6
    MAX_DOCS = 3

    def __init__(self, passages: List[Passage]) -> None:
        self._passages = passages
        self._doc_ids: List[str] = []
        self._doc_titles: Dict[str, List[str]] = {}
        self._doc_rows: Dict[str, List[int]] = {}
        for i, p in enumerate(passages):
            if p.doc_id not in self._doc_rows:
                self._doc_ids.append(p.doc_id)
                self._doc_titles[p.doc_id] = tokenize(p.title)
                self._doc_rows[p.doc_id] = []
            self._doc_rows[p.doc_id].append(i)
        self._title_bm25 = BM25Okapi([self._doc_titles[d] or ["_"] for d in self._doc_ids])
        self._text_bm25 = BM25Okapi([tokenize(f"{p.section} {p.text}") or ["_"] for p in passages])

    def _candidate_docs(self, tokens: List[str]) -> List[str]:
        """Contracts whose title matches the question.

        Parameters
        ----------
        tokens: List[str]
            Query tokens.

        Returns
        -------
        List[str]
            doc_ids in score order; empty when no title matches.
        """
        scores = self._title_bm25.get_scores(tokens)
        best = float(np.max(scores)) if len(scores) else 0.0
        if best <= 0:
            return []
        order = np.argsort(-scores)[: self.MAX_DOCS]
        return [self._doc_ids[i] for i in order if scores[i] >= self.DOC_SCORE_RATIO * best]

    def retrieve(self, query: str, top_k: int) -> List[Passage]:
        """Return up to top-k passages with a positive score.

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
        tokens = tokenize(query)
        docs = self._candidate_docs(tokens)
        if not docs:
            scores = self._text_bm25.get_scores(tokens)
            ranked = np.argsort(-scores)[:top_k]
            return [self._passages[i] for i in ranked if scores[i] > 0]

        title_words = set().union(*(self._doc_titles[d] for d in docs))
        content = [t for t in tokens if t not in title_words] or tokens
        scores = self._text_bm25.get_scores(content)
        per_doc = [
            [i for i in sorted(self._doc_rows[d], key=lambda i: -scores[i]) if scores[i] > 0]
            for d in docs
        ]
        merged: List[int] = []
        for rank in range(max((len(r) for r in per_doc), default=0)):
            for rows in per_doc:
                if rank < len(rows):
                    merged.append(rows[rank])
        return [self._passages[i] for i in merged[:top_k]]


@lru_cache(maxsize=1)
def default_retriever() -> BM25Retriever:
    """Return a cached retriever over the default corpus.

    Returns
    -------
    BM25Retriever
        Retriever built from ``agent/corpus/passages.jsonl``.
    """
    return BM25Retriever(load_passages())
