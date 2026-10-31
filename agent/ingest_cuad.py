"""Convert CUADv1.json (SQuAD format) into the agent corpus and an annotation sidecar.

Usage
-----
    python -m agent.ingest_cuad path/to/CUADv1.json

Outputs
-------
agent/corpus/passages.jsonl
    One retrievable passage per line: doc_id, passage_id, title, section, text,
    source_file. ``title`` is "<Document Name> between <Party A> and <Party B>"
    built from CUAD's labels (falling back to the CUAD file name), so the
    retriever can match questions that name the contract.
agent/corpus/cuad_annotations.jsonl
    CUAD's lawyer labels mapped to passage IDs: one row per (contract, clause
    category), with the answer spans and the passages that contain them. Use it
    to build golden cases with exact ``expected_sources``.

Contracts get stable IDs (``CUAD-0001`` ...) in sorted-title order, so re-running
the script on the same file reproduces the same IDs.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

from agent import constants as c

TARGET_CHARS = 1200   # merge paragraphs up to about this size
MAX_CHARS = 2000      # split any longer paragraph at sentence boundaries
MAX_HEADING_CHARS = 90
MIN_CHARS = 200       # shorter fragments are folded into a neighbour

_BLANK_LINES = re.compile(r"\n\s*\n")
_WS = re.compile(r"\s+")
_SENTENCE_END = re.compile(r"(?<=[.;:])\s+")
_HEADING_PREFIX = re.compile(r"^(ARTICLE|SECTION|SCHEDULE|EXHIBIT|ANNEX|APPENDIX)\b", re.IGNORECASE)
_ENTITY = re.compile(
    r"\b(Inc|Incorporated|Corp|Corporation|LLC|L\.L\.C|Ltd|Limited|LP|L\.P|PLC|S\.A|GmbH|AG|N\.V|B\.V|"
    r"Co|Company|Pte|Pty|SA|SE|LLP)\b\.?", re.IGNORECASE)
MAX_DOC_NAME_CHARS = 80
MAX_PARTY_CHARS = 70
CLAUSE_PARTIES = "Parties"
CLAUSE_DOC_NAME = "Document Name"


def readable_doc_name(raw: str) -> Optional[str]:
    """Clean a CUAD "Document Name" span into a short title.

    Parameters
    ----------
    raw: str
        Labelled document-name span.

    Returns
    -------
    Optional[str]
        Title-cased name, or ``None`` if unusable.
    """
    name = _WS.sub(" ", raw).split("(")[0].strip(" .,:;\"'")
    if not name or len(name) > MAX_DOC_NAME_CHARS or not re.search(r"[A-Za-z]{3}", name):
        return None
    return name.title() if name.isupper() else name


def contract_title(doc_name: Optional[str], parties: List[str]) -> Optional[str]:
    """Readable contract title: ``<Document Name> between <A> and <B>``.

    Parameters
    ----------
    doc_name: Optional[str]
        CUAD "Document Name" span.
    parties: List[str]
        CUAD "Parties" spans.

    Returns
    -------
    Optional[str]
        Title, or ``None`` when fewer than two legal-entity names are found.
    """
    entities: List[str] = []
    for p in parties:
        p = _WS.sub(" ", p).strip(" .,;:\"'()")
        words = [w for w in re.findall(r"[A-Za-z][A-Za-z.&'-]*", p) if not _ENTITY.fullmatch(w)]
        if 4 <= len(p) <= MAX_PARTY_CHARS and _ENTITY.search(p) and words and '"' not in p \
                and p.lower() not in (e.lower() for e in entities):
            entities.append(p)
    name = readable_doc_name(doc_name) if doc_name else None
    if not name or len(entities) < 2:
        return None
    return f"{name} between {entities[0]} and {entities[1]}"


def title_from_qas(qas: List[dict]) -> Optional[str]:
    """Build the readable title from a contract's CUAD labels.

    Parameters
    ----------
    qas: List[dict]
        SQuAD ``qas`` entries of one contract.

    Returns
    -------
    Optional[str]
        Readable title or ``None``.
    """
    by_clause = {clause_name(q[c.SQUAD_ID]): [a[c.SQUAD_ANSWER_TEXT] for a in q[c.SQUAD_ANSWERS]] for q in qas}
    doc = by_clause.get(CLAUSE_DOC_NAME) or [None]
    return contract_title(doc[0], by_clause.get(CLAUSE_PARTIES, []))


@dataclass
class Span:
    """A piece of a contract with its offsets in the original context.

    Parameters
    ----------
    start: int
        Start offset in the raw context.
    end: int
        End offset (exclusive) in the raw context.
    text: str
        Whitespace-normalized text.
    """

    start: int
    end: int
    text: str


@dataclass
class Chunk:
    """A passage before serialization.

    Parameters
    ----------
    start: int
        Start offset in the raw context.
    end: int
        End offset in the raw context.
    section: str
        Most recent heading before or at the chunk.
    text: str
        Passage text.
    """

    start: int
    end: int
    section: str
    text: str


def paragraphs(context: str) -> Iterator[Span]:
    """Split a contract on blank lines, keeping raw offsets.

    Parameters
    ----------
    context: str
        Raw contract text.

    Yields
    ------
    Span
        Non-empty paragraphs with normalized whitespace.
    """
    pos = 0
    for m in list(_BLANK_LINES.finditer(context)) + [None]:
        end = m.start() if m else len(context)
        raw = context[pos:end]
        text = _WS.sub(" ", raw).strip()
        if text:
            lead = len(raw) - len(raw.lstrip())
            yield Span(pos + lead, end, text)
        if m:
            pos = m.end()


def split_long(span: Span) -> List[Span]:
    """Split an oversized paragraph at sentence boundaries.

    Offsets of the pieces are approximate (proportional), which is enough to
    map annotation spans to passages.

    Parameters
    ----------
    span: Span
        Paragraph.

    Returns
    -------
    List[Span]
        Pieces of at most about ``MAX_CHARS`` characters.
    """
    if len(span.text) <= MAX_CHARS:
        return [span]
    pieces, buf = [], ""
    for sent in _hard_wrap(_SENTENCE_END.split(span.text)):
        if buf and len(buf) + len(sent) + 1 > TARGET_CHARS:
            pieces.append(buf)
            buf = sent
        else:
            buf = f"{buf} {sent}".strip()
    if buf:
        pieces.append(buf)
    out, total, cursor = [], len(span.text), 0
    raw_len = span.end - span.start
    for p in pieces:
        s = span.start + int(cursor / total * raw_len)
        cursor += len(p) + 1
        e = span.start + int(min(cursor, total) / total * raw_len)
        out.append(Span(s, e, p))
    return out


def _hard_wrap(sentences: List[str]) -> List[str]:
    """Break sentences longer than ``MAX_CHARS`` at word boundaries.

    Parameters
    ----------
    sentences: List[str]
        Sentence candidates.

    Returns
    -------
    List[str]
        Pieces of at most ``MAX_CHARS`` characters (unless a single word is longer).
    """
    out: List[str] = []
    for sent in sentences:
        while len(sent) > MAX_CHARS:
            cut = sent.rfind(" ", 0, TARGET_CHARS)
            cut = cut if cut > 0 else TARGET_CHARS
            out.append(sent[:cut].strip())
            sent = sent[cut:].strip()
        if sent:
            out.append(sent)
    return out


def merge_small(chunks: List[Chunk]) -> List[Chunk]:
    """Fold passages shorter than ``MIN_CHARS`` into the next passage (or the previous one at the end).

    Parameters
    ----------
    chunks: List[Chunk]
        Passages in order.

    Returns
    -------
    List[Chunk]
        Passages with no fragment shorter than ``MIN_CHARS`` (unless the contract is tiny).
    """
    out: List[Chunk] = []
    carry = None
    for ch in chunks:
        if carry is not None:
            ch = Chunk(carry.start, ch.end, carry.section if len(carry.text) > MAX_HEADING_CHARS else ch.section,
                       f"{carry.text} {ch.text}")
            carry = None
        if len(ch.text) < MIN_CHARS:
            carry = ch
        else:
            out.append(ch)
    if carry is not None:
        if out:
            last = out.pop()
            out.append(Chunk(last.start, carry.end, last.section, f"{last.text} {carry.text}"))
        else:
            out.append(carry)
    return out


def is_heading(text: str) -> bool:
    """Heuristic: short paragraph that is mostly uppercase or starts with ARTICLE/SECTION.

    Parameters
    ----------
    text: str
        Normalized paragraph.

    Returns
    -------
    bool
        True for likely section headings.
    """
    if len(text) > MAX_HEADING_CHARS:
        return False
    if _HEADING_PREFIX.match(text):
        return True
    letters = [ch for ch in text if ch.isalpha()]
    return len(letters) >= 4 and sum(ch.isupper() for ch in letters) / len(letters) >= 0.8


def chunk_contract(context: str) -> List[Chunk]:
    """Merge paragraphs into passages of about ``TARGET_CHARS`` characters.

    A heading starts a new passage and becomes its section label.

    Parameters
    ----------
    context: str
        Raw contract text.

    Returns
    -------
    List[Chunk]
        Passages in document order.
    """
    chunks: List[Chunk] = []
    section = c.CUAD_DEFAULT_SECTION
    cur: List[Span] = []
    cur_section = section

    def flush() -> None:
        if cur:
            chunks.append(Chunk(cur[0].start, cur[-1].end, cur_section, " ".join(s.text for s in cur)))
            cur.clear()

    for para in paragraphs(context):
        if is_heading(para.text):
            if not all(is_heading(x.text) for x in cur):
                flush()
            section = para.text
            cur_section = section
            cur.append(para)
            continue
        for piece in split_long(para):
            size = sum(len(s.text) + 1 for s in cur)
            only_headings = all(is_heading(x.text) for x in cur)
            if cur and size + len(piece.text) > TARGET_CHARS and not only_headings:
                flush()
                cur_section = section
            cur.append(piece)
    flush()
    return merge_small(chunks)


def passages_for_span(chunks: List[Tuple[str, Chunk]], start: int, end: int) -> List[str]:
    """Passage IDs whose raw span overlaps ``[start, end)``.

    Parameters
    ----------
    chunks: List[Tuple[str, Chunk]]
        (passage_id, chunk) pairs of one contract.
    start: int
        Annotation start offset.
    end: int
        Annotation end offset.

    Returns
    -------
    List[str]
        Overlapping passage IDs in document order.
    """
    return [pid for pid, ch in chunks if ch.start < end and start < ch.end]


def clause_name(qa_id: str) -> str:
    """Extract the CUAD clause category from a question ID.

    Parameters
    ----------
    qa_id: str
        ID like ``<title>__Governing Law``.

    Returns
    -------
    str
        Clause category, e.g. ``Governing Law``.
    """
    return qa_id.rsplit(c.CUAD_CLAUSE_SEPARATOR, 1)[-1]


def ingest(source: Path, passages_path: Path, annotations_path: Path) -> Dict[str, int]:
    """Write the corpus and annotation files from CUADv1.json.

    Parameters
    ----------
    source: Path
        Path to CUADv1.json.
    passages_path: Path
        Destination corpus JSONL (overwritten).
    annotations_path: Path
        Destination annotation JSONL (overwritten).

    Returns
    -------
    Dict[str, int]
        Counts: contracts, passages, annotations, answered annotations,
        answer spans that could not be mapped to a passage.
    """
    data = json.loads(Path(source).read_text(encoding=c.TEXT_ENCODING))[c.SQUAD_DATA]
    data = sorted(data, key=lambda d: d[c.SQUAD_TITLE])
    stats = {"contracts": 0, "readable_titles": 0, "passages": 0, "annotations": 0, "answered": 0, "unmapped_spans": 0}

    with Path(passages_path).open("w", encoding=c.TEXT_ENCODING) as pf, \
            Path(annotations_path).open("w", encoding=c.TEXT_ENCODING) as af:
        for idx, contract in enumerate(data, start=1):
            doc_id = c.CUAD_DOC_ID_TEMPLATE.format(prefix=c.CUAD_DOC_PREFIX, index=idx)
            source_file = contract[c.SQUAD_TITLE]
            stats["contracts"] += 1
            for para in contract[c.SQUAD_PARAGRAPHS]:
                title = title_from_qas(para[c.SQUAD_QAS]) or source_file
                stats["readable_titles"] += title != source_file
                chunks = chunk_contract(para[c.SQUAD_CONTEXT])
                ids = [
                    (c.CUAD_PASSAGE_ID_TEMPLATE.format(doc_id=doc_id, index=i), ch)
                    for i, ch in enumerate(chunks, start=1)
                ]
                for pid, ch in ids:
                    pf.write(json.dumps({
                        c.PASSAGE_DOC_ID: doc_id,
                        c.PASSAGE_ID: pid,
                        c.PASSAGE_TITLE: title,
                        c.PASSAGE_SECTION: ch.section,
                        c.PASSAGE_TEXT: ch.text,
                        c.PASSAGE_SOURCE_FILE: source_file,
                    }, ensure_ascii=False) + "\n")
                    stats["passages"] += 1
                for qa in para[c.SQUAD_QAS]:
                    answers = []
                    for ans in qa[c.SQUAD_ANSWERS]:
                        start = int(ans[c.SQUAD_ANSWER_START])
                        text = ans[c.SQUAD_ANSWER_TEXT]
                        pids = passages_for_span(ids, start, start + len(text))
                        stats["unmapped_spans"] += not pids
                        answers.append({c.ANN_TEXT: _WS.sub(" ", text).strip(), c.ANN_PASSAGE_IDS: pids})
                    has_answer = not qa.get(c.SQUAD_IS_IMPOSSIBLE, False) and bool(answers)
                    af.write(json.dumps({
                        c.ANN_DOC_ID: doc_id,
                        c.ANN_CLAUSE: clause_name(qa[c.SQUAD_ID]),
                        c.ANN_QUESTION: qa[c.SQUAD_QUESTION],
                        c.ANN_HAS_ANSWER: has_answer,
                        c.ANN_ANSWERS: answers,
                    }, ensure_ascii=False) + "\n")
                    stats["annotations"] += 1
                    stats["answered"] += has_answer
    return stats


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Build the agent corpus from CUADv1.json")
    parser.add_argument("source", type=Path, help="path to CUADv1.json")
    parser.add_argument("--passages", type=Path, default=c.CORPUS_PATH)
    parser.add_argument("--annotations", type=Path, default=c.CUAD_ANNOTATIONS_PATH)
    args = parser.parse_args()
    stats = ingest(args.source, args.passages, args.annotations)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
