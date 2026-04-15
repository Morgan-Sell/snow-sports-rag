from __future__ import annotations

import re
from typing import Protocol, runtime_checkable

from ..ingest.models import SourceDocument
from .models import Chunk

__doc__ = """Chunking strategies and internal split/merge helpers.

Implements markdown header, recursive character, and fixed-window strategies
behind a shared :class:`ChunkStrategy` protocol.
"""


@runtime_checkable
class ChunkStrategy(Protocol):
    """Protocol for splitting :class:`~snow_sports_rag.ingest.SourceDocument` values.

    Concrete implementations live in this module and are constructed via
    :func:`~snow_sports_rag.chunking.factory.chunk_strategy_from_config`.
    """

    def chunk(self, document: SourceDocument) -> list[Chunk]:
        """Split ``document`` into ordered retrieval chunks.

        Parameters
        ----------
        document : SourceDocument
            Ingested Markdown document.

        Returns
        -------
        list of Chunk
            Chunks in document order; ``chunk_index`` runs from zero within the
            document.
        """
        ...


def _clamp_overlap(chunk_size: int, chunk_overlap: int) -> int:
    """Validate size/overlap and return overlap clamped to ``chunk_size``.

    Parameters
    ----------
    chunk_size : int
        Maximum chunk length; must be positive.
    chunk_overlap : int
        Desired overlap between consecutive chunks; must be non-negative and not
        greater than ``chunk_size``.

    Returns
    -------
    int
        ``chunk_overlap`` (unchanged when valid).

    Raises
    ------
    ValueError
        If ``chunk_size`` is not positive, ``chunk_overlap`` is negative, or
        ``chunk_overlap`` exceeds ``chunk_size``.
    """
    if chunk_size <= 0:
        msg = f"chunk_size must be positive, got {chunk_size}"
        raise ValueError(msg)
    if chunk_overlap < 0:
        msg = f"chunk_overlap must be non-negative, got {chunk_overlap}"
        raise ValueError(msg)
    if chunk_overlap > chunk_size:
        msg = (
            f"chunk_overlap ({chunk_overlap}) must not exceed chunk_size ({chunk_size})"
        )
        raise ValueError(msg)
    return chunk_overlap


def _window_chunks(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Return fixed-size overlapping windows over ``text``.

    Parameters
    ----------
    text : str
        Input string; empty input yields an empty list.
    chunk_size : int
        Window length in characters.
    chunk_overlap : int
        Characters shared with the next window; validated via :func:`_clamp_overlap`.

    Returns
    -------
    list of str
        Consecutive substrings covering ``text``; the final window may be
        shorter than ``chunk_size``.

    Raises
    ------
    ValueError
        If ``chunk_size`` or ``chunk_overlap`` fail validation in
        :func:`_clamp_overlap`.
    """
    if not text:
        return []
    size = chunk_size
    ov = _clamp_overlap(chunk_size, chunk_overlap)
    n = len(text)
    out: list[str] = []
    start = 0
    while start < n:
        end = min(start + size, n)
        out.append(text[start:end])
        if end >= n:
            break
        nxt = end - ov
        if nxt <= start:
            nxt = start + 1
        start = nxt
    return out


def _join_docs(docs: list[str], separator: str) -> str | None:
    """Join non-empty parts with ``separator`` and strip outer whitespace.

    Parameters
    ----------
    docs : list of str
        Fragments to join.
    separator : str
        Glue string between fragments.

    Returns
    -------
    str or None
        Joined string, or ``None`` if the result is empty after stripping.
    """
    text = separator.join(docs).strip()
    return text or None


def _merge_splits(
    splits: list[str],
    *,
    separator: str,
    chunk_size: int,
    chunk_overlap: int,
) -> list[str]:
    """Merge short splits into length-bounded chunks with overlap.

    Behavior matches LangChain's ``TextSplitter._merge_splits`` for character
    lengths: greedily pack splits, then drop prefixes from the working buffer
    until overlap and size constraints allow appending the next split.

    Parameters
    ----------
    splits : list of str
        Pieces to merge (for example paragraph or line fragments).
    separator : str
        String reinserted between merged pieces (may be empty).
    chunk_size : int
        Maximum length of each output chunk.
    chunk_overlap : int
        Target overlap between consecutive output chunks.

    Returns
    -------
    list of str
        Merged chunks, each at most ``chunk_size`` characters except in edge
        cases inherited from the reference algorithm.

    Raises
    ------
    ValueError
        If ``chunk_size`` or ``chunk_overlap`` fail validation in
        :func:`_clamp_overlap`.
    """
    _clamp_overlap(chunk_size, chunk_overlap)
    separator_len = len(separator)
    docs: list[str] = []
    current_doc: list[str] = []
    total = 0
    for d in splits:
        len_ = len(d)
        if total + len_ + (separator_len if len(current_doc) > 0 else 0) > chunk_size:
            if len(current_doc) > 0:
                doc = _join_docs(current_doc, separator)
                if doc is not None:
                    docs.append(doc)
                while total > chunk_overlap or (
                    total + len_ + (separator_len if len(current_doc) > 0 else 0)
                    > chunk_size
                    and total > 0
                ):
                    total -= len(current_doc[0]) + (
                        separator_len if len(current_doc) > 1 else 0
                    )
                    current_doc = current_doc[1:]
        current_doc.append(d)
        total += len_ + (separator_len if len(current_doc) > 1 else 0)
    doc = _join_docs(current_doc, separator)
    if doc is not None:
        docs.append(doc)
    return docs


_H2_LINE = re.compile(r"^##\s+(.+?)\s*$")


def _h2_sections(markdown: str) -> list[tuple[str, str]]:
    """Partition Markdown into ``(h2_title, body)`` pairs in order.

    Parameters
    ----------
    markdown : str
        Full Markdown source.

    Returns
    -------
    list of tuple of (str, str)
        For each segment, the H2 title (without ``##``) and the body text up to
        the next H2. The preamble before the first ``##`` uses an empty title.
        ``###`` and deeper headings remain inside the body of the current H2.

    Notes
    -----
    Only lines matching ``^##\\s+...$`` start a new segment; ``###`` does not.
    """
    lines = markdown.splitlines()
    current_heading = ""
    buf: list[str] = []
    sections: list[tuple[str, str]] = []

    def flush() -> None:
        nonlocal buf
        body = "\n".join(buf).strip()
        if current_heading or body:
            sections.append((current_heading, body))
        buf = []

    for line in lines:
        m = _H2_LINE.match(line)
        if m:
            flush()
            current_heading = m.group(1).strip()
        else:
            buf.append(line)
    flush()
    return sections


class MarkdownHeaderChunkStrategy:
    """Chunk at ``##`` headings; window long section bodies.

    Each H2 section becomes one or more :class:`Chunk` records sharing the same
    ``section_path``. The preamble before the first H2 uses an empty
    ``section_path``.

    Parameters
    ----------
    chunk_size : int, default 512
        Maximum characters per chunk within a section body.
    chunk_overlap : int, default 64
        Overlap when a section is split into multiple windows.
    min_section_chars : int, default 0
        If positive, skip sections whose stripped body length is below this
        threshold.

    Notes
    -----
    Empty section bodies are omitted. Validation of ``chunk_size`` /
    ``chunk_overlap`` occurs when windowing runs.
    """

    def __init__(
        self,
        *,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        min_section_chars: int = 0,
    ) -> None:
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._min_section_chars = min_section_chars

    def chunk(self, document: SourceDocument) -> list[Chunk]:
        """Split by H2, applying a sliding window per section when needed.

        Parameters
        ----------
        document : SourceDocument
            Document whose ``raw_markdown`` is split.

        Returns
        -------
        list of Chunk
            Chunks in reading order with contiguous ``chunk_index`` values.

        Raises
        ------
        ValueError
            If window parameters are invalid when sub-splitting a long section.
        """
        out: list[Chunk] = []
        idx = 0
        for section_path, body in _h2_sections(document.raw_markdown):
            if self._min_section_chars > 0 and len(body) < self._min_section_chars:
                continue
            if not body:
                continue
            parts = (
                _window_chunks(body, self._chunk_size, self._chunk_overlap)
                if len(body) > self._chunk_size
                else [body]
            )
            for part in parts:
                out.append(
                    Chunk(
                        text=part,
                        doc_id=document.doc_id,
                        entity_type=document.entity_type,
                        section_path=section_path,
                        chunk_index=idx,
                    )
                )
                idx += 1
        return out


class RecursiveCharChunkStrategy:
    """Recursively split text using a separator hierarchy, then merge with overlap.

    Mirrors LangChain's ``RecursiveCharacterTextSplitter`` for literal
    separators: try ``\\n\\n``, then ``\\n``, then space, then per-character
    splits, merging short runs up to ``chunk_size``.

    Parameters
    ----------
    chunk_size : int, default 512
        Target maximum characters per output chunk after merging.
    chunk_overlap : int, default 64
        Overlap between consecutive merged chunks.
    separators : list of str, optional
        Separator list, ordered from coarser to finer; defaults to
        ``["\\n\\n", "\\n", " ", ""]``.

    Notes
    -----
    The final empty-string separator enables character-level splitting when no
    coarser boundary fits. :class:`Chunk` values use ``section_path == ""``.
    """

    def __init__(
        self,
        *,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        separators: list[str] | None = None,
    ) -> None:
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._separators = separators or ["\n\n", "\n", " ", ""]

    def chunk(self, document: SourceDocument) -> list[Chunk]:
        """Split ``document.raw_markdown`` with recursive character rules.

        Parameters
        ----------
        document : SourceDocument
            Source Markdown.

        Returns
        -------
        list of Chunk
            One chunk per merged segment; ``section_path`` is always empty.

        Raises
        ------
        ValueError
            If ``chunk_size`` or ``chunk_overlap`` are invalid during merging.
        """
        texts = self._split_text(document.raw_markdown, list(self._separators))
        out: list[Chunk] = []
        for i, text in enumerate(texts):
            out.append(
                Chunk(
                    text=text,
                    doc_id=document.doc_id,
                    entity_type=document.entity_type,
                    section_path="",
                    chunk_index=i,
                )
            )
        return out

    def _split_text(self, text: str, separators: list[str]) -> list[str]:
        """Recursively split ``text`` using ``separators`` (internal).

        Parameters
        ----------
        text : str
            Remaining text to split.
        separators : list of str
            Remaining separators to try, coarse to fine.

        Returns
        -------
        list of str
            Final chunk strings after recursive splitting and merging.
        """
        final_chunks: list[str] = []
        separator = separators[-1]
        new_separators: list[str] = []
        for i, s in enumerate(separators):
            if not s:
                separator = s
                break
            if s in text:
                separator = s
                new_separators = separators[i + 1 :]
                break

        if separator:
            splits = [p for p in text.split(separator) if p != ""]
            merge_sep = separator
        else:
            splits = list(text)
            merge_sep = ""

        good_splits: list[str] = []
        for s in splits:
            if len(s) < self._chunk_size:
                good_splits.append(s)
            else:
                if good_splits:
                    merged = _merge_splits(
                        good_splits,
                        separator=merge_sep,
                        chunk_size=self._chunk_size,
                        chunk_overlap=self._chunk_overlap,
                    )
                    final_chunks.extend(merged)
                    good_splits = []
                if not new_separators:
                    final_chunks.append(s)
                else:
                    final_chunks.extend(self._split_text(s, new_separators))
        if good_splits:
            merged = _merge_splits(
                good_splits,
                separator=merge_sep,
                chunk_size=self._chunk_size,
                chunk_overlap=self._chunk_overlap,
            )
            final_chunks.extend(merged)
        return final_chunks


class FixedWindowChunkStrategy:
    """Slide a fixed-length window across the full document text.

    Ignores Markdown structure; every :class:`Chunk` has ``section_path == ""``.

    Parameters
    ----------
    chunk_size : int, default 512
        Window length in characters.
    chunk_overlap : int, default 64
        Characters shared between consecutive windows.

    Notes
    -----
    :meth:`chunk` calls :func:`_clamp_overlap`; invalid size or overlap raises
    ``ValueError`` when the document text is non-empty.
    """

    def __init__(
        self,
        *,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
    ) -> None:
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap

    def chunk(self, document: SourceDocument) -> list[Chunk]:
        """Window across ``document.raw_markdown``.

        Parameters
        ----------
        document : SourceDocument
            Document to window.

        Returns
        -------
        list of Chunk
            Sliding-window chunks; empty document yields an empty list.

        Raises
        ------
        ValueError
            If window parameters are invalid and the document is non-empty.
        """
        parts = _window_chunks(
            document.raw_markdown,
            self._chunk_size,
            self._chunk_overlap,
        )
        return [
            Chunk(
                text=p,
                doc_id=document.doc_id,
                entity_type=document.entity_type,
                section_path="",
                chunk_index=i,
            )
            for i, p in enumerate(parts)
        ]
