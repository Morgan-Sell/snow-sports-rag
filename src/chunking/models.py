from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Chunk:
    """Text span produced from a :class:`~snow_sports_rag.ingest.SourceDocument`.

    Used as the unit of embedding and dense retrieval. Metadata fields mirror
    the source document where applicable so hits can be cited by ``doc_id`` and
    section context.

    Attributes
    ----------
    text : str
        Chunk body, often a Markdown section or character window.
    doc_id : str
        Stable id shared with the source document (POSIX path under the KB root).
    entity_type : str
        First path segment of ``doc_id`` (corpus subtree), e.g. ``athletes``.
    section_path : str
        Section label for citation; empty when the strategy does not use
        headings (e.g. fixed window). For H2 chunking, typically the ``##`` title.
    chunk_index : int
        Zero-based index of this chunk in the document's chunk sequence.
    """

    text: str
    doc_id: str
    entity_type: str
    section_path: str
    chunk_index: int
