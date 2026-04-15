from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SourceDocument:
    """One Markdown file from the corpus after ingestion.

    Attributes
    ----------
    doc_id : str
        Stable path relative to the knowledge base root, POSIX-style (e.g.
        ``athletes/Chloe Kim.md``).
    entity_type : str
        First path segment (folder) under the knowledge base, e.g. ``athletes``.
        Empty if the file lives at the root of the knowledge base.
    title : str
        Text of the first ATX H1 line, or the file stem if no H1 exists.
    raw_markdown : str
        Full file contents as UTF-8 text.
    headings : tuple[str, ...]
        In-order list of ``##`` and ``###`` heading texts (markers stripped).
    """

    doc_id: str
    entity_type: str
    title: str
    raw_markdown: str
    headings: tuple[str, ...]
