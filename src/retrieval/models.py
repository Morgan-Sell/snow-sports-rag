from __future__ import annotations

from dataclasses import dataclass

__all__ = ["RetrievalHit"]

__doc__ = """Types for dense retrieval results."""


@dataclass(frozen=True)
class RetrievalHit:
    """One ranked passage from baseline dense retrieval.

    Attributes
    ----------
    chunk_id : str
        Vector store row id (e.g. ``doc_id::chunk_index``).
    text : str
        Chunk body (passage).
    doc_id : str
        Source document id under the knowledge base root.
    section_path : str
        Section label for citation (may be empty).
    chunk_index : int
        Zero-based chunk index within the source document.
    similarity : float
        ``1.0 - chroma_distance`` (higher is better). Chroma may report cosine
        distances outside ``[0, 1]`` depending on version/implementation, so
        this value is not always clamped to ``[0, 1]``.
    distance : float
        Raw Chroma distance (lower is better; kept for debugging).
    """

    chunk_id: str
    text: str
    doc_id: str
    section_path: str
    chunk_index: int
    similarity: float
    distance: float
