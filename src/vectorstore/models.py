from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

__all__ = ["VectorQueryHit", "VectorQueryResult"]

__doc__ = """
Structured results from ``VectorStore.query``.
"""


@dataclass(frozen=True)
class VectorQueryHit:
    """One row from a vector similarity search.

    Attributes
    ----------
    id : str
        Upsert id (for example ``doc_id::chunk_index``).
    document : str
        Stored passage text (chunk body).
    distance : float
        Chroma distance for the configured space (for cosine space, lower is
        more similar).
    metadata : Mapping[str, Any]
        Chunk metadata (``doc_id``, ``entity_type``, ``section_path``,
        ``chunk_index``, etc.).
    """

    id: str
    document: str
    distance: float
    metadata: Mapping[str, Any]


@dataclass(frozen=True)
class VectorQueryResult:
    """Ranked hits from :meth:`~snow_sports_rag.vectorstore.protocol.VectorStore.query`.

    Attributes
    ----------
    hits : list of VectorQueryHit
        Best-first order as returned by the backend (ascending distance).
    """

    hits: list[VectorQueryHit]
