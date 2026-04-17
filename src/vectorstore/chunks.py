from __future__ import annotations

from ..chunking.models import Chunk
from .protocol import FloatMatrix

__all__ = ["chunk_row_id", "chunk_to_metadata", "l1_summary_row_id", "pack_chunk_upsert"]

__doc__ = """
Build upsert payloads from :class:`~snow_sports_rag.chunking.models.Chunk` rows.
"""


def l1_summary_row_id(doc_id: str) -> str:
    """Stable vector id for the Phase 2.1 document-summary (L1) row.

    Parameters
    ----------
    doc_id : str
        Source document id (POSIX path under the KB root).

    Returns
    -------
    str
        Distinct from :func:`chunk_row_id` for ``chunk_index >= 0``.
    """
    return f"{doc_id}::__l1_summary__"


def chunk_row_id(chunk: Chunk) -> str:
    """Stable id for a chunk inside the vector store.

    Parameters
    ----------
    chunk : Chunk
        Chunk whose ``doc_id`` and ``chunk_index`` identify the row.

    Returns
    -------
    str
        ``{doc_id}::{chunk_index}`` (valid for Chroma id strings in practice).
    """
    return f"{chunk.doc_id}::{chunk.chunk_index}"


def chunk_to_metadata(chunk: Chunk) -> dict[str, str | int]:
    """Metadata dict suitable for Chroma ``metadatas`` (non-empty).

    Parameters
    ----------
    chunk : Chunk
        Source row.

    Returns
    -------
    dict
        ``doc_id``, ``entity_type``, ``section_path``, ``chunk_index``.
    """
    return {
        "doc_id": chunk.doc_id,
        "entity_type": chunk.entity_type,
        "section_path": chunk.section_path,
        "chunk_index": chunk.chunk_index,
    }


def pack_chunk_upsert(
    chunks: list[Chunk],
    embeddings: FloatMatrix,
) -> tuple[list[str], list[str], list[dict[str, str | int]], FloatMatrix]:
    """Build ``upsert`` arguments from aligned chunks and embedding rows.

    Parameters
    ----------
    chunks : list of Chunk
        Chunks in batch order.
    embeddings : ndarray of shape (n, dim)
        Embedding matrix with ``n == len(chunks)``.

    Returns
    -------
    ids : list of str
        Row ids from :func:`chunk_row_id`.
    documents : list of str
        Chunk ``text`` fields.
    metadatas : list of dict
        One :func:`chunk_to_metadata` dict per chunk.
    embeddings : ndarray
        Unmodified ``embeddings`` reference.

    Raises
    ------
    ValueError
        If ``chunks`` and ``embeddings`` row counts differ.
    """
    if len(chunks) != embeddings.shape[0]:
        msg = f"len(chunks)={len(chunks)} != embeddings.shape[0]={embeddings.shape[0]}"
        raise ValueError(msg)
    ids = [chunk_row_id(c) for c in chunks]
    documents = [c.text for c in chunks]
    metadatas = [chunk_to_metadata(c) for c in chunks]
    return ids, documents, metadatas, embeddings
