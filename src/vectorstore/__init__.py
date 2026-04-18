"""Vector index abstraction and ChromaDB backend."""

from .chroma_store import ChromaVectorStore
from .chunks import (
    chunk_row_id,
    chunk_to_metadata,
    l1_summary_row_id,
    pack_chunk_upsert,
)
from .factory import chroma_l2_l1_stores_from_config, vector_store_from_config
from .models import VectorQueryHit, VectorQueryResult
from .protocol import VectorStore

__all__ = [
    "ChromaVectorStore",
    "VectorQueryHit",
    "VectorQueryResult",
    "VectorStore",
    "chroma_l2_l1_stores_from_config",
    "chunk_row_id",
    "chunk_to_metadata",
    "l1_summary_row_id",
    "pack_chunk_upsert",
    "vector_store_from_config",
]
