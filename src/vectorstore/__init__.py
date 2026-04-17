"""Vector index abstraction and ChromaDB backend."""

from .chroma_store import ChromaVectorStore
from .chunks import chunk_row_id, chunk_to_metadata, pack_chunk_upsert
from .factory import vector_store_from_config
from .models import VectorQueryHit, VectorQueryResult
from .protocol import VectorStore

__all__ = [
    "ChromaVectorStore",
    "VectorQueryHit",
    "VectorQueryResult",
    "VectorStore",
    "chunk_row_id",
    "chunk_to_metadata",
    "pack_chunk_upsert",
    "vector_store_from_config",
]
