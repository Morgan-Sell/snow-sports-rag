"""Snow Sports RAG library: ingestion, chunking, and configuration.

The installable name is ``snow_sports_rag``; implementation modules live under
``src/`` in the repository while the project directory uses hyphens.
"""

from .chunking import Chunk, ChunkStrategy, chunk_strategy_from_config
from .config import AppConfig, load_config
from .embedding import (
    EmbeddingModel,
    FakeEmbeddingModel,
    SentenceTransformerEmbeddingModel,
    embedding_model_from_config,
)
from .ingest import KnowledgeBaseLoader, SourceDocument
from .vectorstore import (
    ChromaVectorStore,
    VectorQueryHit,
    VectorQueryResult,
    VectorStore,
    chunk_row_id,
    pack_chunk_upsert,
    vector_store_from_config,
)

__all__ = [
    "AppConfig",
    "Chunk",
    "ChunkStrategy",
    "ChromaVectorStore",
    "EmbeddingModel",
    "FakeEmbeddingModel",
    "KnowledgeBaseLoader",
    "SentenceTransformerEmbeddingModel",
    "SourceDocument",
    "VectorQueryHit",
    "VectorQueryResult",
    "VectorStore",
    "chunk_row_id",
    "chunk_strategy_from_config",
    "embedding_model_from_config",
    "load_config",
    "pack_chunk_upsert",
    "vector_store_from_config",
]
