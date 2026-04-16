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

__all__ = [
    "AppConfig",
    "Chunk",
    "ChunkStrategy",
    "EmbeddingModel",
    "FakeEmbeddingModel",
    "KnowledgeBaseLoader",
    "SentenceTransformerEmbeddingModel",
    "SourceDocument",
    "chunk_strategy_from_config",
    "embedding_model_from_config",
    "load_config",
]
