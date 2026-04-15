"""Snow Sports RAG library: ingestion, chunking, and configuration.

The installable name is ``snow_sports_rag``; implementation modules live under
``src/`` in the repository while the project directory uses hyphens.
"""

from .chunking import Chunk, ChunkStrategy, chunk_strategy_from_config
from .config import AppConfig, load_config
from .ingest import KnowledgeBaseLoader, SourceDocument

__all__ = [
    "AppConfig",
    "Chunk",
    "ChunkStrategy",
    "KnowledgeBaseLoader",
    "SourceDocument",
    "chunk_strategy_from_config",
    "load_config",
]
