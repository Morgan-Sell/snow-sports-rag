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
from .retrieval import (
    BaselineRetriever,
    IndexBuilder,
    RetrievalHit,
    chroma_cosine_distance_to_similarity,
    validate_embedder_against_manifest,
)
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
    "BaselineRetriever",
    "Chunk",
    "ChunkStrategy",
    "ChromaVectorStore",
    "EmbeddingModel",
    "FakeEmbeddingModel",
    "IndexBuilder",
    "KnowledgeBaseLoader",
    "RetrievalHit",
    "SentenceTransformerEmbeddingModel",
    "SourceDocument",
    "VectorQueryHit",
    "VectorQueryResult",
    "VectorStore",
    "chunk_row_id",
    "chroma_cosine_distance_to_similarity",
    "chunk_strategy_from_config",
    "embedding_model_from_config",
    "load_config",
    "pack_chunk_upsert",
    "validate_embedder_against_manifest",
    "vector_store_from_config",
]
