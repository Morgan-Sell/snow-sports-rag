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
from .generation import (
    AnswerGenerator,
    AnthropicAnswerGenerator,
    FakeAnswerGenerator,
    GeneratedAnswer,
    HuggingFaceAnswerGenerator,
    OpenAIAnswerGenerator,
    SourceCitation,
    answer_generator_from_config,
)
from .ingest import KnowledgeBaseLoader, SourceDocument
from .pipeline import (
    PRESETS,
    PipelineResult,
    PipelineTrace,
    RAGPipeline,
    RetrievalPreset,
    SourceCard,
    StageLatency,
    TraceLogger,
    compute_config_hash,
    resolve_preset,
)
from .retrieval import (
    BaselineRetriever,
    HierarchicalRetriever,
    IndexBuilder,
    RetrievalHit,
    chroma_cosine_distance_to_similarity,
    l1_summary_text,
    validate_embedder_against_manifest,
)
from .vectorstore import (
    ChromaVectorStore,
    VectorQueryHit,
    VectorQueryResult,
    VectorStore,
    chroma_l2_l1_stores_from_config,
    chunk_row_id,
    pack_chunk_upsert,
    vector_store_from_config,
)

__all__ = [
    "AnswerGenerator",
    "AnthropicAnswerGenerator",
    "AppConfig",
    "BaselineRetriever",
    "Chunk",
    "ChunkStrategy",
    "ChromaVectorStore",
    "FakeAnswerGenerator",
    "GeneratedAnswer",
    "HierarchicalRetriever",
    "HuggingFaceAnswerGenerator",
    "EmbeddingModel",
    "FakeEmbeddingModel",
    "IndexBuilder",
    "KnowledgeBaseLoader",
    "OpenAIAnswerGenerator",
    "PRESETS",
    "PipelineResult",
    "PipelineTrace",
    "RAGPipeline",
    "RetrievalPreset",
    "SourceCard",
    "SourceCitation",
    "StageLatency",
    "TraceLogger",
    "answer_generator_from_config",
    "compute_config_hash",
    "resolve_preset",
    "chroma_l2_l1_stores_from_config",
    "l1_summary_text",
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
