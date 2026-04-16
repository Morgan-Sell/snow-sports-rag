"""Bi-encoder embeddings for dense retrieval.

Exposes the :class:`~snow_sports_rag.embedding.model.EmbeddingModel` protocol,
Sentence-Transformers and fake backends, normalization helpers, and
:func:`~snow_sports_rag.embedding.factory.embedding_model_from_config`.
"""

from .factory import embedding_model_from_config
from .fake import FakeEmbeddingModel
from .model import EmbeddingModel, l2_normalize_rows
from .sentence_transformer import SentenceTransformerEmbeddingModel

__all__ = [
    "EmbeddingModel",
    "FakeEmbeddingModel",
    "SentenceTransformerEmbeddingModel",
    "embedding_model_from_config",
    "l2_normalize_rows",
]
