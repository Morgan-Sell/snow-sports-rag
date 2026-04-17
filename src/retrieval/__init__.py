"""Dense baseline retrieval and index construction (Phase 1.4)."""

from .baseline import BaselineRetriever, IndexBuilder
from .manifest import ManifestReadableStore, validate_embedder_against_manifest
from .models import RetrievalHit
from .scoring import chroma_cosine_distance_to_similarity

__all__ = [
    "BaselineRetriever",
    "IndexBuilder",
    "ManifestReadableStore",
    "RetrievalHit",
    "chroma_cosine_distance_to_similarity",
    "validate_embedder_against_manifest",
]
