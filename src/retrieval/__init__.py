"""Dense retrieval, index construction, and Phase 2.1 hierarchical retrieval."""

from .baseline import BaselineRetriever, IndexBuilder
from .hierarchical import HierarchicalRetriever
from .l1_summary import l1_summary_text
from .manifest import ManifestReadableStore, validate_embedder_against_manifest
from .models import RetrievalHit
from .scoring import chroma_cosine_distance_to_similarity

__all__ = [
    "BaselineRetriever",
    "HierarchicalRetriever",
    "IndexBuilder",
    "ManifestReadableStore",
    "RetrievalHit",
    "chroma_cosine_distance_to_similarity",
    "l1_summary_text",
    "validate_embedder_against_manifest",
]
