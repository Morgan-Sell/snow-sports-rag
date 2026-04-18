"""Dense retrieval, index construction, hierarchical retrieval, and fusion."""

from .baseline import BaselineRetriever, IndexBuilder
from .fusion import fuse_retrieval_hits_max_score, fuse_retrieval_hits_rrf
from .hierarchical import HierarchicalRetriever
from .l1_summary import l1_summary_text
from .manifest import ManifestReadableStore, validate_embedder_against_manifest
from .models import RetrievalHit
from .query_expansion import QueryExpander, SupportsRetrieve
from .scoring import chroma_cosine_distance_to_similarity

__all__ = [
    "BaselineRetriever",
    "HierarchicalRetriever",
    "IndexBuilder",
    "ManifestReadableStore",
    "QueryExpander",
    "RetrievalHit",
    "SupportsRetrieve",
    "chroma_cosine_distance_to_similarity",
    "fuse_retrieval_hits_max_score",
    "fuse_retrieval_hits_rrf",
    "l1_summary_text",
    "validate_embedder_against_manifest",
]
