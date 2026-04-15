"""
Snow Sports RAG library: Phase 0 ingestion and configuration.

The installable import name is ``snow_sports_rag``; implementation files live
under ``src/`` in the repository.
"""

from .config import AppConfig, load_config
from .ingest import KnowledgeBaseLoader, SourceDocument

__all__ = [
    "AppConfig",
    "KnowledgeBaseLoader",
    "SourceDocument",
    "load_config",
]
