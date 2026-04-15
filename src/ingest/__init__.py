"""Knowledge-base ingestion: load Markdown sources and metadata."""

from .loader import KnowledgeBaseLoader
from .models import SourceDocument

__all__ = ["KnowledgeBaseLoader", "SourceDocument"]
