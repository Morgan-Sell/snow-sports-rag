"""Document chunking for RAG indexing.

Exposes chunk datatypes, a :class:`~snow_sports_rag.chunking.ChunkStrategy`
protocol, concrete splitters, and config-driven factory helpers.
"""

from .factory import chunk_strategy_from_config
from .models import Chunk
from .strategies import (
    ChunkStrategy,
    FixedWindowChunkStrategy,
    MarkdownHeaderChunkStrategy,
    RecursiveCharChunkStrategy,
)

__all__ = [
    "Chunk",
    "ChunkStrategy",
    "FixedWindowChunkStrategy",
    "MarkdownHeaderChunkStrategy",
    "RecursiveCharChunkStrategy",
    "chunk_strategy_from_config",
]
