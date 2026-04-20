"""Phase 3 runtime pipeline shared by the CLI, Gradio UI, and evaluation.

The pipeline glues together hierarchical retrieval, optional query expansion,
optional reranking, and optional grounded generation, while emitting a
structured :class:`PipelineTrace` for the Debug UI and JSONL trace file.
"""

from .models import PipelineResult, PipelineTrace, SourceCard, StageLatency
from .presets import PRESETS, RetrievalPreset, resolve_preset
from .rag_pipeline import RAGPipeline
from .trace import TraceLogger, compute_config_hash

__all__ = [
    "PRESETS",
    "PipelineResult",
    "PipelineTrace",
    "RAGPipeline",
    "RetrievalPreset",
    "SourceCard",
    "StageLatency",
    "TraceLogger",
    "compute_config_hash",
    "resolve_preset",
]
