from __future__ import annotations

from dataclasses import dataclass, field

from ..generation.models import GeneratedAnswer
from ..retrieval.models import RetrievalHit

__all__ = [
    "SourceCard",
    "StageLatency",
    "PipelineTrace",
    "PipelineResult",
]

__doc__ = """Dataclasses describing a full RAG pipeline execution."""


@dataclass(frozen=True)
class SourceCard:
    """UI-ready view of a single retrieved passage.

    Attributes
    ----------
    index : int
        1-based citation marker shown in the answer.
    chunk_id : str
        Vector store row id (``doc_id::chunk_index``).
    doc_id : str
        Source document id under the knowledge base root.
    section_path : str
        Section label for display (may be empty).
    chunk_index : int
        Zero-based chunk index within the document.
    similarity : float
        Rerank score when available, otherwise retrieval similarity.
    pre_rerank_similarity : float or None
        Retrieval similarity the chunk had in the pre-rerank candidate pool,
        if any. ``None`` when the chunk was not present in the pre-rerank
        list (e.g., baseline retriever with no expansion). Used to compute
        the rerank delta in the "Why this source?" tooltip.
    snippet : str
        Short plain-text excerpt rendered on the card.
    """

    index: int
    chunk_id: str
    doc_id: str
    section_path: str
    chunk_index: int
    similarity: float
    snippet: str
    pre_rerank_similarity: float | None = None


@dataclass(frozen=True)
class StageLatency:
    """Wall-clock timings per pipeline stage, in milliseconds.

    Attributes
    ----------
    expansion_ms : float
        LLM query expansion (0.0 when disabled).
    retrieval_ms : float
        L1 shortlist + filtered L2 + optional global fallback.
    rerank_ms : float
        Reranker call (0.0 when disabled).
    generation_ms : float
        Answer-generation call (0.0 when disabled).
    total_ms : float
        End-to-end latency of :meth:`RAGPipeline.run`.
    """

    expansion_ms: float = 0.0
    retrieval_ms: float = 0.0
    rerank_ms: float = 0.0
    generation_ms: float = 0.0
    total_ms: float = 0.0


@dataclass(frozen=True)
class PipelineTrace:
    """Intermediate stage outputs surfaced in the Debug panel and trace log.

    Attributes
    ----------
    query : str
        The original user question.
    expansions : list of str
        Paraphrases emitted by the LLM (does not include ``query`` itself).
    variants : list of str
        Full set of query strings actually retrieved with (``query`` plus
        deduplicated ``expansions``).
    l1_shortlist : list of str
        Union of L1 doc_ids across variants, in first-seen order.
    l2_pre_rerank : list of RetrievalHit
        Fused L2 candidates before the reranker is applied.
    reranked : list of RetrievalHit
        Output of the reranker; identical to ``l2_pre_rerank[:top_k_out]``
        when reranking is disabled.
    latency : StageLatency
        Wall-clock timings per stage.
    """

    query: str
    expansions: list[str] = field(default_factory=list)
    variants: list[str] = field(default_factory=list)
    l1_shortlist: list[str] = field(default_factory=list)
    l2_pre_rerank: list[RetrievalHit] = field(default_factory=list)
    reranked: list[RetrievalHit] = field(default_factory=list)
    latency: StageLatency = field(default_factory=StageLatency)


@dataclass(frozen=True)
class PipelineResult:
    """End-to-end output of :meth:`RAGPipeline.run`.

    Attributes
    ----------
    query : str
        Original user question (echoed for convenience).
    cards : list of SourceCard
        UI-ready cards corresponding to the final passages.
    answer : GeneratedAnswer or None
        ``None`` when generation is disabled; populated otherwise.
    trace : PipelineTrace
        Intermediate stage outputs for debugging and trace logging.
    trace_id : str
        Correlation id for JSONL trace + feedback records.
    config_hash : str
        Short hash of the effective configuration.
    index_empty : bool
        ``True`` when the underlying vector store has zero rows, so the
        pipeline short-circuited and ``cards`` is guaranteed empty. The UI
        uses this to show a targeted "run the index command" banner instead
        of the generic "no matching sources" message.
    """

    query: str
    cards: list[SourceCard]
    answer: GeneratedAnswer | None
    trace: PipelineTrace
    trace_id: str
    config_hash: str
    index_empty: bool = False
