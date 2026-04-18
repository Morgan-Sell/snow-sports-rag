from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..retrieval.models import RetrievalHit

__all__ = ["Reranker"]

__doc__ = """Reranker protocol for Phase 2.3 (cross-encoder or listwise LLM)."""


@runtime_checkable
class Reranker(Protocol):
    """Re-score and reorder retrieval candidates for a single query."""

    def rerank(
        self,
        query: str,
        hits: list[RetrievalHit],
        *,
        top_k: int,
    ) -> list[RetrievalHit]:
        """Return up to ``top_k`` hits in best-first order by the reranker.

        Parameters
        ----------
        query : str
            User question (same string used for retrieval).
        hits : list of RetrievalHit
            Candidate passages; callers typically cap length before calling
            (for example ``rerank.top_n_in``).
        top_k : int
            Maximum number of hits to return after reranking.

        Returns
        -------
        list of RetrievalHit
            Reordered subset; ``similarity`` / ``distance`` may be updated to
            reflect reranker scores where applicable.
        """
        ...
