from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..llm.protocol import LLMClient
from .fusion import fuse_retrieval_hits_max_score, fuse_retrieval_hits_rrf
from .models import RetrievalHit

__all__ = ["QueryExpander", "SupportsRetrieve"]

__doc__ = """Phase 2.2: LLM paraphrases, multi-query retrieve, fuse, cap."""


@runtime_checkable
class SupportsRetrieve(Protocol):
    """Anything that can run dense or hierarchical :meth:`retrieve`."""

    def retrieve(self, query: str, *, k: int | None = None) -> list[RetrievalHit]:
        ...


class QueryExpander:
    """Wrap a retriever with optional LLM query expansion and score fusion.

    When ``enabled`` is false, :meth:`retrieve` delegates to ``inner`` only.

    Parameters
    ----------
    inner : SupportsRetrieve
        Baseline or hierarchical retriever.
    llm : LLMClient
        Supplies paraphrases via :meth:`~LLMClient.expand_query`.
    enabled : bool, default True
        If false, expansion and fusion are skipped.
    num_paraphrases : int, default 3
        Passed to ``llm.expand_query``.
    fusion : str, default 'max_score'
        ``max_score`` or ``rrf`` (case-insensitive).
    rrf_k : int, default 60
        RRF smoothing constant when ``fusion`` is ``rrf``.
    per_query_k : int or None, optional
        ``k`` passed to ``inner.retrieve`` for each query variant. If
        ``None``, uses ``max(top_n_fused, default_k * 5)`` with ``default_k``
        from ``default_inner_k``.
    top_n_fused : int, default 48
        Cap on fused hit count (aligns with rerank ``top_n_pre_rerank``).
    default_inner_k : int, default 8
        Used with ``per_query_k is None`` and as the default final slice when
        ``retrieve(..., k=None)``.
    """

    def __init__(
        self,
        inner: SupportsRetrieve,
        llm: LLMClient,
        *,
        enabled: bool = True,
        num_paraphrases: int = 3,
        fusion: str = "max_score",
        rrf_k: int = 60,
        per_query_k: int | None = None,
        top_n_fused: int = 48,
        default_inner_k: int = 8,
    ) -> None:
        self._inner = inner
        self._llm = llm
        self._enabled = bool(enabled)
        self._num_paraphrases = max(0, int(num_paraphrases))
        self._fusion = str(fusion).strip().lower()
        self._rrf_k = max(1, int(rrf_k))
        self._per_query_k = per_query_k
        self._top_n_fused = max(1, int(top_n_fused))
        self._default_inner_k = max(1, int(default_inner_k))

    def retrieve(self, query: str, *, k: int | None = None) -> list[RetrievalHit]:
        """Expand (optional), multi-query retrieve, fuse, return top-``k``."""
        k_eff = self._default_inner_k if k is None else max(1, int(k))
        if not self._enabled:
            return self._inner.retrieve(query, k=k_eff)

        paraphrases = self._llm.expand_query(
            query,
            num_paraphrases=self._num_paraphrases,
        )
        variants: list[str] = []
        seen: set[str] = set()
        for q in [query, *paraphrases]:
            s = q.strip()
            if s and s not in seen:
                seen.add(s)
                variants.append(s)

        inner_k = self._per_query_k
        if inner_k is None:
            inner_k = max(self._top_n_fused, k_eff * max(5, len(variants)))

        lists = [self._inner.retrieve(q, k=inner_k) for q in variants]

        if self._fusion == "rrf":
            fused = fuse_retrieval_hits_rrf(
                lists,
                top_n=self._top_n_fused,
                rrf_k=self._rrf_k,
            )
        else:
            fused = fuse_retrieval_hits_max_score(lists, top_n=self._top_n_fused)

        return fused[:k_eff]
