from __future__ import annotations

from ..retrieval.models import RetrievalHit

__all__ = ["IdentityReranker"]

__doc__ = """Passthrough reranker (no-op scoring)."""


class IdentityReranker:
    """Return the first ``top_k`` hits in input order (no re-scoring)."""

    def rerank(
        self,
        _query: str,
        hits: list[RetrievalHit],
        *,
        top_k: int,
    ) -> list[RetrievalHit]:
        """Passthrough slice used when reranking is disabled or backend is ``noop``.

        Parameters
        ----------
        _query : str
            Ignored; present for :class:`~snow_sports_rag.rerank.protocol.Reranker`
            API parity.
        hits : list of RetrievalHit
            Retrieval output in caller-defined order.
        top_k : int
            Maximum rows to return (may be zero).

        Returns
        -------
        list of RetrievalHit
            ``hits[:top_k]`` as a shallow copy.
        """
        k = max(0, int(top_k))
        return list(hits[:k])
