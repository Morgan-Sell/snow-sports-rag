from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np

from ..retrieval.models import RetrievalHit

__all__ = ["CrossEncoderReranker"]

__doc__ = """Local cross-encoder reranking (sentence-transformers)."""


class CrossEncoderReranker:
    """Score (query, passage) pairs with a cross-encoder and sort by score.

    Parameters
    ----------
    model_name : str
        Hugging Face id or path (e.g. ``cross-encoder/ms-marco-MiniLM-L-6-v2``).
    device : str or None, optional
        Passed through to ``CrossEncoder`` when supported (e.g. ``cpu``, ``cuda``).
    """

    def __init__(
        self,
        model_name: str,
        *,
        device: str | None = None,
    ) -> None:
        """Defer ``CrossEncoder`` construction until first :meth:`rerank` call.

        Parameters
        ----------
        model_name : str
            Hub checkpoint id for ``sentence_transformers.CrossEncoder``.
        device : str or None, optional
            Optional device string (``cpu``, ``cuda``, …).
        """
        self._model_name = model_name
        self._device = device
        self._model: Any = None

    def _ensure_model(self) -> None:
        """Lazy-import and construct the cross-encoder exactly once."""
        if self._model is not None:
            return
        from sentence_transformers import CrossEncoder

        kwargs: dict[str, Any] = {}
        if self._device is not None:
            kwargs["device"] = self._device
        self._model = CrossEncoder(self._model_name, **kwargs)

    def rerank(
        self,
        query: str,
        hits: list[RetrievalHit],
        *,
        top_k: int,
    ) -> list[RetrievalHit]:
        """Score ``(query, passage)`` pairs and return the best ``top_k`` rows.

        Parameters
        ----------
        query : str
            User question identical to retrieval input.
        hits : list of RetrievalHit
            Candidate passages (caller controls length, e.g. ``top_n_in``).
        top_k : int
            Number of hits to return after sorting by cross-encoder score.

        Returns
        -------
        list of RetrievalHit
            Reordered subset. ``similarity`` is the raw score; ``distance`` is
            its negation.
        """
        if not hits:
            return []
        k = max(1, int(top_k))
        self._ensure_model()
        pairs = [(query, h.text) for h in hits]
        raw_scores = self._model.predict(pairs)
        scores = np.asarray(raw_scores, dtype=np.float64).reshape(-1)
        order = np.argsort(-scores)
        out: list[RetrievalHit] = []
        for idx in order[:k]:
            i = int(idx)
            hit = hits[i]
            s = float(scores[i])
            out.append(replace(hit, similarity=s, distance=-s))
        return out
