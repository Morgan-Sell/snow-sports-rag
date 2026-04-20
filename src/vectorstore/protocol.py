from __future__ import annotations

from typing import Any, Mapping, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from .models import VectorQueryResult

__all__ = ["VectorStore", "FloatMatrix"]

__doc__ = """Abstract vector index for dense retrieval."""

FloatMatrix = NDArray[np.float64]


@runtime_checkable
class VectorStore(Protocol):
    """Key--value vector index with similarity search.

    Implementations persist data on disk or in memory depending on config.
    """

    def upsert(
        self,
        *,
        ids: list[str],
        embeddings: FloatMatrix,
        documents: list[str],
        metadatas: list[Mapping[str, str | int | float | bool]],
    ) -> None:
        """Insert or replace vectors and sidecar fields.

        Parameters
        ----------
        ids : list of str
            Stable primary keys (unique per chunk).
        embeddings : ndarray of shape (n, dim)
            L2-normalized or raw vectors matching the index configuration.
        documents : list of str
            Chunk text aligned with ``ids``.
        metadatas : list of mapping
            Per-row metadata. Chroma requires at least one key per row; use
            corpus fields such as ``doc_id`` and ``chunk_index``.

        Raises
        ------
        ValueError
            If list lengths or embedding shape ``n`` are inconsistent.
        """
        ...

    def count(self) -> int:
        """Return the number of rows currently stored in this index.

        Returns
        -------
        int
            Non-negative row count. Zero indicates the collection exists but
            has not yet been populated (the common "forgot to run ``index``"
            state for a fresh checkout).
        """
        ...

    def query(
        self,
        *,
        query_embedding: NDArray[np.float64],
        k: int,
        where: Mapping[str, Any] | None = None,
    ) -> VectorQueryResult:
        """Return up to ``k`` nearest neighbors to ``query_embedding``.

        Parameters
        ----------
        query_embedding : ndarray of shape (dim,)
            Query vector matching the stored embedding width.
        k : int
            Maximum hits to return (clamped to a positive value by the caller).
        where : mapping or None, optional
            Backend-specific metadata filter (Chroma ``where``). When
            ``None``, the whole collection is searched.

        Returns
        -------
        VectorQueryResult
            Ranked hits; distance interpretation depends on index space
            (cosine, L2, inner product).
        """
        ...
