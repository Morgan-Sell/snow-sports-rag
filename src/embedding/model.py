from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

__all__ = ["EmbeddingModel", "l2_normalize_rows"]

__doc__ = """Embedding protocol and shared vector normalization helpers.

Concrete backends live in :mod:`snow_sports_rag.embedding.sentence_transformer`
and :mod:`snow_sports_rag.embedding.fake`.
"""

FloatMatrix = NDArray[np.float64]


def l2_normalize_rows(vectors: FloatMatrix, *, eps: float = 1e-12) -> FloatMatrix:
    """L2-normalize each row so cosine similarity equals the dot product.

    Parameters
    ----------
    vectors : ndarray of shape (n, dim)
        Embedding rows to normalize.
    eps : float, optional
        Small constant to avoid division by zero when a row is all zeros.

    Returns
    -------
    ndarray of shape (n, dim)
        Row-wise unit vectors (same dtype as ``vectors``).

    Raises
    ------
    ValueError
        If ``vectors`` is not two-dimensional.
    """
    if vectors.ndim != 2:
        msg = f"expected 2-D array, got shape {vectors.shape}"
        raise ValueError(msg)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    out: FloatMatrix = vectors / norms
    return out


@runtime_checkable
class EmbeddingModel(Protocol):
    """Bi-encoder for chunk/passage lists and a single query string.

    Callers persist :attr:`model_name` and :attr:`dimension` with the vector
    index. Implementations that support :meth:`index_metadata` return the same
    fields in a mapping.

    Notes
    -----
    When normalization is enabled, implementations should use
    :func:`l2_normalize_rows` so cosine retrieval stays consistent across
    backends.
    """

    @property
    def model_name(self) -> str:
        """Hugging Face hub id or logical name (for example fakes).

        Returns
        -------
        str
            Stable identifier stored in index metadata.
        """
        ...

    @property
    def dimension(self) -> int:
        """Embedding width for each row of :meth:`embed_documents`.

        Returns
        -------
        int
            Number of components per vector.
        """
        ...

    def embed_documents(self, texts: list[str]) -> FloatMatrix:
        """Embed multiple passages in batch order.

        Parameters
        ----------
        texts : list of str
            Chunk or passage strings.

        Returns
        -------
        ndarray of shape (n, dimension)
            One row per input string. Rows are L2-normalized when the
            implementation's ``normalize`` flag is true.
        """
        ...

    def embed_query(self, text: str) -> NDArray[np.float64]:
        """Embed a single user query.

        Parameters
        ----------
        text : str
            Natural-language query.

        Returns
        -------
        ndarray of shape (dimension,)
            Query vector, L2-normalized when normalization is enabled.
        """
        ...
