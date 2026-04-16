from __future__ import annotations

import hashlib
import struct
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .model import FloatMatrix, l2_normalize_rows

__all__ = ["FakeEmbeddingModel"]

__doc__ = """Deterministic embedding backend for unit tests and offline pipelines."""


def _text_to_raw_vector(text: str, dim: int) -> NDArray[np.float64]:
    """Build a fixed-length vector from ``text`` using repeated SHA-256 blocks.

    Parameters
    ----------
    text : str
        Input string; encoded as UTF-8 before hashing.
    dim : int
        Required vector length; must be positive.

    Returns
    -------
    ndarray of shape (dim,)
        Raw components in roughly ``[-1, 1]`` before optional L2 normalization.

    Raises
    ------
    ValueError
        If ``dim`` is not positive.
    """
    if dim <= 0:
        msg = f"dimension must be positive, got {dim}"
        raise ValueError(msg)
    seed = hashlib.sha256(text.encode("utf-8")).digest()
    values: list[float] = []
    counter = 0
    while len(values) < dim:
        block = hashlib.sha256(seed + counter.to_bytes(4, "big")).digest()
        counter += 1
        for i in range(0, len(block) - 3, 4):
            if len(values) >= dim:
                break
            (x,) = struct.unpack(">i", block[i : i + 4])
            values.append((x % 20001) / 10000.0 - 1.0)
    return np.asarray(values, dtype=np.float64)


class FakeEmbeddingModel:
    """Deterministic embedder with no neural network dependency.

    Vectors are derived from SHA-256 digests of the input text, so the same
    string always yields the same embedding.

    Parameters
    ----------
    dimension : int
        Output vector size; must be positive.
    model_name : str, default 'fake-deterministic'
        Logical name recorded in :meth:`index_metadata`.
    normalize : bool, default True
        If true, apply :func:`~snow_sports_rag.embedding.model.l2_normalize_rows`
        to batch outputs.

    Notes
    -----
    Suitable for integration tests that must not download model weights.
    """

    def __init__(
        self,
        *,
        dimension: int,
        model_name: str = "fake-deterministic",
        normalize: bool = True,
    ) -> None:
        if dimension <= 0:
            msg = f"dimension must be positive, got {dimension}"
            raise ValueError(msg)
        self._dimension = dimension
        self._model_name = model_name
        self._normalize = normalize

    @property
    def model_name(self) -> str:
        """Logical model identifier for metadata.

        Returns
        -------
        str
            Value passed as ``model_name`` at construction.
        """
        return self._model_name

    @property
    def dimension(self) -> int:
        """Embedding dimensionality.

        Returns
        -------
        int
            Fixed width of each embedding vector.
        """
        return self._dimension

    def embed_documents(self, texts: list[str]) -> FloatMatrix:
        """Embed each string as an independent deterministic vector.

        Parameters
        ----------
        texts : list of str
            Passages to embed. An empty list yields a zero-row matrix.

        Returns
        -------
        ndarray of shape (len(texts), dimension)
            Row ``i`` corresponds to ``texts[i]``, L2-normalized per row when
            ``normalize`` is true.
        """
        if not texts:
            return np.zeros((0, self._dimension), dtype=np.float64)
        rows = np.stack([_text_to_raw_vector(t, self._dimension) for t in texts])
        if self._normalize:
            rows = l2_normalize_rows(rows)
        return rows

    def embed_query(self, text: str) -> NDArray[np.float64]:
        """Embed a single query string.

        Parameters
        ----------
        text : str
            Query text.

        Returns
        -------
        ndarray of shape (dimension,)
            Same convention as a one-row :meth:`embed_documents` call.
        """
        q = self.embed_documents([text])
        return q[0]

    def index_metadata(self) -> dict[str, Any]:
        """Metadata to store beside vectors built with this embedder.

        Returns
        -------
        dict
            Keys ``model_name`` and ``dimension`` suitable for index manifests.
        """
        return {"model_name": self.model_name, "dimension": self.dimension}
