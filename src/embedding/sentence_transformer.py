from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from .model import FloatMatrix, l2_normalize_rows

__all__ = ["SentenceTransformerEmbeddingModel"]

__doc__ = """Bi-encoder via ``sentence_transformers``.

Implements :class:`~snow_sports_rag.embedding.model.EmbeddingModel` using Hub
checkpoints.
"""


class SentenceTransformerEmbeddingModel:
    """Bi-encoder using ``sentence_transformers.SentenceTransformer``.

    ``model_name`` should be a Hugging Face hub identifier, for example
    ``sentence-transformers/all-MiniLM-L6-v2`` or
    ``sentence-transformers/all-mpnet-base-v2``.

    Parameters
    ----------
    model_name : str
        Hub id passed to :class:`sentence_transformers.SentenceTransformer`.
    device : str or None, optional
        Device string forwarded to the underlying model (for example ``'cuda'``
        or ``'cpu'``). ``None`` lets the library choose.
    normalize : bool, default True
        If true, L2-normalize outputs with
        :func:`~snow_sports_rag.embedding.model.l2_normalize_rows` after
        encoding (the encoder itself is called with ``normalize_embeddings=False``
        so normalization stays centralized).

    Notes
    -----
    First construction may download weights from the Hugging Face Hub.
    """

    def __init__(
        self,
        model_name: str,
        *,
        device: str | None = None,
        normalize: bool = True,
    ) -> None:
        from sentence_transformers import SentenceTransformer

        self._model_name = model_name
        self._normalize = normalize
        self._model = SentenceTransformer(model_name, device=device)

    @property
    def model_name(self) -> str:
        """Hub id supplied at construction.

        Returns
        -------
        str
            Same string as the constructor ``model_name`` argument.
        """
        return self._model_name

    @property
    def dimension(self) -> int:
        """Embedding size reported by the loaded transformer.

        Returns
        -------
        int
            Sentence embedding width for the loaded checkpoint.

        Notes
        -----
        Uses :meth:`sentence_transformers.SentenceTransformer.get_embedding_dimension`.
        """
        return int(self._model.get_embedding_dimension())

    def embed_documents(self, texts: list[str]) -> FloatMatrix:
        """Encode passages with the underlying bi-encoder.

        Parameters
        ----------
        texts : list of str
            Batch of chunk or document strings. Empty input returns a
            ``(0, dimension)`` float array.

        Returns
        -------
        ndarray of shape (len(texts), dimension)
            Float64 rows, optionally L2-normalized.
        """
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float64)
        raw = self._model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=False,
        )
        vectors = np.asarray(raw, dtype=np.float64)
        if self._normalize:
            vectors = l2_normalize_rows(vectors)
        return vectors

    def embed_query(self, text: str) -> NDArray[np.float64]:
        """Encode a single query.

        Parameters
        ----------
        text : str
            User query string.

        Returns
        -------
        ndarray of shape (dimension,)
            Query embedding, same normalization policy as :meth:`embed_documents`.
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
