from __future__ import annotations

from typing import Any, Mapping

from .fake import FakeEmbeddingModel
from .model import EmbeddingModel
from .sentence_transformer import SentenceTransformerEmbeddingModel

__all__ = ["embedding_model_from_config"]

__doc__ = """Config-driven construction of embedding models.

Wraps :class:`~snow_sports_rag.embedding.model.EmbeddingModel` factories.
"""


def embedding_model_from_config(embedding: Mapping[str, Any]) -> EmbeddingModel:
    """Build a concrete embedder from the ``embedding`` config subsection.

    Parameters
    ----------
    embedding : Mapping[str, Any]
        Typically :attr:`~snow_sports_rag.config.loader.AppConfig.embedding`.

        Recognized keys:

        - ``backend`` : str — ``sentence_transformers`` (default), ``st``,
          ``huggingface``, or ``fake``.
        - ``model_name`` : str — Hugging Face hub id for sentence-transformers
          backends, or logical name for ``fake``.
        - ``normalize`` : bool — whether to L2-normalize outputs (default
          ``True``).
        - ``device`` : str or None — optional device hint for
          ``SentenceTransformer``.
        - ``dimension`` : int — required when ``backend`` is ``fake``.

    Returns
    -------
    EmbeddingModel
        Instance implementing :class:`~snow_sports_rag.embedding.model.EmbeddingModel`.

    Raises
    ------
    ValueError
        If ``backend`` is unknown, ``model_name`` is missing or blank for
        sentence-transformers, or ``dimension`` is missing for ``fake``.
    TypeError
        If ``normalize`` is provided but not a bool, or ``model_name`` is not a
        string for ``fake``.
    """
    raw_backend = str(embedding.get("backend", "sentence_transformers")).strip().lower()
    name = raw_backend.replace("-", "_")
    normalize = embedding.get("normalize", True)
    if not isinstance(normalize, bool):
        msg = "embedding.normalize must be a bool when provided"
        raise TypeError(msg)

    if name in ("sentence_transformers", "st", "huggingface"):
        model_name = embedding.get("model_name")
        if not isinstance(model_name, str) or not model_name.strip():
            msg = (
                "embedding.model_name must be a non-empty string "
                "for sentence_transformers"
            )
            raise ValueError(msg)
        device_raw = embedding.get("device")
        device = None if device_raw in (None, "") else str(device_raw)
        return SentenceTransformerEmbeddingModel(
            model_name.strip(),
            device=device,
            normalize=normalize,
        )

    if name == "fake":
        dim_raw = embedding.get("dimension")
        if dim_raw is None:
            msg = "embedding.dimension is required for backend 'fake'"
            raise ValueError(msg)
        dimension = int(dim_raw)
        model_name = embedding.get("model_name", "fake-deterministic")
        if not isinstance(model_name, str):
            msg = "embedding.model_name must be a string when provided"
            raise TypeError(msg)
        return FakeEmbeddingModel(
            dimension=dimension,
            model_name=model_name,
            normalize=normalize,
        )

    msg = f"Unknown embedding backend: {raw_backend!r}"
    raise ValueError(msg)
