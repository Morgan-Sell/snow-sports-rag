from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .chroma_store import ChromaVectorStore
from .protocol import VectorStore

__all__ = ["vector_store_from_config"]

__doc__ = """
Construct a ``VectorStore`` from the ``vector_store`` config subsection.
"""


def vector_store_from_config(vector_store: Mapping[str, Any]) -> VectorStore:
    """Build a vector store from the ``vector_store`` config subsection.

    Parameters
    ----------
    vector_store : Mapping[str, Any]
        Typically :attr:`~snow_sports_rag.config.loader.AppConfig.vector_store`.
        ``persist_directory`` should already be absolute (see
        :func:`~snow_sports_rag.config.loader.load_config`).

        Keys:

        - ``backend`` : str — ``chroma`` (default).
        - ``persist_directory`` : pathlib.Path or str — Chroma root directory.
        - ``collection_name`` : str — collection id (default ``snow_sports_kb``).

    Returns
    -------
    VectorStore
        Concrete store implementing the protocol.

    Raises
    ------
    ValueError
        If ``backend`` is unknown or ``collection_name`` / path is invalid.
    """
    raw_backend = str(vector_store.get("backend", "chroma")).strip().lower()
    name = raw_backend.replace("-", "_")
    if name != "chroma":
        msg = f"Unknown vector_store backend: {raw_backend!r}"
        raise ValueError(msg)

    pd_raw = vector_store.get("persist_directory")
    if pd_raw is None:
        msg = "vector_store.persist_directory is required"
        raise ValueError(msg)
    path = pd_raw if isinstance(pd_raw, Path) else Path(str(pd_raw))

    coll = vector_store.get("collection_name", "snow_sports_kb")
    if not isinstance(coll, str) or len(coll.strip()) < 3:
        msg = "vector_store.collection_name must be a string of length >= 3"
        raise ValueError(msg)

    return ChromaVectorStore(path.resolve(), coll.strip())
