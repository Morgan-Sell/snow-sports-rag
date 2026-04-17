from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .chroma_store import ChromaVectorStore
from .protocol import VectorStore

__all__ = ["chroma_l2_l1_stores_from_config", "vector_store_from_config"]

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


def chroma_l2_l1_stores_from_config(
    vector_store: Mapping[str, Any],
) -> tuple[ChromaVectorStore, ChromaVectorStore]:
    """Build paired Chroma collections: L2 chunks and L1 document summaries.

    Parameters
    ----------
    vector_store : Mapping[str, Any]
        Same subsection as :func:`vector_store_from_config`, plus optional
        ``l1_collection_name`` (default: ``{collection_name}_l1``).

    Returns
    -------
    tuple[ChromaVectorStore, ChromaVectorStore]
        ``(l2_store, l1_store)`` sharing ``persist_directory``.
    """
    l2 = vector_store_from_config(vector_store)
    if not isinstance(l2, ChromaVectorStore):
        msg = "chroma_l2_l1_stores_from_config requires chroma backend"
        raise TypeError(msg)
    raw_l1 = vector_store.get("l1_collection_name")
    base = str(vector_store.get("collection_name", "snow_sports_kb")).strip()
    if isinstance(raw_l1, str) and len(raw_l1.strip()) >= 3:
        l1_name = raw_l1.strip()
    else:
        l1_name = f"{base}_l1"
    l1 = ChromaVectorStore(l2.persist_directory, l1_name)
    return l2, l1
