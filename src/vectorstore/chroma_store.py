from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import chromadb
import numpy as np
from numpy.typing import NDArray

from .models import VectorQueryHit, VectorQueryResult
from .protocol import FloatMatrix

__all__ = ["ChromaVectorStore"]

__doc__ = """
Persistent vector index using ChromaDB (``VectorStore`` protocol).
"""

EMBEDDING_MANIFEST_FILENAME = "embedding_manifest.json"


class ChromaVectorStore:
    """Persistent vector index using ``chromadb.PersistentClient``.

    Uses cosine distance in embedding space (``hnsw:space`` = ``cosine``),
    which matches L2-normalized sentence-embedding workflows.

    Parameters
    ----------
    persist_directory : pathlib.Path
        Root directory for Chroma persistence (created if missing).
    collection_name : str
        Collection name; must satisfy Chroma rules (length 3--512, characters
        ``[a-zA-Z0-9._-]``).

    Raises
    ------
    ValueError
        If ``collection_name`` is too short for Chroma.
    """

    def __init__(
        self,
        persist_directory: Path,
        collection_name: str,
    ) -> None:
        name = str(collection_name).strip()
        if len(name) < 3:
            msg = "collection_name must be at least 3 characters for Chroma, "
            msg += f"got {name!r}"
            raise ValueError(msg)
        self._persist_directory = Path(persist_directory)
        self._persist_directory.mkdir(parents=True, exist_ok=True)
        self._collection_name = name
        self._client = chromadb.PersistentClient(path=str(self._persist_directory))
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    @property
    def persist_directory(self) -> Path:
        """Root path passed at construction."""
        return self._persist_directory

    @property
    def collection_name(self) -> str:
        """Chroma collection name."""
        return self._collection_name

    def reset(self) -> None:
        """Delete this collection (if it exists) and recreate it empty."""
        try:
            self._client.delete_collection(self._collection_name)
        except chromadb.errors.NotFoundError:
            pass
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def write_embedding_manifest(self, model_name: str, dimension: int) -> None:
        """Write :data:`EMBEDDING_MANIFEST_FILENAME` next to Chroma files.

        Parameters
        ----------
        model_name : str
            Bi-encoder id or logical name (matches Phase 1.2 metadata).
        dimension : int
            Vector width for the indexed embeddings.
        """
        path = self._persist_directory / EMBEDDING_MANIFEST_FILENAME
        payload = {"model_name": model_name, "dimension": int(dimension)}
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def read_embedding_manifest(self) -> dict[str, Any] | None:
        """Load :data:`EMBEDDING_MANIFEST_FILENAME` if present.

        Returns
        -------
        dict or None
            Parsed JSON, or ``None`` if the file does not exist.
        """
        path = self._persist_directory / EMBEDDING_MANIFEST_FILENAME
        if not path.is_file():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def upsert(
        self,
        *,
        ids: list[str],
        embeddings: FloatMatrix,
        documents: list[str],
        metadatas: list[Mapping[str, str | int | float | bool]],
    ) -> None:
        """Insert or replace embeddings in the Chroma collection.

        Parameters
        ----------
        ids : list of str
            Unique row ids.
        embeddings : ndarray of shape (n, dim)
            Batch matrix co-aligned with ``ids``.
        documents : list of str
            Passage text per row.
        metadatas : list of mapping
            Chroma-compatible metadata (non-empty dicts per row).

        Raises
        ------
        ValueError
            If ``ids``, ``documents``, ``metadatas``, and embedding row count
            disagree. A zero-length batch is a no-op.
        """
        n = len(ids)
        if n != len(documents) or n != len(metadatas):
            msg = "ids, documents, and metadatas must have the same length"
            raise ValueError(msg)
        if embeddings.ndim != 2 or embeddings.shape[0] != n:
            msg = (
                f"embeddings must have shape (n, dim) with n={n}, "
                f"got {embeddings.shape}"
            )
            raise ValueError(msg)
        if n == 0:
            return
        emb_list = np.asarray(embeddings, dtype=np.float64).tolist()
        meta_out = [dict(m) for m in metadatas]
        self._collection.upsert(
            ids=ids,
            embeddings=emb_list,
            documents=documents,
            metadatas=meta_out,
        )

    def query(
        self,
        *,
        query_embedding: NDArray[np.float64],
        k: int,
        where: Mapping[str, Any] | None = None,
    ) -> VectorQueryResult:
        """Query nearest neighbors by embedding.

        Parameters
        ----------
        query_embedding : ndarray of shape (dim,)
            Single query vector.
        k : int
            Requested hit count; internally at least ``1`` for the Chroma call,
            but an empty collection yields no hits.
        where : mapping or None, optional
            Chroma ``where`` metadata filter (e.g. ``doc_id`` with ``$in``).

        Returns
        -------
        VectorQueryResult
            Hits ordered by Chroma (ascending distance).
        """
        k_eff = max(1, int(k))
        vec = np.asarray(query_embedding, dtype=np.float64).reshape(-1)
        q = vec.tolist()
        query_kwargs: dict[str, Any] = {
            "query_embeddings": [q],
            "n_results": k_eff,
            "include": ["documents", "metadatas", "distances"],
        }
        if where is not None:
            query_kwargs["where"] = dict(where)
        raw = self._collection.query(**query_kwargs)
        ids_batch = raw.get("ids") or [[]]
        docs_batch = raw.get("documents") or [[]]
        meta_batch = raw.get("metadatas") or [[]]
        dist_batch = raw.get("distances") or [[]]
        ids_l = ids_batch[0] if ids_batch else []
        docs_l = docs_batch[0] if docs_batch else []
        metas_l = meta_batch[0] if meta_batch else []
        dists_l = dist_batch[0] if dist_batch else []
        hits: list[VectorQueryHit] = []
        for i, eid in enumerate(ids_l):
            doc = docs_l[i] if i < len(docs_l) and docs_l[i] is not None else ""
            meta = metas_l[i] if i < len(metas_l) and metas_l[i] is not None else {}
            dist = float(dists_l[i]) if i < len(dists_l) else float("inf")
            hits.append(
                VectorQueryHit(
                    id=str(eid),
                    document=str(doc),
                    distance=dist,
                    metadata=meta,
                )
            )
        return VectorQueryResult(hits=hits)
