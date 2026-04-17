from __future__ import annotations

from ..chunking import ChunkStrategy
from ..chunking.models import Chunk
from ..embedding.model import EmbeddingModel
from ..ingest.models import SourceDocument
from ..vectorstore import VectorStore, l1_summary_row_id, pack_chunk_upsert
from ..vectorstore.chunks import chunk_to_metadata
from ..vectorstore.models import VectorQueryHit
from .l1_summary import l1_summary_text
from .manifest import ManifestReadableStore, validate_embedder_against_manifest
from .models import RetrievalHit
from .scoring import chroma_cosine_distance_to_similarity

__all__ = ["BaselineRetriever", "IndexBuilder"]

__doc__ = """Phase 1.4 dense baseline: index build and retrieval."""


def _vector_hit_to_retrieval(hit: VectorQueryHit) -> RetrievalHit:
    meta = hit.metadata
    doc_id = str(meta.get("doc_id", ""))
    section_path = str(meta.get("section_path", ""))
    raw_ci = meta.get("chunk_index", -1)
    chunk_index = int(raw_ci) if raw_ci is not None else -1
    sim = chroma_cosine_distance_to_similarity(hit.distance)
    return RetrievalHit(
        chunk_id=hit.id,
        text=hit.document,
        doc_id=doc_id,
        section_path=section_path,
        chunk_index=chunk_index,
        similarity=sim,
        distance=hit.distance,
    )


class BaselineRetriever:
    """Single-stage dense retriever: embed query → vector search → ranked hits.

    Parameters
    ----------
    embedder : EmbeddingModel
        Same bi-encoder used when building the index.
    store : VectorStore
        Populated index (e.g. Chroma).
    top_k : int, default 8
        Default ``k`` when :meth:`retrieve` is called without ``k``.
    validate_manifest : bool, default True
        If true and ``store`` implements :class:`ManifestReadableStore`, load
        the manifest and verify ``model_name`` / ``dimension`` against
        ``embedder`` before each :meth:`retrieve`.

    Notes
    -----
    Does not deduplicate by ``doc_id`` in top-``k`` (Phase 1.4 policy).
    """

    def __init__(
        self,
        embedder: EmbeddingModel,
        store: VectorStore,
        *,
        top_k: int = 8,
        validate_manifest: bool = True,
    ) -> None:
        self._embedder = embedder
        self._store = store
        self._top_k = int(top_k)
        self._validate_manifest = validate_manifest

    def _maybe_validate_manifest(self) -> None:
        if not self._validate_manifest:
            return
        if not isinstance(self._store, ManifestReadableStore):
            return
        manifest = self._store.read_embedding_manifest()
        validate_embedder_against_manifest(self._embedder, manifest)

    def retrieve(self, query: str, *, k: int | None = None) -> list[RetrievalHit]:
        """Dense retrieval for ``query``.

        Parameters
        ----------
        query : str
            Natural-language query.
        k : int or None, optional
            Max hits; defaults to ``top_k`` from construction.

        Returns
        -------
        list of RetrievalHit
            Best-first order by Chroma distance (ascending), with ``similarity``
            derived as ``1.0 - distance``.
        """
        self._maybe_validate_manifest()
        k_eff = self._top_k if k is None else int(k)
        k_eff = max(1, k_eff)
        q_emb = self._embedder.embed_query(query)
        raw = self._store.query(query_embedding=q_emb, k=k_eff)
        return [_vector_hit_to_retrieval(h) for h in raw.hits]


class IndexBuilder:
    """Chunk documents, embed, and upsert into a vector store (full rebuild).

    Calls :meth:`reset` on the store when available (e.g. Chroma), then
    upserts all chunks and writes the embedding manifest when supported.

    When ``l1_store`` is set, also resets and rebuilds one L1 summary embedding
    per document (Phase 2.1) in that collection.

    Parameters
    ----------
    chunk_strategy : ChunkStrategy
        Strategy from §1.1.
    embedder : EmbeddingModel
        Bi-encoder for passages.
    store : VectorStore
        Target L2 chunk index (typically Chroma).
    l1_store : VectorStore or None, optional
        Optional second collection for document-level summaries. When
        ``None``, only L2 chunks are indexed (Phase 1.4 behavior).
    """

    def __init__(
        self,
        chunk_strategy: ChunkStrategy,
        embedder: EmbeddingModel,
        store: VectorStore,
        *,
        l1_store: VectorStore | None = None,
    ) -> None:
        self._chunk_strategy = chunk_strategy
        self._embedder = embedder
        self._store = store
        self._l1_store = l1_store

    def build(self, documents: list[SourceDocument]) -> int:
        """Rebuild the index from ``documents``.

        Parameters
        ----------
        documents : list of SourceDocument
            Corpus to index.

        Returns
        -------
        int
            Number of L2 chunks upserted (L1 rows are not included in this count).
        """
        reset = getattr(self._store, "reset", None)
        if callable(reset):
            reset()
        if self._l1_store is not None:
            reset_l1 = getattr(self._l1_store, "reset", None)
            if callable(reset_l1):
                reset_l1()

        if self._l1_store is not None and documents:
            l1_chunks = [
                Chunk(
                    text=l1_summary_text(doc),
                    doc_id=doc.doc_id,
                    entity_type=doc.entity_type,
                    section_path="__l1_summary__",
                    chunk_index=-1,
                )
                for doc in documents
            ]
            t1 = [c.text for c in l1_chunks]
            e1 = self._embedder.embed_documents(t1)
            ids1 = [l1_summary_row_id(c.doc_id) for c in l1_chunks]
            metas1 = [chunk_to_metadata(c) for c in l1_chunks]
            self._l1_store.upsert(
                ids=ids1,
                embeddings=e1,
                documents=t1,
                metadatas=metas1,
            )

        chunks: list[Chunk] = []
        for doc in documents:
            chunks.extend(self._chunk_strategy.chunk(doc))

        if not chunks:
            writer = getattr(self._store, "write_embedding_manifest", None)
            if callable(writer):
                writer(self._embedder.model_name, self._embedder.dimension)
            return 0

        texts = [c.text for c in chunks]
        embeddings = self._embedder.embed_documents(texts)
        ids, docs, metadatas, mat = pack_chunk_upsert(chunks, embeddings)
        self._store.upsert(
            ids=ids,
            embeddings=mat,
            documents=docs,
            metadatas=metadatas,
        )
        writer = getattr(self._store, "write_embedding_manifest", None)
        if callable(writer):
            writer(self._embedder.model_name, self._embedder.dimension)
        return len(chunks)
