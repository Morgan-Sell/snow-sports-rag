from __future__ import annotations

from collections import Counter

from ..embedding.model import EmbeddingModel
from ..vectorstore import VectorStore
from ..vectorstore.models import VectorQueryHit
from .manifest import ManifestReadableStore, validate_embedder_against_manifest
from .models import RetrievalHit
from .scoring import chroma_cosine_distance_to_similarity

__all__ = ["HierarchicalRetriever"]

__doc__ = """Phase 2.1 two-stage L1 (doc summary) then L2 (chunk) retrieval."""


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


def _dedupe_max_per_doc(
    hits: list[RetrievalHit],
    *,
    top_k: int,
    max_chunks_per_doc: int,
) -> list[RetrievalHit]:
    """Keep global rank order; cap how many passages per ``doc_id``."""
    out: list[RetrievalHit] = []
    counts: Counter[str] = Counter()
    for h in hits:
        if len(out) >= top_k:
            break
        if counts[h.doc_id] >= max_chunks_per_doc:
            continue
        out.append(h)
        counts[h.doc_id] += 1
    return out


def _merge_global_fallback(
    primary: list[RetrievalHit],
    global_hits: list[RetrievalHit],
    *,
    top_k: int,
    max_chunks_per_doc: int,
) -> list[RetrievalHit]:
    """Append globally retrieved passages without duplicating ``chunk_id``."""
    result = list(primary)
    seen = {h.chunk_id for h in result}
    counts = Counter(h.doc_id for h in result)
    for h in global_hits:
        if len(result) >= top_k:
            break
        if h.chunk_id in seen:
            continue
        if counts[h.doc_id] >= max_chunks_per_doc:
            continue
        result.append(h)
        seen.add(h.chunk_id)
        counts[h.doc_id] += 1
    return result


class HierarchicalRetriever:
    """L1 shortlist of documents, then L2 chunk search with light deduplication.

    Parameters
    ----------
    embedder : EmbeddingModel
        Same bi-encoder used when building L1 and L2 rows.
    l2_store : VectorStore
        Section-level chunk index (Phase 1.1 / 1.3).
    l1_store : VectorStore
        One row per document (L1 summaries).
    top_k : int, default 8
        Target number of L2 hits after deduplication.
    l1_top_m : int, default 5
        Number of documents to keep from L1 dense search.
    max_chunks_per_doc : int, default 2
        At most this many L2 chunks per ``doc_id`` in the returned list.
    l2_prefetch_k : int, optional
        Neighbors requested from Chroma before deduplication. If ``None``,
        uses ``max(32, top_k * 8)``.
    global_fallback : bool, default True
        If fewer than ``top_k`` hits remain after the filtered L2 pass (and
        dedupe), query L2 without a ``doc_id`` filter and merge.

    Notes
    -----
    Requires a Chroma backend whose :meth:`~ChromaVectorStore.query` accepts
    a ``where`` filter (``doc_id`` in shortlist) for the filtered L2 step.
    """

    def __init__(
        self,
        embedder: EmbeddingModel,
        l2_store: VectorStore,
        l1_store: VectorStore,
        *,
        top_k: int = 8,
        l1_top_m: int = 5,
        max_chunks_per_doc: int = 2,
        l2_prefetch_k: int | None = None,
        global_fallback: bool = True,
        validate_manifest: bool = True,
    ) -> None:
        self._embedder = embedder
        self._l2 = l2_store
        self._l1 = l1_store
        self._top_k = max(1, int(top_k))
        self._l1_top_m = max(1, int(l1_top_m))
        self._max_chunks_per_doc = max(1, int(max_chunks_per_doc))
        if l2_prefetch_k is None:
            self._l2_prefetch_k = max(32, self._top_k * 8)
        else:
            self._l2_prefetch_k = max(1, int(l2_prefetch_k))
        self._global_fallback = bool(global_fallback)
        self._validate_manifest = validate_manifest

    def _maybe_validate_manifest(self) -> None:
        if not self._validate_manifest:
            return
        if not isinstance(self._l2, ManifestReadableStore):
            return
        manifest = self._l2.read_embedding_manifest()
        validate_embedder_against_manifest(self._embedder, manifest)

    def retrieve(self, query: str, *, k: int | None = None) -> list[RetrievalHit]:
        """Run L1 → filtered L2 → optional global L2, with per-doc chunk caps.

        Parameters
        ----------
        query : str
            User question.
        k : int or None, optional
            Overrides ``top_k`` from construction when set.

        Returns
        -------
        list of RetrievalHit
            L2 passages in best-first order with at most ``max_chunks_per_doc``
            rows per ``doc_id`` (unless ``k`` / ``top_k`` is smaller).
        """
        self._maybe_validate_manifest()
        top = self._top_k if k is None else max(1, int(k))
        q_emb = self._embedder.embed_query(query)

        l1_raw = self._l1.query(query_embedding=q_emb, k=self._l1_top_m)
        shortlist: list[str] = []
        seen_docs: set[str] = set()
        for hit in l1_raw.hits:
            did = str(hit.metadata.get("doc_id", ""))
            if did and did not in seen_docs:
                shortlist.append(did)
                seen_docs.add(did)

        if not shortlist:
            global_raw = self._l2.query(
                query_embedding=q_emb,
                k=self._l2_prefetch_k,
                where=None,
            )
            ranked = [_vector_hit_to_retrieval(h) for h in global_raw.hits]
            return _dedupe_max_per_doc(
                ranked,
                top_k=top,
                max_chunks_per_doc=self._max_chunks_per_doc,
            )[:top]

        filtered = self._l2.query(
            query_embedding=q_emb,
            k=self._l2_prefetch_k,
            where={"doc_id": {"$in": shortlist}},
        )
        ranked = [_vector_hit_to_retrieval(h) for h in filtered.hits]
        primary = _dedupe_max_per_doc(
            ranked,
            top_k=top,
            max_chunks_per_doc=self._max_chunks_per_doc,
        )

        if len(primary) >= top or not self._global_fallback:
            return primary[:top]

        global_raw = self._l2.query(
            query_embedding=q_emb,
            k=self._l2_prefetch_k,
            where=None,
        )
        global_ranked = [_vector_hit_to_retrieval(h) for h in global_raw.hits]
        merged = _merge_global_fallback(
            primary,
            global_ranked,
            top_k=top,
            max_chunks_per_doc=self._max_chunks_per_doc,
        )
        return merged[:top]
