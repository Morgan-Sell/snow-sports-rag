from __future__ import annotations

import time
from typing import Any, Mapping

from ..config import AppConfig
from ..embedding import embedding_model_from_config
from ..generation import answer_generator_from_config
from ..llm import llm_client_from_config
from ..rerank import reranker_from_config
from ..retrieval.fusion import (
    fuse_retrieval_hits_max_score,
    fuse_retrieval_hits_rrf,
)
from ..retrieval.hierarchical import (
    _dedupe_max_per_doc,
    _merge_global_fallback,
    _vector_hit_to_retrieval,
)
from ..retrieval.models import RetrievalHit
from ..vectorstore import chroma_l2_l1_stores_from_config
from .models import (
    PipelineResult,
    PipelineTrace,
    SourceCard,
    StageLatency,
)
from .presets import RetrievalPreset, resolve_preset
from .trace import compute_config_hash, new_trace_id

__all__ = ["RAGPipeline"]

__doc__ = """Hierarchical RAG pipeline with per-stage trace capture.

Unlike the thin composition used in :mod:`snow_sports_rag.cli`, this module
inlines the L1 → L2 flow so it can emit the intermediate stage outputs the
Debug UI and trace logger need. It reuses the same helper functions as the
standalone retrievers, so the ranking logic is not duplicated.
"""


def _snippet(text: str, max_chars: int = 320) -> str:
    """Return a UI-friendly short excerpt of a passage body.

    Parameters
    ----------
    text : str
        Raw chunk body as stored in the vector index.
    max_chars : int, optional
        Hard ceiling on the returned string length.

    Returns
    -------
    str
        ``text`` trimmed at whitespace when possible, with ``" …"`` when
        truncation occurred.
    """
    t = text.strip().replace("\n", " ")
    if len(t) <= max_chars:
        return t
    cut = t[: max(0, max_chars - 2)]
    last_space = cut.rfind(" ")
    if last_space > max_chars // 2:
        cut = cut[:last_space]
    return cut.rstrip() + " …"


def _hits_to_cards(
    hits: list[RetrievalHit],
    *,
    pre_rerank_lookup: dict[str, float] | None = None,
) -> list[SourceCard]:
    """Number ``hits`` into :class:`SourceCard` rows for the Sources panel.

    Parameters
    ----------
    hits : list of RetrievalHit
        Final post-rerank list to show to the user.
    pre_rerank_lookup : dict[str, float] or None, optional
        Map from ``chunk_id`` to the similarity it had before the reranker
        ran. Chunks not present in the map get ``pre_rerank_similarity=None``.

    Returns
    -------
    list of SourceCard
        1-indexed cards with truncated snippets and (optional) pre-rerank
        scores for the "Why this source?" tooltip.
    """
    lookup = pre_rerank_lookup or {}
    out: list[SourceCard] = []
    for i, h in enumerate(hits, start=1):
        out.append(
            SourceCard(
                index=i,
                chunk_id=h.chunk_id,
                doc_id=h.doc_id,
                section_path=h.section_path,
                chunk_index=h.chunk_index,
                similarity=float(h.similarity),
                snippet=_snippet(h.text),
                pre_rerank_similarity=lookup.get(h.chunk_id),
            )
        )
    return out


class RAGPipeline:
    """Run hierarchical retrieval + optional expansion / rerank / generation.

    Stage outputs are collected into a :class:`PipelineTrace` so the Gradio
    Debug panel and the JSONL trace logger can present them without the UI
    needing to know about any of the retrieval internals.

    Parameters
    ----------
    cfg : AppConfig
        Full application configuration; models and backends are built lazily
        (see :meth:`_ensure_loaded`) so import-time cost stays low.
    """

    def __init__(self, cfg: AppConfig) -> None:
        """Store the config and pre-compute the :func:`compute_config_hash` value.

        Parameters
        ----------
        cfg : AppConfig
            Parsed configuration from :func:`snow_sports_rag.load_config`.
        """
        self._cfg = cfg
        self._config_hash = compute_config_hash(cfg)
        self._embedder: Any | None = None
        self._l2_store: Any | None = None
        self._l1_store: Any | None = None

    @property
    def config_hash(self) -> str:
        """Expose the stable 12-hex-char config fingerprint.

        Returns
        -------
        str
            Same value as :func:`compute_config_hash(cfg)` from construction.
        """
        return self._config_hash

    @property
    def cfg(self) -> AppConfig:
        """Return the configuration object the pipeline was built from.

        Returns
        -------
        AppConfig
            The exact instance passed to :meth:`__init__`.
        """
        return self._cfg

    def _ensure_loaded(self) -> None:
        """Lazily build the embedder and vector stores on first :meth:`run`.

        Notes
        -----
        Loading is deferred so that constructing a pipeline (and rendering the
        Settings panel) does not force the embedding model into memory.
        """
        if self._embedder is None:
            self._embedder = embedding_model_from_config(self._cfg.embedding)
        if self._l2_store is None or self._l1_store is None:
            l2, l1 = chroma_l2_l1_stores_from_config(self._cfg.vector_store)
            self._l2_store = l2
            self._l1_store = l1

    def _retrieve_variant(
        self,
        query: str,
        preset: RetrievalPreset,
    ) -> tuple[list[RetrievalHit], list[str]]:
        """Execute one hierarchical retrieval pass for a single query variant.

        Parameters
        ----------
        query : str
            One of the query variants (original or paraphrase).
        preset : RetrievalPreset
            Active preset; drives ``top_k``, ``l1_top_m``, and the
            ``max_chunks_per_doc`` dedupe cap.

        Returns
        -------
        tuple
            ``(hits, shortlist)``:

            - ``hits``: deduplicated L2 hits, with global fallback applied
              when ``retrieval.hierarchical_global_fallback`` is set and the
              filtered pass came up short.
            - ``shortlist``: L1 doc_ids that were used to filter L2 (empty
              when L1 returned nothing).
        """
        assert self._embedder is not None
        assert self._l1_store is not None
        assert self._l2_store is not None
        global_fb = bool(self._cfg.retrieval.get("hierarchical_global_fallback", True))
        prefetch_k = max(32, preset.top_k * 8)

        q_emb = self._embedder.embed_query(query)
        l1_raw = self._l1_store.query(query_embedding=q_emb, k=preset.l1_top_m)
        shortlist: list[str] = []
        seen_docs: set[str] = set()
        for h in l1_raw.hits:
            did = str(h.metadata.get("doc_id", ""))
            if did and did not in seen_docs:
                shortlist.append(did)
                seen_docs.add(did)

        if not shortlist:
            global_raw = self._l2_store.query(
                query_embedding=q_emb, k=prefetch_k, where=None
            )
            ranked = [_vector_hit_to_retrieval(h) for h in global_raw.hits]
            hits = _dedupe_max_per_doc(
                ranked,
                top_k=preset.top_k,
                max_chunks_per_doc=preset.max_chunks_per_doc,
            )
            return hits, []

        filtered = self._l2_store.query(
            query_embedding=q_emb,
            k=prefetch_k,
            where={"doc_id": {"$in": shortlist}},
        )
        ranked = [_vector_hit_to_retrieval(h) for h in filtered.hits]
        primary = _dedupe_max_per_doc(
            ranked,
            top_k=preset.top_k,
            max_chunks_per_doc=preset.max_chunks_per_doc,
        )
        if len(primary) >= preset.top_k or not global_fb:
            return primary, shortlist

        global_raw = self._l2_store.query(
            query_embedding=q_emb, k=prefetch_k, where=None
        )
        global_ranked = [_vector_hit_to_retrieval(h) for h in global_raw.hits]
        merged = _merge_global_fallback(
            primary,
            global_ranked,
            top_k=preset.top_k,
            max_chunks_per_doc=preset.max_chunks_per_doc,
        )
        return merged, shortlist

    def _fuse(
        self,
        per_variant: list[list[RetrievalHit]],
        *,
        top_n_fused: int,
    ) -> list[RetrievalHit]:
        """Merge per-variant rankings using the configured fusion strategy.

        Parameters
        ----------
        per_variant : list of list of RetrievalHit
            One ranked list per query variant.
        top_n_fused : int
            Upper bound on the returned list length.

        Returns
        -------
        list of RetrievalHit
            Either a single list (when only one variant was retrieved) or the
            fused ranking produced by :func:`fuse_retrieval_hits_max_score` /
            :func:`fuse_retrieval_hits_rrf`.
        """
        if len(per_variant) == 1:
            return per_variant[0][:top_n_fused]
        qe = self._cfg.query_expansion
        mode = str(qe.get("fusion", "max_score")).strip().lower()
        if mode == "rrf":
            return fuse_retrieval_hits_rrf(
                per_variant,
                top_n=top_n_fused,
                rrf_k=int(qe.get("rrf_k", 60)),
            )
        return fuse_retrieval_hits_max_score(per_variant, top_n=top_n_fused)

    def _resolve_toggles(
        self,
        *,
        rerank_enabled: bool | None,
        expansion_enabled: bool | None,
        generation_enabled: bool | None,
    ) -> tuple[bool, bool, bool]:
        """Combine caller overrides with ``cfg`` defaults for the three toggles.

        Parameters
        ----------
        rerank_enabled, expansion_enabled, generation_enabled : bool or None
            ``None`` falls back to the ``enabled`` flag in the corresponding
            config section.

        Returns
        -------
        tuple of bool
            ``(rerank, expansion, generation)`` in that order.
        """
        rr = (
            bool(self._cfg.rerank.get("enabled", False))
            if rerank_enabled is None
            else bool(rerank_enabled)
        )
        qx = (
            bool(self._cfg.query_expansion.get("enabled", False))
            if expansion_enabled is None
            else bool(expansion_enabled)
        )
        gen = (
            bool(self._cfg.generation.get("enabled", False))
            if generation_enabled is None
            else bool(generation_enabled)
        )
        return rr, qx, gen

    def run(
        self,
        query: str,
        *,
        preset: str | RetrievalPreset = "Balanced",
        rerank_enabled: bool | None = None,
        expansion_enabled: bool | None = None,
        generation_enabled: bool | None = None,
    ) -> PipelineResult:
        """Execute the full pipeline and capture per-stage intermediates.

        Parameters
        ----------
        query : str
            User question (stripped before use; empty strings return a no-op
            :class:`PipelineResult`).
        preset : str or RetrievalPreset, optional
            Preset label (``Fast`` / ``Balanced`` / ``Deep``) or a fully
            constructed :class:`RetrievalPreset`.
        rerank_enabled, expansion_enabled, generation_enabled : bool or None
            UI toggle overrides; ``None`` defers to ``cfg``.

        Returns
        -------
        PipelineResult
            Final cards, optional generated answer, debug trace, trace id,
            and config hash.
        """
        t_total = time.perf_counter()
        preset_obj: RetrievalPreset = (
            preset if isinstance(preset, RetrievalPreset) else resolve_preset(preset)
        )
        rr_on, qx_on, gen_on = self._resolve_toggles(
            rerank_enabled=rerank_enabled,
            expansion_enabled=expansion_enabled,
            generation_enabled=generation_enabled,
        )

        q = query.strip()
        if not q:
            return PipelineResult(
                query=query,
                cards=[],
                answer=None,
                trace=PipelineTrace(query=query),
                trace_id=new_trace_id(),
                config_hash=self._config_hash,
            )

        self._ensure_loaded()

        t0 = time.perf_counter()
        expansions: list[str] = []
        if qx_on:
            llm = llm_client_from_config(self._cfg.llm)
            expansions = llm.expand_query(
                q,
                num_paraphrases=int(
                    self._cfg.query_expansion.get("num_paraphrases", 3)
                ),
            )
        expansion_ms = (time.perf_counter() - t0) * 1000.0

        variants: list[str] = [q]
        seen_v = {q}
        for p in expansions:
            s = p.strip()
            if s and s not in seen_v:
                seen_v.add(s)
                variants.append(s)

        t0 = time.perf_counter()
        per_variant_hits: list[list[RetrievalHit]] = []
        all_shortlists: list[str] = []
        seen_shortlist: set[str] = set()
        for v in variants:
            hits, shortlist = self._retrieve_variant(v, preset_obj)
            per_variant_hits.append(hits)
            for did in shortlist:
                if did and did not in seen_shortlist:
                    seen_shortlist.add(did)
                    all_shortlists.append(did)
        l2_pre_rerank = self._fuse(
            per_variant_hits, top_n_fused=preset_obj.top_n_pre_rerank
        )
        retrieval_ms = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        if rr_on:
            reranker = reranker_from_config(
                _preset_override_rerank(self._cfg.rerank, preset_obj),
                llm=self._cfg.llm,
            )
            reranked = reranker.rerank(
                q,
                l2_pre_rerank[: preset_obj.top_n_in],
                top_k=preset_obj.top_k_out,
            )
        else:
            reranked = l2_pre_rerank[: preset_obj.top_k_out]
        rerank_ms = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        answer = None
        if gen_on:
            generator = answer_generator_from_config(
                self._cfg.generation, llm=self._cfg.llm
            )
            answer = generator.generate(q, reranked)
        generation_ms = (time.perf_counter() - t0) * 1000.0

        total_ms = (time.perf_counter() - t_total) * 1000.0
        latency = StageLatency(
            expansion_ms=round(expansion_ms, 2),
            retrieval_ms=round(retrieval_ms, 2),
            rerank_ms=round(rerank_ms, 2),
            generation_ms=round(generation_ms, 2),
            total_ms=round(total_ms, 2),
        )
        trace = PipelineTrace(
            query=q,
            expansions=list(expansions),
            variants=list(variants),
            l1_shortlist=list(all_shortlists),
            l2_pre_rerank=list(l2_pre_rerank),
            reranked=list(reranked),
            latency=latency,
        )
        pre_lookup = {h.chunk_id: float(h.similarity) for h in l2_pre_rerank}
        return PipelineResult(
            query=q,
            cards=_hits_to_cards(reranked, pre_rerank_lookup=pre_lookup),
            answer=answer,
            trace=trace,
            trace_id=new_trace_id(),
            config_hash=self._config_hash,
        )


def _preset_override_rerank(
    rerank_cfg: Mapping[str, Any],
    preset: RetrievalPreset,
) -> dict[str, Any]:
    """Clone ``rerank_cfg`` with ``top_n_in`` / ``top_k_out`` from the preset.

    Parameters
    ----------
    rerank_cfg : Mapping[str, Any]
        The user's rerank configuration (backend, model, etc.).
    preset : RetrievalPreset
        Active preset; its ``top_n_in`` / ``top_k_out`` win.

    Returns
    -------
    dict
        Shallow-copied config where the preset values override the user's
        generic rerank settings. The backend selection is left untouched.
    """
    out = dict(rerank_cfg)
    out["top_n_in"] = preset.top_n_in
    out["top_k_out"] = preset.top_k_out
    return out
