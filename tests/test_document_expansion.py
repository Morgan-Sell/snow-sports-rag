from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import yaml

from snow_sports_rag.config import load_config
from snow_sports_rag.pipeline import RAGPipeline
from snow_sports_rag.retrieval.document_expansion import (
    expand_retrieval_hits,
    plan_expansion_requests,
)
from snow_sports_rag.retrieval.models import RetrievalHit
from snow_sports_rag.vectorstore.models import VectorQueryHit


def _hit(
    doc_id: str,
    idx: int,
    *,
    section: str,
    text: str = "body",
    similarity: float = 0.8,
) -> RetrievalHit:
    return RetrievalHit(
        chunk_id=f"{doc_id}::{idx}",
        text=text,
        doc_id=doc_id,
        section_path=section,
        chunk_index=idx,
        similarity=similarity,
        distance=1.0 - similarity,
    )


def _vector_hit(doc_id: str, idx: int, *, section: str, text: str) -> VectorQueryHit:
    return VectorQueryHit(
        id=f"{doc_id}::{idx}",
        document=text,
        distance=float("inf"),
        metadata={
            "doc_id": doc_id,
            "entity_type": doc_id.split("/", 1)[0],
            "section_path": section,
            "chunk_index": idx,
        },
    )


class _DocStore:
    def __init__(self, rows: list[VectorQueryHit]) -> None:
        self.rows = rows
        self.calls: list[str] = []

    def get_by_doc_id(self, doc_id: str) -> list[VectorQueryHit]:
        self.calls.append(doc_id)
        return [r for r in self.rows if r.metadata["doc_id"] == doc_id]


def test_route_planner_emits_neighbor_section_and_anchor_requests() -> None:
    seed = _hit("athletes/Red Gerard.md", 2, section="Career Progression")

    requests = plan_expansion_requests(
        seed,
        "What is the home resort for this Olympic medalist?",
        modes=("neighbors", "same_section", "anchor_sections"),
        window=1,
        anchor_sections_by_entity_type={"athletes": ["Summary"]},
    )

    assert [r.mode for r in requests] == [
        "neighbors",
        "same_section",
        "anchor_sections",
    ]
    assert requests[-1].section_paths == ("Summary",)


def test_expand_retrieval_hits_adds_anchor_summary_for_field_query() -> None:
    doc_id = "athletes/Red Gerard.md"
    seed = _hit(doc_id, 2, section="Career Progression", text="Olympic results")
    store = _DocStore(
        [
            _vector_hit(
                doc_id,
                0,
                section="Summary",
                text="Home Resort: Copper Mountain. Olympic medals: one gold.",
            ),
            _vector_hit(doc_id, 1, section="Career Progression", text="Early career"),
            _vector_hit(
                doc_id,
                2,
                section="Career Progression",
                text="Olympic results",
            ),
            _vector_hit(doc_id, 3, section="Career Progression", text="Later career"),
        ]
    )

    expanded, added, requests = expand_retrieval_hits(
        [seed],
        query="Which home resort belongs to the athlete with an Olympic medal?",
        store=store,
        config={
            "enabled": True,
            "modes": ["neighbors", "same_section", "anchor_sections"],
            "window": 1,
            "max_seed_hits": 4,
            "max_extra_chunks_per_doc": 3,
            "max_total_extra_chunks": 3,
            "expansion_score_penalty": 0.05,
            "anchor_sections_by_entity_type": {"athletes": ["Summary"]},
        },
    )

    assert store.calls == [doc_id]
    assert any(r.mode == "anchor_sections" for r in requests)
    assert any("Home Resort: Copper Mountain" in h.text for h in added)
    assert len(added) == 3
    assert [h.chunk_id for h in expanded][:2] == [seed.chunk_id, f"{doc_id}::1"]
    assert all(h.similarity == seed.similarity - 0.05 for h in added)


def test_expand_retrieval_hits_caps_total_and_per_doc() -> None:
    doc_id = "athletes/a.md"
    seed = _hit(doc_id, 2, section="Career")
    store = _DocStore(
        [
            _vector_hit(doc_id, 0, section="Summary", text="home resort"),
            _vector_hit(doc_id, 1, section="Career", text="left"),
            _vector_hit(doc_id, 2, section="Career", text="seed"),
            _vector_hit(doc_id, 3, section="Career", text="right"),
        ]
    )

    _, added, _ = expand_retrieval_hits(
        [seed],
        query="home resort",
        store=store,
        config={
            "enabled": True,
            "modes": ["neighbors", "same_section", "anchor_sections"],
            "window": 1,
            "max_seed_hits": 1,
            "max_extra_chunks_per_doc": 1,
            "max_total_extra_chunks": 10,
            "anchor_sections_by_entity_type": {"athletes": ["Summary"]},
        },
    )

    assert len(added) == 1


def test_pipeline_expands_wrong_red_gerard_chunk_before_rerank(tmp_path: Path) -> None:
    cfg_path = tmp_path / "cfg.yaml"
    (tmp_path / "kb").mkdir()
    cfg_path.write_text(
        yaml.dump(
            {
                "knowledge_base_path": "kb",
                "embedding": {"backend": "fake", "model_name": "fake", "dimension": 8},
                "generation": {"enabled": False},
                "query_expansion": {"enabled": False},
                "rerank": {"enabled": False},
                "vector_store": {
                    "backend": "chroma",
                    "persist_directory": str(tmp_path / "db"),
                    "collection_name": "snow_sports_kb",
                    "l1_collection_name": "snow_sports_kb_l1",
                },
            }
        ),
        encoding="utf-8",
    )
    cfg = load_config(cfg_path, base_dir=tmp_path)
    cfg = replace(
        cfg,
        document_expansion={
            **dict(cfg.document_expansion),
            "max_extra_chunks_per_doc": 2,
            "max_total_extra_chunks": 2,
        },
    )
    doc_id = "athletes/Red Gerard.md"
    seed = _hit(doc_id, 2, section="Career Progression", text="Olympic gold result")
    pipe = RAGPipeline(cfg)
    pipe._embedder = object()
    pipe._l1_store = object()
    pipe._l2_store = _DocStore(
        [
            _vector_hit(
                doc_id,
                0,
                section="Summary",
                text="Home Resort: Copper Mountain. Olympic medals: one gold.",
            ),
            _vector_hit(doc_id, 1, section="Career Progression", text="Early career"),
            _vector_hit(doc_id, 2, section="Career Progression", text=seed.text),
        ]
    )
    pipe._index_empty = False
    pipe._retrieve_variant = lambda _query, _preset: ([seed], [doc_id])  # type: ignore[method-assign]

    result = pipe.run(
        "What is the home resort for the athlete with an Olympic gold medal?",
        rerank_enabled=False,
        expansion_enabled=False,
        generation_enabled=False,
    )

    assert any(
        "Home Resort: Copper Mountain" in h.text for h in result.trace.l2_pre_rerank
    )
    assert any(
        "Home Resort: Copper Mountain" in h.text
        for h in result.trace.document_expansion_added
    )
    assert len(result.cards) >= 2
