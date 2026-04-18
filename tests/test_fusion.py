from __future__ import annotations

import pytest
from snow_sports_rag.retrieval import (
    fuse_retrieval_hits_max_score,
    fuse_retrieval_hits_rrf,
)
from snow_sports_rag.retrieval.models import RetrievalHit


def _hit(chunk_id: str, sim: float) -> RetrievalHit:
    return RetrievalHit(
        chunk_id=chunk_id,
        text="t",
        doc_id="d",
        section_path="",
        chunk_index=0,
        similarity=sim,
        distance=1.0 - sim,
    )


def test_fuse_max_score_picks_best_similarity_per_chunk() -> None:
    a = [
        _hit("c1", 0.5),
        _hit("c2", 0.4),
    ]
    b = [
        _hit("c1", 0.9),
        _hit("c3", 0.1),
    ]
    out = fuse_retrieval_hits_max_score([a, b], top_n=10)
    by_id = {h.chunk_id: h for h in out}
    assert by_id["c1"].similarity == pytest.approx(0.9)
    assert set(by_id) == {"c1", "c2", "c3"}


def test_fuse_max_score_top_n() -> None:
    lists = [[_hit("a", 0.1), _hit("b", 0.2)], [_hit("c", 0.99)]]
    out = fuse_retrieval_hits_max_score(lists, top_n=2)
    assert len(out) == 2
    assert out[0].similarity >= out[1].similarity


def test_fuse_rrf_orders_by_fused_score() -> None:
    # c2 ranks high in both lists -> strong RRF
    q1 = [_hit("c1", 0.5), _hit("c2", 0.4)]
    q2 = [_hit("c2", 0.3), _hit("c1", 0.2)]
    out = fuse_retrieval_hits_rrf([q1, q2], top_n=2, rrf_k=60)
    assert len(out) == 2
    assert {out[0].chunk_id, out[1].chunk_id} == {"c1", "c2"}
