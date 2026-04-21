"""Unit tests for retrieval metrics."""

from __future__ import annotations

from snow_sports_rag.evaluation.gold import GoldItem
from snow_sports_rag.evaluation.metrics import (
    aggregate_query_metrics,
    binary_ndcg_at_k,
    latency_percentiles_ms,
    mrr_for_hits,
    per_query_metrics,
    recall_success,
)
from snow_sports_rag.retrieval.models import RetrievalHit


def _hit(doc_id: str, text: str = "x") -> RetrievalHit:
    return RetrievalHit(
        chunk_id=f"{doc_id}::0",
        text=text,
        doc_id=doc_id,
        section_path="",
        chunk_index=0,
        similarity=0.9,
        distance=0.1,
    )


def test_recall_success_doc_and_keywords() -> None:
    item = GoldItem(
        question="q",
        expected_doc_ids=("a.md",),
        must_contain_keywords=("foo",),
    )
    hits = [_hit("b.md", "no"), _hit("a.md", "foo bar")]
    assert recall_success(item, hits)


def test_mrr_expected_doc_rank2() -> None:
    item = GoldItem(question="q", expected_doc_ids=("want.md",))
    hits = [_hit("other.md"), _hit("want.md")]
    assert mrr_for_hits(item, hits) == 0.5


def test_ndcg_binary() -> None:
    item = GoldItem(question="q", expected_doc_ids=("a.md",))
    hits = [_hit("b.md"), _hit("a.md")]
    assert binary_ndcg_at_k(item, hits, k=8) > 0.0


def test_latency_percentiles() -> None:
    p50, p95 = latency_percentiles_ms([10.0, 20.0, 30.0, 40.0])
    assert p50 == 25.0
    assert p95 >= 37.0


def test_aggregate_empty() -> None:
    assert aggregate_query_metrics([])["n_queries"] == 0


def test_per_query_keyword_mrr() -> None:
    item = GoldItem(
        question="q",
        expected_doc_ids=(),
        must_contain_keywords=("needle",),
    )
    hits = [_hit("a.md", "hay"), _hit("b.md", "needle")]
    m = per_query_metrics(item, hits, k=5, latency_ms=1.0)
    assert m.reciprocal_rank == 0.5
