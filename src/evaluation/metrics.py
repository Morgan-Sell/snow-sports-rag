"""Retrieval metrics for gold-set evaluation (Phase 4.2)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

from ..retrieval.models import RetrievalHit
from .gold import GoldItem

__all__ = [
    "QueryMetrics",
    "aggregate_query_metrics",
    "binary_ndcg_at_k",
    "latency_percentiles_ms",
    "mrr_for_hits",
    "per_query_metrics",
    "recall_success",
]


@dataclass(frozen=True)
class QueryMetrics:
    """Per-gold-item retrieval scores."""

    recall_hit: bool
    reciprocal_rank: float
    ndcg: float
    latency_ms: float


def recall_success(item: GoldItem, hits: Sequence[RetrievalHit]) -> bool:
    """Return True if all stated gold constraints are satisfied.

    - When ``expected_doc_ids`` is non-empty: at least one ``doc_id`` must
      appear among ``hits`` (any position).
    - When ``must_contain_keywords`` is non-empty: every keyword must appear as
      a case-insensitive substring in the concatenation of ``hits`` texts
      (order preserved).
    """
    doc_ok = True
    if item.expected_doc_ids:
        got = {h.doc_id for h in hits}
        doc_ok = bool(got.intersection(set(item.expected_doc_ids)))

    kw_ok = True
    if item.must_contain_keywords:
        blob = " ".join(h.text for h in hits).lower()
        kw_ok = all(k.lower() in blob for k in item.must_contain_keywords)

    return doc_ok and kw_ok


def mrr_for_hits(item: GoldItem, hits: Sequence[RetrievalHit]) -> float:
    """Mean reciprocal rank for a single query.

    If ``expected_doc_ids`` is set, uses the 1-based rank of the first hit
    whose ``doc_id`` is relevant. Otherwise uses keyword satisfaction on a
    per-hit basis (first hit whose text contains all keywords).
    """
    if item.expected_doc_ids:
        rel = set(item.expected_doc_ids)
        for i, h in enumerate(hits, start=1):
            if h.doc_id in rel:
                return 1.0 / float(i)
        return 0.0

    if not item.must_contain_keywords:
        return 0.0

    keys = [k.lower() for k in item.must_contain_keywords]
    acc: list[str] = []
    for i, h in enumerate(hits, start=1):
        acc.append(h.text)
        blob = " ".join(acc).lower()
        if all(k in blob for k in keys):
            return 1.0 / float(i)
    return 0.0


def binary_ndcg_at_k(item: GoldItem, hits: Sequence[RetrievalHit], *, k: int) -> float:
    """nDCG@k with binary relevance from ``expected_doc_ids`` only.

    If ``expected_doc_ids`` is empty, returns ``1.0`` when :func:`recall_success`
    is true (keywords-only items), else ``0.0``.
    """
    k_eff = max(1, min(k, len(hits)))
    if not item.expected_doc_ids:
        return 1.0 if recall_success(item, hits) else 0.0

    rel = set(item.expected_doc_ids)

    def dcg_from_rels(rels: list[int]) -> float:
        s = 0.0
        for i, r in enumerate(rels[:k_eff]):
            gain = (2**r - 1) if r else 0.0
            s += gain / math.log2(float(i) + 2.0)
        return s

    rels = [1 if h.doc_id in rel else 0 for h in hits[:k_eff]]
    dcg = dcg_from_rels(rels)
    ideal = sorted(rels, reverse=True)
    idcg = dcg_from_rels(ideal)
    if idcg <= 0.0:
        return 0.0
    return dcg / idcg


def latency_percentiles_ms(latencies_ms: Sequence[float]) -> tuple[float, float]:
    """Return (p50, p95) using linear interpolation on sorted samples."""
    xs = sorted(float(x) for x in latencies_ms)
    if not xs:
        return (0.0, 0.0)
    n = len(xs)

    def pct(p: float) -> float:
        if n == 1:
            return xs[0]
        idx = p * (n - 1)
        lo = int(math.floor(idx))
        hi = int(math.ceil(idx))
        if lo == hi:
            return xs[lo]
        frac = idx - lo
        return xs[lo] * (1.0 - frac) + xs[hi] * frac

    return (pct(0.50), pct(0.95))


def per_query_metrics(
    item: GoldItem,
    hits: Sequence[RetrievalHit],
    *,
    k: int,
    latency_ms: float,
) -> QueryMetrics:
    """Aggregate per-query metrics for one gold row."""
    rh = recall_success(item, hits)
    rr = mrr_for_hits(item, hits)
    nd = binary_ndcg_at_k(item, hits, k=k)
    return QueryMetrics(
        recall_hit=rh,
        reciprocal_rank=rr,
        ndcg=nd,
        latency_ms=latency_ms,
    )


def aggregate_query_metrics(rows: Sequence[QueryMetrics]) -> dict[str, Any]:
    """Mean recall, MRR, nDCG, and latency percentiles across queries."""
    if not rows:
        return {
            "recall_at_k": 0.0,
            "mrr": 0.0,
            "ndcg_at_k": 0.0,
            "latency_ms_p50": 0.0,
            "latency_ms_p95": 0.0,
            "n_queries": 0,
        }
    n = len(rows)
    recall_at_k = sum(1.0 if r.recall_hit else 0.0 for r in rows) / n
    mrr = sum(r.reciprocal_rank for r in rows) / n
    ndcg = sum(r.ndcg for r in rows) / n
    p50, p95 = latency_percentiles_ms([r.latency_ms for r in rows])
    return {
        "recall_at_k": recall_at_k,
        "mrr": mrr,
        "ndcg_at_k": ndcg,
        "latency_ms_p50": p50,
        "latency_ms_p95": p95,
        "n_queries": n,
    }
