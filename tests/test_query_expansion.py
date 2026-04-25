from __future__ import annotations

import pytest

from snow_sports_rag.llm import FakeLLMClient
from snow_sports_rag.retrieval import QueryExpander
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


class StubRetriever:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def retrieve(self, query: str, *, k: int | None = None) -> list[RetrievalHit]:
        self.calls.append(query)
        k_eff = k if k is not None else 8
        q = query.strip()
        if q == "original q":
            return [_hit("a", 0.5), _hit("b", 0.4)][:k_eff]
        if q.startswith("paraphrase:"):
            return [_hit("b", 0.95), _hit("c", 0.2)][:k_eff]
        return []


def test_query_expander_disabled_delegates() -> None:
    inner = StubRetriever()
    wrapped = QueryExpander(inner, FakeLLMClient(), enabled=False)
    hits = wrapped.retrieve("original q", k=2)
    assert len(hits) == 2
    assert inner.calls == ["original q"]


def test_query_expander_fuses_multi_query_max_score() -> None:
    inner = StubRetriever()
    llm = FakeLLMClient(prefix="paraphrase:")
    wrapped = QueryExpander(
        inner,
        llm,
        enabled=True,
        num_paraphrases=1,
        fusion="max_score",
        per_query_k=10,
        top_n_fused=10,
        default_inner_k=5,
    )
    hits = wrapped.retrieve("original q", k=5)
    assert len(inner.calls) >= 2
    by_id = {h.chunk_id: h for h in hits}
    assert "b" in by_id
    assert by_id["b"].similarity == pytest.approx(0.95)


def test_query_expander_rrf_mode() -> None:
    inner = StubRetriever()
    wrapped = QueryExpander(
        inner,
        FakeLLMClient(),
        enabled=True,
        num_paraphrases=1,
        fusion="rrf",
        per_query_k=10,
        top_n_fused=10,
        default_inner_k=5,
    )
    hits = wrapped.retrieve("original q", k=5)
    assert hits
