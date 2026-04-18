from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from snow_sports_rag.rerank import (
    CrossEncoderReranker,
    IdentityReranker,
    reranker_from_config,
)
from snow_sports_rag.rerank.anthropic_listwise import AnthropicListwiseReranker
from snow_sports_rag.rerank.listwise_utils import parse_ranked_indices, passage_snippet
from snow_sports_rag.rerank.openai_listwise import OpenAIListwiseReranker
from snow_sports_rag.retrieval.models import RetrievalHit


def _hit(cid: str, sim: float) -> RetrievalHit:
    return RetrievalHit(
        chunk_id=cid,
        text=f"text-{cid}",
        doc_id="d.md",
        section_path="S",
        chunk_index=0,
        similarity=sim,
        distance=1.0 - sim,
    )


def test_identity_reranker_truncates() -> None:
    r = IdentityReranker()
    hits = [_hit(str(i), 0.5) for i in range(10)]
    out = r.rerank("q", hits, top_k=3)
    assert len(out) == 3
    assert out[0].chunk_id == "0"


def test_parse_ranked_indices_basic() -> None:
    s = 'Preamble {"ranked_indices": [2, 0, 1]}'
    assert parse_ranked_indices(s, n_passages=3) == [2, 0, 1]


def test_parse_ranked_indices_skips_invalid() -> None:
    s = '{"ranked_indices": [1, 99, 1, 0]}'
    assert parse_ranked_indices(s, n_passages=3) == [1, 0]


def test_passage_snippet_truncates() -> None:
    long = "word " * 200
    s = passage_snippet(long, max_chars=20)
    assert len(s) <= 21
    assert s.endswith("…")


def test_reranker_from_config_disabled_is_identity() -> None:
    r = reranker_from_config({"enabled": False, "backend": "cross_encoder"})
    assert isinstance(r, IdentityReranker)


def test_reranker_from_config_noop() -> None:
    r = reranker_from_config({"enabled": True, "backend": "noop"})
    assert isinstance(r, IdentityReranker)


def test_reranker_from_config_unknown_backend() -> None:
    with pytest.raises(ValueError, match="Unknown"):
        reranker_from_config({"enabled": True, "backend": "tensorrt"})


def test_cross_encoder_reranker_orders_by_mocked_scores() -> None:
    hits = [_hit("a", 0.1), _hit("b", 0.2), _hit("c", 0.3)]
    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([0.2, 0.9, 0.5], dtype=np.float64)
    mock_cls = MagicMock(return_value=mock_model)
    with patch("sentence_transformers.CrossEncoder", mock_cls):
        r = CrossEncoderReranker("dummy-model")
        out = r.rerank("query text", hits, top_k=2)
    assert [h.chunk_id for h in out] == ["b", "c"]
    assert out[0].similarity == pytest.approx(0.9)


def test_openai_listwise_reranker_parses_order() -> None:
    hits = [_hit("a", 0.1), _hit("b", 0.2), _hit("c", 0.3)]
    api_body = {
        "choices": [
            {
                "message": {
                    "content": '{"ranked_indices": [2, 0, 1]}',
                }
            }
        ]
    }
    raw = json.dumps(api_body).encode("utf-8")

    class _FakeResp:
        def read(self) -> bytes:
            return raw

        def __enter__(self) -> _FakeResp:
            return self

        def __exit__(self, *args: object) -> None:
            return None

    def fake_urlopen(req: object, timeout: float = 0) -> _FakeResp:
        return _FakeResp()

    with patch.object(OpenAIListwiseReranker, "_resolve_api_key", return_value="sk"):
        with patch("urllib.request.urlopen", fake_urlopen):
            r = OpenAIListwiseReranker(model="gpt-4o-mini", api_key="x")
            out = r.rerank("q", hits, top_k=2)
    assert [h.chunk_id for h in out] == ["c", "a"]


def test_anthropic_listwise_reranker_parses_order() -> None:
    hits = [_hit("a", 0.1), _hit("b", 0.2)]
    api_body = {
        "content": [{"type": "text", "text": '{"ranked_indices": [1, 0]}'}],
    }
    raw = json.dumps(api_body).encode("utf-8")

    class _FakeRespB:
        def read(self) -> bytes:
            return raw

        def __enter__(self) -> _FakeRespB:
            return self

        def __exit__(self, *args: object) -> None:
            return None

    def fake_urlopen_b(req: object, timeout: float = 0) -> _FakeRespB:
        return _FakeRespB()

    with patch.object(AnthropicListwiseReranker, "_resolve_api_key", return_value="k"):
        with patch("urllib.request.urlopen", fake_urlopen_b):
            r = AnthropicListwiseReranker(
                model="claude-3-5-haiku-20241022",
                api_key="x",
            )
            out = r.rerank("q", hits, top_k=1)
    assert out[0].chunk_id == "b"


def test_reranker_from_config_openai_inherits_llm() -> None:
    r = reranker_from_config(
        {
            "enabled": True,
            "backend": "openai",
            "openai_model": None,
        },
        llm={
            "model": "gpt-4o-mini",
            "base_url": "https://api.openai.com/v1",
            "api_key_env": "OPENAI_API_KEY",
            "temperature": 0.1,
            "timeout_s": 30.0,
        },
    )
    assert isinstance(r, OpenAIListwiseReranker)
