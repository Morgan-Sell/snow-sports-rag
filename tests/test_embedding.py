from __future__ import annotations

import numpy as np
import pytest

from snow_sports_rag.embedding import (
    FakeEmbeddingModel,
    SentenceTransformerEmbeddingModel,
    embedding_model_from_config,
    l2_normalize_rows,
)


def test_l2_normalize_rows_unit_norms() -> None:
    v = np.array([[3.0, 4.0], [1.0, 0.0]], dtype=np.float64)
    n = l2_normalize_rows(v)
    np.testing.assert_allclose(np.linalg.norm(n, axis=1), [1.0, 1.0], rtol=1e-10)


def test_l2_normalize_rows_rejects_non_2d() -> None:
    with pytest.raises(ValueError, match="2-D"):
        l2_normalize_rows(np.array([1.0, 2.0]))


def test_fake_embedding_deterministic() -> None:
    m = FakeEmbeddingModel(dimension=16, model_name="test-fake")
    a = m.embed_query("same input")
    b = m.embed_query("same input")
    np.testing.assert_array_equal(a, b)
    assert a.shape == (16,)
    np.testing.assert_allclose(np.linalg.norm(a), 1.0, rtol=1e-10)


def test_fake_embedding_different_inputs() -> None:
    m = FakeEmbeddingModel(dimension=32)
    a = m.embed_query("a")
    b = m.embed_query("b")
    assert not np.allclose(a, b)


def test_fake_embed_documents_empty_batch() -> None:
    m = FakeEmbeddingModel(dimension=8)
    out = m.embed_documents([])
    assert out.shape == (0, 8)


def test_fake_index_metadata() -> None:
    m = FakeEmbeddingModel(dimension=384, model_name="fake-deterministic")
    assert m.index_metadata() == {"model_name": "fake-deterministic", "dimension": 384}


def test_embedding_factory_fake() -> None:
    emb = embedding_model_from_config(
        {
            "backend": "fake",
            "dimension": 4,
            "model_name": "unit-test",
            "normalize": True,
        }
    )
    assert isinstance(emb, FakeEmbeddingModel)
    assert emb.dimension == 4
    assert emb.model_name == "unit-test"


def test_embedding_factory_fake_requires_dimension() -> None:
    with pytest.raises(ValueError, match="dimension"):
        embedding_model_from_config({"backend": "fake"})


def test_embedding_factory_unknown_backend() -> None:
    with pytest.raises(ValueError, match="Unknown"):
        embedding_model_from_config({"backend": "not-a-backend", "model_name": "x"})


def test_embedding_factory_sentence_transformers_requires_model_name() -> None:
    with pytest.raises(ValueError, match="model_name"):
        embedding_model_from_config(
            {"backend": "sentence_transformers", "model_name": ""},
        )


@pytest.mark.integration
def test_sentence_transformer_mini_lm_roundtrip() -> None:
    pytest.importorskip("sentence_transformers")
    m = SentenceTransformerEmbeddingModel(
        "sentence-transformers/all-MiniLM-L6-v2",
        normalize=True,
    )
    assert m.dimension == 384
    q = m.embed_query("alpine skiing world cup")
    assert q.shape == (384,)
    np.testing.assert_allclose(np.linalg.norm(q), 1.0, rtol=1e-5)
    batch = m.embed_documents(["a", "b"])
    assert batch.shape == (2, 384)
    meta = m.index_metadata()
    assert meta["model_name"] == "sentence-transformers/all-MiniLM-L6-v2"
    assert meta["dimension"] == 384
