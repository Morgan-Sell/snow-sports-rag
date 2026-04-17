from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from snow_sports_rag.chunking import (
    MarkdownHeaderChunkStrategy,
    chunk_strategy_from_config,
)
from snow_sports_rag.chunking.models import Chunk
from snow_sports_rag.config import load_config
from snow_sports_rag.embedding import FakeEmbeddingModel, embedding_model_from_config
from snow_sports_rag.ingest import KnowledgeBaseLoader
from snow_sports_rag.ingest.models import SourceDocument
from snow_sports_rag.retrieval import (
    BaselineRetriever,
    IndexBuilder,
    chroma_cosine_distance_to_similarity,
    validate_embedder_against_manifest,
)
from snow_sports_rag.vectorstore import ChromaVectorStore, vector_store_from_config


class FixedWindowStrategyStub:
    """Minimal one-chunk strategy for manifest mismatch test."""

    def chunk(self, document: SourceDocument) -> list[Chunk]:
        return [
            Chunk(
                text=document.raw_markdown,
                doc_id=document.doc_id,
                entity_type=document.entity_type,
                section_path="",
                chunk_index=0,
            )
        ]


def test_chroma_cosine_distance_to_similarity() -> None:
    assert chroma_cosine_distance_to_similarity(0.0) == 1.0
    assert chroma_cosine_distance_to_similarity(1.0) == 0.0


def test_validate_manifest_mismatch_model_name() -> None:
    emb = FakeEmbeddingModel(dimension=8, model_name="a")
    with pytest.raises(ValueError, match="model_name"):
        validate_embedder_against_manifest(emb, {"model_name": "b", "dimension": 8})


def test_validate_manifest_mismatch_dimension() -> None:
    emb = FakeEmbeddingModel(dimension=8, model_name="x")
    with pytest.raises(ValueError, match="dimension"):
        validate_embedder_against_manifest(emb, {"model_name": "x", "dimension": 4})


def test_validate_manifest_none_is_noop() -> None:
    emb = FakeEmbeddingModel(dimension=8)
    validate_embedder_against_manifest(emb, None)


def test_baseline_retriever_similarity_and_metadata(tmp_path: Path) -> None:
    embedder = FakeEmbeddingModel(dimension=8, model_name="unit-fake")
    store = ChromaVectorStore(tmp_path / "db", "snow_sports_kb")
    strategy = MarkdownHeaderChunkStrategy(chunk_size=200, chunk_overlap=40)
    doc = SourceDocument(
        doc_id="athletes/x.md",
        entity_type="athletes",
        title="X",
        raw_markdown="# X\n\n## Bio\n\nalpha alpine racing career.\n",
        headings=["Bio"],
    )
    builder = IndexBuilder(strategy, embedder, store)
    assert builder.build([doc]) >= 1
    retriever = BaselineRetriever(embedder, store, top_k=5, validate_manifest=True)
    hits = retriever.retrieve("alpha alpine racing")
    assert len(hits) >= 1
    h0 = hits[0]
    assert h0.doc_id == "athletes/x.md"
    assert h0.section_path == "Bio"
    assert h0.similarity == pytest.approx(1.0 - h0.distance)
    assert "alpha" in h0.text


def test_baseline_retriever_manifest_mismatch_after_build(tmp_path: Path) -> None:
    embed_a = FakeEmbeddingModel(dimension=8, model_name="model-a")
    store = ChromaVectorStore(tmp_path / "db", "snow_sports_kb")
    strategy = FixedWindowStrategyStub()
    doc = SourceDocument(
        doc_id="d.md",
        entity_type="t",
        title="T",
        raw_markdown="hello world",
        headings=[],
    )
    IndexBuilder(strategy, embed_a, store).build([doc])
    embed_b = FakeEmbeddingModel(dimension=8, model_name="model-b")
    bad = BaselineRetriever(embed_b, store, top_k=3, validate_manifest=True)
    with pytest.raises(ValueError, match="model_name"):
        bad.retrieve("hello")


@pytest.mark.integration
def test_baseline_retrieval_pipeline_integration(tmp_path: Path) -> None:
    """Config-driven ingest, index rebuild, and query (tmp KB + YAML + Chroma)."""
    kb = tmp_path / "knowledge-base" / "athletes"
    kb.mkdir(parents=True)
    (kb / "Skier One.md").write_text(
        "# Skier One\n\n## Bio\n\nDownhill specialist from Vermont.\n",
        encoding="utf-8",
    )
    (kb / "Skier Two.md").write_text(
        "# Skier Two\n\n## Bio\n\nSlalom focus and Vermont training.\n",
        encoding="utf-8",
    )
    cfg_dir = tmp_path / "configs"
    cfg_dir.mkdir()
    chroma_dir = tmp_path / "chroma"
    cfg = {
        "knowledge_base_path": "knowledge-base",
        "chunking": {
            "strategy": "markdown_header",
            "chunk_size": 256,
            "chunk_overlap": 32,
            "min_section_chars": 0,
        },
        "embedding": {
            "backend": "fake",
            "dimension": 12,
            "model_name": "integration-fake",
            "normalize": True,
        },
        "vector_store": {
            "backend": "chroma",
            "persist_directory": str(chroma_dir),
            "collection_name": "snow_sports_kb",
        },
        "retrieval": {"top_k": 5, "l1_shortlist_m": 10},
    }
    (cfg_dir / "default.yaml").write_text(yaml.dump(cfg), encoding="utf-8")

    app = load_config(cfg_dir / "default.yaml", base_dir=tmp_path)
    docs = KnowledgeBaseLoader(app).load_all()
    assert len(docs) == 2
    strategy = chunk_strategy_from_config(app.chunking)
    embedder = embedding_model_from_config(app.embedding)
    store = vector_store_from_config(app.vector_store)
    n = IndexBuilder(strategy, embedder, store).build(docs)
    assert n >= 2
    retriever = BaselineRetriever(
        embedder,
        store,
        top_k=int(app.retrieval["top_k"]),
        validate_manifest=True,
    )
    hits = retriever.retrieve("Vermont slalom")
    assert len(hits) >= 1
    for h in hits:
        assert h.doc_id
        assert h.similarity == pytest.approx(1.0 - h.distance)
    assert any(h.section_path == "Bio" for h in hits)
    doc_ids = {h.doc_id for h in hits}
    assert "athletes/Skier One.md" in doc_ids or "athletes/Skier Two.md" in doc_ids
