from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from snow_sports_rag.chunking.models import Chunk
from snow_sports_rag.config import load_config
from snow_sports_rag.embedding import FakeEmbeddingModel
from snow_sports_rag.vectorstore import (
    ChromaVectorStore,
    pack_chunk_upsert,
    vector_store_from_config,
)


def test_vector_store_factory_chroma(tmp_path: Path) -> None:
    store = vector_store_from_config(
        {
            "backend": "chroma",
            "persist_directory": tmp_path / "ch",
            "collection_name": "snow_sports_kb",
        }
    )
    assert isinstance(store, ChromaVectorStore)


def test_vector_store_factory_unknown_backend() -> None:
    with pytest.raises(ValueError, match="Unknown"):
        vector_store_from_config({"backend": "lancedb", "persist_directory": "/x"})


def test_vector_store_factory_requires_persist() -> None:
    with pytest.raises(ValueError, match="persist_directory"):
        vector_store_from_config({"backend": "chroma"})


def test_vector_store_factory_collection_name_too_short(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="collection_name"):
        vector_store_from_config(
            {
                "backend": "chroma",
                "persist_directory": tmp_path,
                "collection_name": "ab",
            }
        )


def test_chroma_upsert_query_roundtrip(tmp_path: Path) -> None:
    store = ChromaVectorStore(tmp_path / "db", "snow_sports_kb")
    chunks = [
        Chunk(
            text="alpha alpine skiing",
            doc_id="athletes/a.md",
            entity_type="athletes",
            section_path="Bio",
            chunk_index=0,
        ),
        Chunk(
            text="beta snowboard halfpipe",
            doc_id="athletes/b.md",
            entity_type="athletes",
            section_path="Career",
            chunk_index=0,
        ),
    ]
    embed = FakeEmbeddingModel(dimension=8, normalize=True)
    mat = embed.embed_documents([c.text for c in chunks])
    ids, docs, metas, emb = pack_chunk_upsert(chunks, mat)
    store.upsert(ids=ids, embeddings=emb, documents=docs, metadatas=metas)

    q = embed.embed_query("alpha alpine skiing")
    result = store.query(query_embedding=q, k=2)
    assert len(result.hits) == 2
    best = result.hits[0]
    assert best.metadata["doc_id"] == "athletes/a.md"
    assert best.metadata["section_path"] == "Bio"
    assert "alpha" in best.document


def test_chroma_write_read_manifest(tmp_path: Path) -> None:
    store = ChromaVectorStore(tmp_path / "db", "snow_sports_kb")
    store.write_embedding_manifest("fake-deterministic", 384)
    data = store.read_embedding_manifest()
    assert data == {"model_name": "fake-deterministic", "dimension": 384}


def test_chroma_reset_clears_collection(tmp_path: Path) -> None:
    store = ChromaVectorStore(tmp_path / "db", "snow_sports_kb")
    store.upsert(
        ids=["x"],
        embeddings=np.array([[1.0, 0.0]], dtype=np.float64),
        documents=["one"],
        metadatas=[
            {
                "doc_id": "d",
                "entity_type": "t",
                "section_path": "",
                "chunk_index": 0,
            }
        ],
    )
    store.reset()
    out = store.query(query_embedding=np.array([1.0, 0.0], dtype=np.float64), k=1)
    assert out.hits == []


def test_load_config_resolves_vector_store_path(tmp_path: Path) -> None:
    cfg_dir = tmp_path / "configs"
    cfg_dir.mkdir()
    (cfg_dir / "default.yaml").write_text(
        "knowledge_base_path: kb\n",
        encoding="utf-8",
    )
    kb = tmp_path / "kb"
    kb.mkdir()
    cfg = load_config(cfg_dir / "default.yaml", base_dir=tmp_path)
    expected = (tmp_path / ".rag_index" / "chroma").resolve()
    assert cfg.vector_store["persist_directory"] == expected


def test_pack_chunk_upsert_length_mismatch() -> None:
    chunks = [
        Chunk(
            text="a",
            doc_id="d",
            entity_type="t",
            section_path="",
            chunk_index=0,
        )
    ]
    bad = np.zeros((2, 3), dtype=np.float64)
    with pytest.raises(ValueError, match="len"):
        pack_chunk_upsert(chunks, bad)
