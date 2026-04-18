from __future__ import annotations

from pathlib import Path

from snow_sports_rag.chunking import MarkdownHeaderChunkStrategy
from snow_sports_rag.embedding import FakeEmbeddingModel
from snow_sports_rag.ingest.models import SourceDocument
from snow_sports_rag.retrieval import (
    HierarchicalRetriever,
    IndexBuilder,
    l1_summary_text,
)
from snow_sports_rag.vectorstore import (
    ChromaVectorStore,
    chroma_l2_l1_stores_from_config,
)


def test_l1_summary_prefers_overview_heading() -> None:
    doc = SourceDocument(
        doc_id="x.md",
        entity_type="t",
        title="Alpha",
        raw_markdown="# Alpha\n\n## Bio\n\nignore.\n\n## Overview\n\npick this text.\n",
        headings=("Bio", "Overview"),
    )
    s = l1_summary_text(doc)
    assert "Alpha" in s
    assert "pick this text" in s
    assert "ignore" not in s


def test_l1_summary_first_h2_when_no_keyword() -> None:
    doc = SourceDocument(
        doc_id="y.md",
        entity_type="t",
        title="Beta",
        raw_markdown="# Beta\n\n## Intro\n\nfirst section only.\n\n## Later\n\nno.",
        headings=("Intro", "Later"),
    )
    s = l1_summary_text(doc)
    assert "Beta" in s
    assert "first section only" in s


def test_chroma_l2_l1_stores_from_config_default_l1_name(tmp_path: Path) -> None:
    l2, l1 = chroma_l2_l1_stores_from_config(
        {
            "backend": "chroma",
            "persist_directory": str(tmp_path / "db"),
            "collection_name": "snow_sports_kb",
        }
    )
    assert l2.collection_name == "snow_sports_kb"
    assert l1.collection_name == "snow_sports_kb_l1"
    assert l2.persist_directory == l1.persist_directory


def test_hierarchical_retriever_max_two_chunks_per_doc(tmp_path: Path) -> None:
    """L1 shortlists both docs; L2 has many chunks; output obeys per-doc cap."""
    embedder = FakeEmbeddingModel(dimension=8, model_name="unit-fake")
    root = tmp_path / "db"
    l2 = ChromaVectorStore(root, "snow_sports_kb")
    l1 = ChromaVectorStore(root, "snow_sports_kb_l1")
    strategy = MarkdownHeaderChunkStrategy(chunk_size=80, chunk_overlap=10)
    docs = [
        SourceDocument(
            doc_id="athletes/a.md",
            entity_type="athletes",
            title="Skier A",
            raw_markdown=(
                "# Skier A\n\n## Bio\n\n"
                + "wordone " * 30
                + "\n\n## Career\n\n"
                + "wordtwo " * 30
                + "\n\n## Results\n\n"
                + "wordthree " * 30
                + "\n"
            ),
            headings=("Bio", "Career", "Results"),
        ),
        SourceDocument(
            doc_id="athletes/b.md",
            entity_type="athletes",
            title="Skier B",
            raw_markdown=(
                "# Skier B\n\n## Bio\n\n"
                + "wordone " * 30
                + "\n\n## Career\n\n"
                + "wordfour " * 30
                + "\n"
            ),
            headings=("Bio", "Career"),
        ),
    ]
    IndexBuilder(strategy, embedder, l2, l1_store=l1).build(docs)
    retriever = HierarchicalRetriever(
        embedder,
        l2,
        l1,
        top_k=8,
        l1_top_m=5,
        max_chunks_per_doc=2,
        global_fallback=False,
    )
    hits = retriever.retrieve("wordone wordtwo wordthree wordfour")
    counts: dict[str, int] = {}
    for h in hits:
        counts[h.doc_id] = counts.get(h.doc_id, 0) + 1
    assert sum(counts.values()) == len(hits)
    for _doc_id, n in counts.items():
        assert n <= 2
    assert len(hits) <= 8


def test_hierarchical_global_fallback_fills_shortlist(tmp_path: Path) -> None:
    """When L2 has no chunks in shortlisted docs, global L2 supplies passages."""
    embedder = FakeEmbeddingModel(dimension=8, model_name="unit-fake")
    root = tmp_path / "db"
    l2 = ChromaVectorStore(root, "snow_sports_kb")
    l1 = ChromaVectorStore(root, "snow_sports_kb_l1")
    strategy = MarkdownHeaderChunkStrategy(chunk_size=200, chunk_overlap=20)
    good = SourceDocument(
        doc_id="athletes/good.md",
        entity_type="athletes",
        title="Good",
        raw_markdown="# Good\n\n## Bio\n\nunique zebra alpha content.\n",
        headings=("Bio",),
    )
    empty_body = SourceDocument(
        doc_id="athletes/emptyish.md",
        entity_type="athletes",
        title="Emptyish",
        raw_markdown="# Emptyish\n\n",
        headings=(),
    )
    IndexBuilder(strategy, embedder, l2, l1_store=l1).build([good, empty_body])
    retriever = HierarchicalRetriever(
        embedder,
        l2,
        l1,
        top_k=3,
        l1_top_m=2,
        max_chunks_per_doc=2,
        global_fallback=True,
        l2_prefetch_k=16,
    )
    hits = retriever.retrieve("unique zebra alpha")
    assert hits
    assert any("zebra" in h.text for h in hits)
