from __future__ import annotations

from pathlib import Path

import pytest

from snow_sports_rag.chunking import (
    FixedWindowChunkStrategy,
    MarkdownHeaderChunkStrategy,
    RecursiveCharChunkStrategy,
    chunk_strategy_from_config,
)
from snow_sports_rag.chunking.models import Chunk
from snow_sports_rag.config import load_config
from snow_sports_rag.ingest.models import SourceDocument


def _doc(
    raw: str,
    *,
    doc_id: str = "athletes/x.md",
    entity_type: str = "athletes",
    title: str = "T",
    headings: tuple[str, ...] = (),
) -> SourceDocument:
    return SourceDocument(
        doc_id=doc_id,
        entity_type=entity_type,
        title=title,
        raw_markdown=raw,
        headings=headings,
    )


def test_markdown_header_splits_on_h2_and_subwindows() -> None:
    raw = (
        "# Title\n\nIntro line\n\n## Section A\n\n" + ("x" * 120) + "\n\n## B\n\nDone\n"
    )
    strat = MarkdownHeaderChunkStrategy(
        chunk_size=40,
        chunk_overlap=8,
        min_section_chars=0,
    )
    chunks = strat.chunk(_doc(raw))
    paths = [c.section_path for c in chunks]
    assert paths[0] == ""
    assert "Section A" in paths
    assert paths[-1] == "B"
    assert all(isinstance(c, Chunk) for c in chunks)
    assert list(range(len(chunks))) == [c.chunk_index for c in chunks]
    a_chunks = [c for c in chunks if c.section_path == "Section A"]
    assert len(a_chunks) > 1


def test_markdown_header_min_section_chars() -> None:
    raw = "## Tiny\n\nab\n\n## Big\n\n" + "y" * 50
    strat = MarkdownHeaderChunkStrategy(
        chunk_size=80,
        chunk_overlap=0,
        min_section_chars=10,
    )
    chunks = strat.chunk(_doc(raw))
    assert [c.section_path for c in chunks] == ["Big"]


def test_fixed_window_sliding() -> None:
    strat = FixedWindowChunkStrategy(chunk_size=4, chunk_overlap=2)
    chunks = strat.chunk(_doc("abcdefghij"))
    assert [c.text for c in chunks] == ["abcd", "cdef", "efgh", "ghij"]
    assert all(c.section_path == "" for c in chunks)


def test_recursive_character_merges_short_splits() -> None:
    raw = "Paragraph one.\n\nParagraph two is here.\n\nThird block."
    strat = RecursiveCharChunkStrategy(chunk_size=36, chunk_overlap=4)
    chunks = strat.chunk(_doc(raw))
    joined = "".join(c.text for c in chunks)
    assert "Paragraph one" in joined
    assert all(c.section_path == "" for c in chunks)


def test_chunk_strategy_from_config_unknown() -> None:
    with pytest.raises(ValueError, match="Unknown chunking strategy"):
        chunk_strategy_from_config(
            {"strategy": "nope", "chunk_size": 10, "chunk_overlap": 0},
        )


def test_chunk_strategy_from_config_recursive_separators_type() -> None:
    with pytest.raises(TypeError, match="recursive_separators"):
        chunk_strategy_from_config(
            {
                "strategy": "recursive_character",
                "chunk_size": 32,
                "chunk_overlap": 0,
                "recursive_separators": "not-a-list",
            }
        )


def test_chunk_strategy_from_config_loads_default(tmp_path: Path) -> None:
    cfg_dir = tmp_path / "configs"
    cfg_dir.mkdir()
    (cfg_dir / "default.yaml").write_text(
        "knowledge_base_path: kb\nchunking:\n  strategy: fixed_window\n",
        encoding="utf-8",
    )
    (tmp_path / "kb").mkdir()
    cfg = load_config(cfg_dir / "default.yaml", base_dir=tmp_path)
    strat = chunk_strategy_from_config(cfg.chunking)
    assert isinstance(strat, FixedWindowChunkStrategy)
