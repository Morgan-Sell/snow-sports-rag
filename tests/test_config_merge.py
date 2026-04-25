"""Tests for evaluation sweep :mod:`snow_sports_rag.evaluation.config_merge`."""

from __future__ import annotations

from pathlib import Path

import pytest
from snow_sports_rag.config.loader import AppConfig, load_config
from snow_sports_rag.evaluation.config_merge import (
    flat_overrides_to_nested,
    merge_app_config_overrides,
    merged_config_from_flat,
)


def _minimal_cfg(tmp_path: Path) -> AppConfig:
    p = tmp_path / "c.yaml"
    p.write_text(
        "\n".join(
            [
                "knowledge_base_path: kb",
                "chunking:",
                "  strategy: markdown_header",
                "  chunk_size: 512",
                "  chunk_overlap: 50",
            ]
        ),
        encoding="utf-8",
    )
    (tmp_path / "kb").mkdir()
    return load_config(p, base_dir=tmp_path)


def test_flat_overrides_to_nested() -> None:
    n = flat_overrides_to_nested(
        {"chunking.strategy": "recursive_char", "chunking.chunk_size": 256}
    )
    assert n == {"chunking": {"strategy": "recursive_char", "chunk_size": 256}}


def test_merged_config_from_flat_overrides(tmp_path: Path) -> None:
    base = _minimal_cfg(tmp_path)
    m = merged_config_from_flat(
        base,
        {"chunking.chunk_size": 100, "retrieval.top_k": 12},
    )
    assert m.chunking["chunk_size"] == 100
    assert m.chunking["strategy"] == "markdown_header"
    assert m.retrieval["top_k"] == 12


def test_merge_app_config_unknown_section_raises() -> None:
    base = AppConfig(
        knowledge_base_path=Path("/tmp"),
        chunking={},
        embedding={},
        vector_store={},
        retrieval={},
        rerank={},
        llm={},
        query_expansion={},
        document_expansion={},
        generation={},
        logging={},
    )
    with pytest.raises(ValueError, match="unknown"):
        merge_app_config_overrides(base, {"not_a_section": {}})
