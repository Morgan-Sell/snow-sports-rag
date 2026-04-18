from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from snow_sports_rag.config import load_config


def test_load_config_merges_defaults(tmp_path: Path) -> None:
    cfg_dir = tmp_path / "configs"
    cfg_dir.mkdir()
    minimal = {"knowledge_base_path": "kb"}
    (cfg_dir / "default.yaml").write_text(yaml.dump(minimal), encoding="utf-8")

    kb = tmp_path / "kb"
    kb.mkdir()
    cfg = load_config(cfg_dir / "default.yaml", base_dir=tmp_path)

    assert cfg.knowledge_base_path == kb.resolve()
    assert cfg.chunking["strategy"] == "markdown_header"
    assert cfg.embedding["backend"] == "sentence_transformers"
    assert "model_name" in cfg.embedding
    assert cfg.embedding["normalize"] is True
    assert cfg.vector_store["backend"] == "chroma"
    assert cfg.vector_store["collection_name"] == "snow_sports_kb"
    assert "persist_directory" in cfg.vector_store
    assert cfg.query_expansion["enabled"] is False
    assert cfg.query_expansion["fusion"] == "max_score"


def test_env_overrides_knowledge_base_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg_dir = tmp_path / "configs"
    cfg_dir.mkdir()
    (cfg_dir / "c.yaml").write_text(
        yaml.dump({"knowledge_base_path": "ignored"}),
        encoding="utf-8",
    )
    alt = tmp_path / "other_kb"
    alt.mkdir()
    monkeypatch.setenv("SNOW_SPORTS_RAG_KNOWLEDGE_BASE_PATH", str(alt))

    cfg = load_config(cfg_dir / "c.yaml", base_dir=tmp_path)
    assert cfg.knowledge_base_path == alt.resolve()


def test_config_missing_file() -> None:
    with pytest.raises(FileNotFoundError):
        load_config(Path("/nonexistent/nope.yaml"))
