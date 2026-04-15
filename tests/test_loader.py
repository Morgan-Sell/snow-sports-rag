from __future__ import annotations

from pathlib import Path

import yaml

from snow_sports_rag.config import load_config
from snow_sports_rag.ingest import KnowledgeBaseLoader


def test_loader_returns_documents(tmp_path: Path) -> None:
    kb = tmp_path / "knowledge-base"
    (kb / "athletes").mkdir(parents=True)
    (kb / "athletes" / "Alex.md").write_text(
        "# Athlete Profile: Alex\n\n## Bio\nHello\n",
        encoding="utf-8",
    )
    cfg_dir = tmp_path / "configs"
    cfg_dir.mkdir()
    (cfg_dir / "default.yaml").write_text(
        yaml.dump({"knowledge_base_path": "knowledge-base"}),
        encoding="utf-8",
    )

    cfg = load_config(cfg_dir / "default.yaml", base_dir=tmp_path)
    docs = KnowledgeBaseLoader(cfg).load_all()

    assert len(docs) == 1
    d = docs[0]
    assert d.doc_id == "athletes/Alex.md"
    assert d.entity_type == "athletes"
    assert d.title == "Athlete Profile: Alex"
    assert d.headings == ("Bio",)


def test_load_repository_knowledge_base() -> None:
    repo = Path(__file__).resolve().parent.parent
    cfg = load_config(repo / "configs/default.yaml", base_dir=repo)
    docs = KnowledgeBaseLoader(cfg).load_all()
    assert len(docs) == 70
    types = {d.entity_type for d in docs}
    assert types == {"athletes", "circuits", "competitions", "resorts", "results"}
