"""Integration test for Phase 4.2 sweep runner (fake embedder, tmp corpus)."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from snow_sports_rag.config.loader import load_config
from snow_sports_rag.evaluation.gold import load_gold_qa
from snow_sports_rag.evaluation.sweep import SweepConfig, run_sweep_retrieval
from snow_sports_rag.ingest import KnowledgeBaseLoader


def _write_min_kb(root: Path) -> None:
    kb = root / "kb"
    (kb / "athletes").mkdir(parents=True)
    (kb / "athletes" / "Test Skier.md").write_text(
        "# Athlete Profile: Test Skier\n\n"
        "## Summary\n"
        "Test Skier trains at **Park City** and won Olympic gold in slalom.\n",
        encoding="utf-8",
    )


def _write_min_config(root: Path) -> Path:
    cfg = root / "configs" / "sweep_test.yaml"
    cfg.parent.mkdir(parents=True)
    cfg.write_text(
        "\n".join(
            [
                "knowledge_base_path: kb",
                "chunking:",
                "  strategy: markdown_header",
                "  chunk_size: 400",
                "  chunk_overlap: 40",
                "embedding:",
                "  backend: fake",
                "  dimension: 64",
                "  model_name: sweep-test",
                "vector_store:",
                "  persist_directory: .rag_index/chroma",
                "  collection_name: sweeptest_l2",
                "  l1_collection_name: sweeptest_l1",
                "retrieval:",
                "  mode: baseline",
                "  top_k: 5",
            ]
        ),
        encoding="utf-8",
    )
    return cfg


def _write_min_gold(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(
        {
            "question": "Where does Test Skier train?",
            "expected_doc_ids": ["athletes/Test Skier.md"],
            "must_contain_keywords": ["Park City"],
        }
    )
    path.write_text(line + "\n", encoding="utf-8")


def test_sweep_two_cell_grid(tmp_path: Path) -> None:
    _write_min_kb(tmp_path)
    cfg_path = _write_min_config(tmp_path)
    gold_path = tmp_path / "evaluation" / "mini_gold.jsonl"
    _write_min_gold(gold_path)

    grid = tmp_path / "grid.yaml"
    grid.write_text(
        yaml.dump(
            {
                "seed": 1,
                "axes": {"chunking.chunk_size": [300, 400]},
            }
        ),
        encoding="utf-8",
    )

    base = load_config(cfg_path, base_dir=tmp_path)
    docs = KnowledgeBaseLoader(base).load_all()
    gold = load_gold_qa(gold_path)
    sweep = SweepConfig(seed=1, axes={"chunking.chunk_size": [300, 400]})

    out = tmp_path / "run1"
    res = run_sweep_retrieval(
        base,
        sweep,
        gold,
        docs,
        out,
        write_csv=True,
    )

    assert res.metrics_path.is_file()
    assert res.csv_path is not None and res.csv_path.is_file()
    payload = json.loads(res.metrics_path.read_text(encoding="utf-8"))
    assert payload["n_cells"] == 2
    assert all(c["status"] == "ok" for c in payload["cells"])
    assert payload["cells"][0]["metrics"]["n_queries"] == 1
