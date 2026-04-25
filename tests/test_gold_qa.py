"""Tests for Phase 4.1 gold Q&A JSONL loader."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from snow_sports_rag.evaluation import (
    default_gold_qa_path,
    GoldItem,
    load_gold_qa,
)


def test_default_gold_file_exists() -> None:
    p = default_gold_qa_path()
    assert p.name == "gold_qa.jsonl"
    assert (p.parent / "gold_qa.jsonl").is_file()


def test_load_gold_qa_non_empty() -> None:
    items = load_gold_qa()
    assert len(items) >= 20
    for it in items:
        assert isinstance(it, GoldItem)
        assert it.question
        assert it.expected_doc_ids or it.must_contain_keywords


def test_gold_item_doc_ids_are_posix_paths() -> None:
    items = load_gold_qa()
    for it in items:
        for doc_id in it.expected_doc_ids:
            assert "\\" not in doc_id
            assert doc_id.endswith(".md")


def test_load_from_explicit_path(tmp_path: Path) -> None:
    f = tmp_path / "mini.jsonl"
    f.write_text(
        json.dumps(
            {
                "question": "Test?",
                "expected_doc_ids": ["athletes/Foo.md"],
                "must_contain_keywords": ["bar"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    items = load_gold_qa(f)
    assert len(items) == 1
    assert items[0].question == "Test?"
    assert items[0].expected_doc_ids == ("athletes/Foo.md",)
    assert items[0].must_contain_keywords == ("bar",)


def test_rejects_empty_constraints(tmp_path: Path) -> None:
    f = tmp_path / "bad.jsonl"
    f.write_text(
        json.dumps({"question": "Only question", "expected_doc_ids": []}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="at least one"):
        load_gold_qa(f)


def test_rejects_unknown_field(tmp_path: Path) -> None:
    f = tmp_path / "bad2.jsonl"
    f.write_text(
        json.dumps(
            {
                "question": "Q",
                "expected_doc_ids": ["a.md"],
                "extra": 1,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unknown fields"):
        load_gold_qa(f)
