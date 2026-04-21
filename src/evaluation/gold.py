"""Load the Phase 4.1 gold Q&A set from ``evaluation/gold_qa.jsonl``."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class GoldItem:
    """One evaluation question with optional retrieval targets and reference answer.

    At least one of ``expected_doc_ids`` or ``must_contain_keywords`` must be
    non-empty so metrics can score retrieval quality.
    """

    question: str
    expected_doc_ids: tuple[str, ...] = ()
    must_contain_keywords: tuple[str, ...] = ()
    gold_answer: str | None = None


def default_gold_qa_path() -> Path:
    """Return ``evaluation/gold_qa.jsonl`` relative to the repository root.

    Returns
    -------
    Path
        Resolved path to the bundled gold file (repository root is the parent
        of ``src/``).
    """
    return Path(__file__).resolve().parents[2] / "evaluation" / "gold_qa.jsonl"


def load_gold_qa(path: Path | str | None = None) -> list[GoldItem]:
    """Load gold items from a JSONL file.

    Each line must be a JSON object with:

    * ``question`` (str, required)
    * ``expected_doc_ids`` (list of str, optional)
    * ``must_contain_keywords`` (list of str, optional)
    * ``gold_answer`` (str, optional)

    At least one of ``expected_doc_ids`` or ``must_contain_keywords`` must be
    present and non-empty.

    Parameters
    ----------
    path : Path or str, optional
        JSONL path. Defaults to :func:`default_gold_qa_path`.

    Returns
    -------
    list of GoldItem
        One entry per non-empty line, in file order.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    ValueError
        If a line is invalid JSON or fails validation.
    """
    p = Path(path) if path is not None else default_gold_qa_path()
    if not p.is_file():
        raise FileNotFoundError(f"Gold Q&A file not found: {p}")

    out: list[GoldItem] = []
    text = p.read_text(encoding="utf-8")
    for lineno, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            raw = json.loads(stripped)
        except json.JSONDecodeError as e:
            raise ValueError(f"{p}:{lineno}: invalid JSON: {e}") from e
        if not isinstance(raw, dict):
            tname = type(raw).__name__
            raise ValueError(f"{p}:{lineno}: expected JSON object, got {tname}")
        try:
            item = _record_to_item(raw)
        except ValueError as e:
            raise ValueError(f"{p}:{lineno}: {e}") from e
        out.append(item)
    return out


def _record_to_item(raw: dict) -> GoldItem:
    q = raw.get("question")
    if not isinstance(q, str) or not q.strip():
        raise ValueError("field 'question' must be a non-empty string")

    ids = _str_list(raw, "expected_doc_ids")
    keys = _str_list(raw, "must_contain_keywords")
    if not ids and not keys:
        raise ValueError(
            "at least one of 'expected_doc_ids' or 'must_contain_keywords' "
            "must be a non-empty list"
        )

    ga = raw.get("gold_answer")
    if ga is not None and not isinstance(ga, str):
        raise ValueError("field 'gold_answer' must be a string when present")
    extra = set(raw.keys()) - {
        "question",
        "expected_doc_ids",
        "must_contain_keywords",
        "gold_answer",
    }
    if extra:
        raise ValueError(f"unknown fields: {sorted(extra)}")

    return GoldItem(
        question=q.strip(),
        expected_doc_ids=tuple(ids),
        must_contain_keywords=tuple(keys),
        gold_answer=ga.strip() if isinstance(ga, str) and ga.strip() else None,
    )


def _str_list(raw: dict, key: str) -> list[str]:
    val = raw.get(key)
    if val is None:
        return []
    if not isinstance(val, list):
        raise ValueError(f"field '{key}' must be a list of strings")
    out: list[str] = []
    for i, x in enumerate(val):
        if not isinstance(x, str) or not x.strip():
            raise ValueError(f"field '{key}[{i}]' must be a non-empty string")
        out.append(x.strip())
    return out
