from __future__ import annotations

import json
import re
from typing import Any

__all__ = ["passage_snippet", "parse_ranked_indices"]

_JSON_OBJECT = re.compile(r"\{[\s\S]*\}")


def passage_snippet(text: str, max_chars: int = 400) -> str:
    """Collapse whitespace and truncate ``text`` for compact LLM prompts.

    Parameters
    ----------
    text : str
        Full passage body.
    max_chars : int, optional
        Maximum output length; an ellipsis is appended when truncating.

    Returns
    -------
    str
        Single-line snippet safe to embed in ranking instructions.
    """
    t = " ".join(text.strip().split())
    if len(t) <= max_chars:
        return t
    return t[: max_chars - 1] + "…"


def parse_ranked_indices(content: str, *, n_passages: int) -> list[int]:
    """Parse ``ranked_indices`` from model JSON output.

    Parameters
    ----------
    content : str
        Assistant text containing a JSON object with a ``ranked_indices`` list.
    n_passages : int
        Number of candidate passages; indices must fall in ``[0, n_passages)``.

    Returns
    -------
    list of int
        Zero-based indices in model preference order (most relevant first).
        Invalid or duplicate indices are skipped.

    Raises
    ------
    ValueError
        If JSON is missing, malformed, or lacks the expected structure.
    """
    raw = content.strip()
    m = _JSON_OBJECT.search(raw)
    if not m:
        msg = f"Expected JSON object in reranker response, got: {raw[:300]!r}"
        raise ValueError(msg)
    obj: Any = json.loads(m.group())
    if not isinstance(obj, dict):
        msg = f"JSON root must be object, got {type(obj)}"
        raise ValueError(msg)
    arr = obj.get("ranked_indices")
    if not isinstance(arr, list):
        msg = "JSON must contain key 'ranked_indices' with a list value"
        raise ValueError(msg)
    out: list[int] = []
    seen: set[int] = set()
    for x in arr:
        if isinstance(x, bool):
            continue
        if isinstance(x, float) and x == int(x):
            xi = int(x)
        elif isinstance(x, int):
            xi = x
        else:
            continue
        if xi < 0 or xi >= n_passages or xi in seen:
            continue
        seen.add(xi)
        out.append(xi)
    return out
