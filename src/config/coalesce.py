"""Small config coalescing helpers.

Merged YAML defaults often set optional keys to explicit ``None`` (meaning
"inherit"). :meth:`dict.get` does not fall back when the key is present, so
call sites must coalesce explicitly.
"""

from __future__ import annotations

from typing import Any, Mapping

__all__ = ["coalesce_openai_timeout_s"]


def coalesce_openai_timeout_s(
    section: Mapping[str, Any],
    llm: Mapping[str, Any],
    *,
    default: float = 120.0,
) -> float:
    """Resolve OpenAI HTTP timeout from ``openai_timeout_s`` or ``llm.timeout_s``.

    Parameters
    ----------
    section : Mapping[str, Any]
        ``generation`` or ``rerank`` subsection; may contain ``openai_timeout_s``.
    llm : Mapping[str, Any]
        Shared LLM config; may contain ``timeout_s``.
    default : float, optional
        Used when both resolved values are missing or ``None``.

    Returns
    -------
    float
        Positive timeout in seconds.
    """
    raw = section.get("openai_timeout_s")
    if raw is not None:
        return float(raw)
    raw = llm.get("timeout_s")
    if raw is not None:
        return float(raw)
    return float(default)
