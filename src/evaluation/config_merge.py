"""Merge dotted-key overrides into :class:`~snow_sports_rag.config.loader.AppConfig`."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping

from ..config.loader import AppConfig

__all__ = [
    "flat_overrides_to_nested",
    "merge_app_config_overrides",
    "merged_config_from_flat",
]


def flat_overrides_to_nested(flat: Mapping[str, Any]) -> dict[str, Any]:
    """Turn ``chunking.strategy``-style keys into nested dicts.

    Parameters
    ----------
    flat : Mapping[str, Any]
        Keys use dot notation for nesting; values are scalars or shallow JSON
        values (lists, etc.).

    Returns
    -------
    dict
        Tree suitable for merging into config subsections.
    """
    root: dict[str, Any] = {}
    for dotted, val in flat.items():
        parts = dotted.split(".")
        cur: Any = root
        for p in parts[:-1]:
            nxt = cur.get(p)
            if nxt is None:
                nxt = {}
                cur[p] = nxt
            if not isinstance(nxt, dict):
                msg = f"cannot nest under {dotted!r}: {p!r} is not a mapping"
                raise ValueError(msg)
            cur = nxt
        cur[parts[-1]] = val
    return root


def merge_app_config_overrides(
    base: AppConfig,
    nested: Mapping[str, Any],
) -> AppConfig:
    """Shallow-merge each subsection from ``nested`` into ``base``.

    Unknown top-level keys in ``nested`` raise ``ValueError``.

    Parameters
    ----------
    base : AppConfig
        Starting configuration (typically from
        :func:`~snow_sports_rag.config.load_config`).
    nested : Mapping[str, Any]
        Top-level keys are subsection names (``chunking``, ``embedding``, …);
        values are dicts merged into the corresponding base subsection.

    Returns
    -------
    AppConfig
        New frozen config instance.
    """
    allowed = frozenset(
        {
            "chunking",
            "embedding",
            "vector_store",
            "retrieval",
            "rerank",
            "llm",
            "query_expansion",
            "document_expansion",
            "generation",
            "logging",
        }
    )
    unknown = set(nested.keys()) - allowed
    if unknown:
        raise ValueError(f"unknown config subsection(s): {sorted(unknown)}")

    def merge(name: str) -> dict[str, Any]:
        b = dict(getattr(base, name))
        u = nested.get(name)
        if u is None:
            return b
        if not isinstance(u, dict):
            raise TypeError(f"override for {name!r} must be a mapping")
        return {**b, **u}

    return replace(
        base,
        chunking=merge("chunking"),
        embedding=merge("embedding"),
        vector_store=merge("vector_store"),
        retrieval=merge("retrieval"),
        rerank=merge("rerank"),
        llm=merge("llm"),
        query_expansion=merge("query_expansion"),
        document_expansion=merge("document_expansion"),
        generation=merge("generation"),
        logging=merge("logging"),
    )


def merged_config_from_flat(base: AppConfig, flat: Mapping[str, Any]) -> AppConfig:
    """Apply dotted-key overrides and return a new :class:`AppConfig`."""
    nested = flat_overrides_to_nested(flat)
    return merge_app_config_overrides(base, nested)
