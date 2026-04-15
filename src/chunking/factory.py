from __future__ import annotations

from typing import Any, Mapping

from .strategies import (
    ChunkStrategy,
    FixedWindowChunkStrategy,
    MarkdownHeaderChunkStrategy,
    RecursiveCharChunkStrategy,
)


def chunk_strategy_from_config(chunking: Mapping[str, Any]) -> ChunkStrategy:
    """Instantiate a chunking strategy from a config subsection.

    Parameters
    ----------
    chunking : Mapping[str, Any]
        Typically :attr:`~snow_sports_rag.config.loader.AppConfig.chunking`.
        Expected keys:

        - ``strategy`` : str — ``markdown_header``, ``recursive_character`` (or
          ``recursive_char``), ``fixed_window`` (or ``fixed_size``); hyphens are
          normalized to underscores.
        - ``chunk_size`` : int — maximum chunk length in characters.
        - ``chunk_overlap`` : int — overlap between consecutive windows or merged
          splits; must satisfy ``0 <= chunk_overlap <= chunk_size``.
        - ``min_section_chars`` : int, optional — for ``markdown_header``, skip
          sections whose body is shorter than this (default ``0``).
        - ``recursive_separators`` : list of str, optional — for recursive
          strategies; defaults are applied when omitted.

    Returns
    -------
    ChunkStrategy
        Concrete splitter implementing the protocol ``chunk`` method.

    Raises
    ------
    KeyError
        If ``chunk_size`` or ``chunk_overlap`` is missing.
    TypeError
        If ``recursive_separators`` is provided and is not a ``list``.
    ValueError
        If ``strategy`` is unknown, or if ``chunk_size`` / ``chunk_overlap`` are
        invalid when the strategy validates them.
    """
    raw = str(chunking.get("strategy", "markdown_header")).strip().lower()
    name = raw.replace("-", "_")
    chunk_size = int(chunking["chunk_size"])
    chunk_overlap = int(chunking["chunk_overlap"])

    if name == "markdown_header":
        min_section = int(chunking.get("min_section_chars", 0))
        return MarkdownHeaderChunkStrategy(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            min_section_chars=min_section,
        )
    if name in ("recursive_character", "recursive_char"):
        seps = chunking.get("recursive_separators")
        if seps is not None and not isinstance(seps, list):
            msg = "recursive_separators must be a list of strings when set"
            raise TypeError(msg)
        str_seps = [str(x) for x in seps] if seps is not None else None
        return RecursiveCharChunkStrategy(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=str_seps,
        )
    if name in ("fixed_window", "fixed_size"):
        return FixedWindowChunkStrategy(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
    msg = f"Unknown chunking strategy: {raw!r}"
    raise ValueError(msg)
