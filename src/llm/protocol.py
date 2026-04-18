from __future__ import annotations

from typing import Protocol, runtime_checkable

__all__ = ["LLMClient"]

__doc__ = """Protocol for LLM-backed RAG helpers (expansion, later generation)."""


@runtime_checkable
class LLMClient(Protocol):
    """Minimal LLM surface for Phase 2.2 query expansion.

    Implementations return *additional* phrasings; callers merge with the
    original user query for multi-query retrieval.
    """

    def expand_query(self, query: str, *, num_paraphrases: int = 3) -> list[str]:
        """Produce up to ``num_paraphrases`` paraphrases of ``query``.

        Parameters
        ----------
        query : str
            Original user question or search string.
        num_paraphrases : int, optional
            Maximum number of distinct paraphrases to return (not counting
            ``query`` itself).

        Returns
        -------
        list of str
            Zero or more paraphrases; empty list disables multi-query beyond
            the original string. Entries should be stripped and non-empty;
            duplicates may be filtered by the caller.
        """
        ...
