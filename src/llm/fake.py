from __future__ import annotations

__all__ = ["FakeLLMClient"]

__doc__ = """Deterministic LLM stub for tests and offline pipelines."""


class FakeLLMClient:
    """Returns predictable paraphrases without network calls.

    Parameters
    ----------
    prefix : str, default 'paraphrase:'
        Prepended to each synthetic variant (after the original query text).
    """

    def __init__(self, *, prefix: str = "paraphrase:") -> None:
        """Configure the deterministic paraphrase prefix.

        Parameters
        ----------
        prefix : str, optional
            Literal prefix inserted before each synthetic variant.
        """
        self._prefix = prefix

    def expand_query(self, query: str, *, num_paraphrases: int = 3) -> list[str]:
        """Return ``num_paraphrases`` synthetic strings derived from ``query``.

        Parameters
        ----------
        query : str
            Original user text (stripped before use).
        num_paraphrases : int, optional
            Number of variants to emit; zero yields an empty list.

        Returns
        -------
        list of str
            Deterministic strings ``{prefix} {query} [i]`` for ``i`` in range.
        """
        n = max(0, int(num_paraphrases))
        q = query.strip()
        if not q or n == 0:
            return []
        return [f"{self._prefix} {q} [{i}]".strip() for i in range(n)]
