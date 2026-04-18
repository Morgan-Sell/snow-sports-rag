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
        self._prefix = prefix

    def expand_query(self, query: str, *, num_paraphrases: int = 3) -> list[str]:
        """Build ``num_paraphrases`` strings derived from ``query``."""
        n = max(0, int(num_paraphrases))
        q = query.strip()
        if not q or n == 0:
            return []
        return [f"{self._prefix} {q} [{i}]".strip() for i in range(n)]
