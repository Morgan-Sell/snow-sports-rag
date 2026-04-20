from __future__ import annotations

from ..retrieval.models import RetrievalHit
from .models import GeneratedAnswer
from .prompt import DEFAULT_REFUSAL_MESSAGE, build_citations

__all__ = ["FakeAnswerGenerator"]

__doc__ = """Deterministic generator for tests and offline pipelines."""


class FakeAnswerGenerator:
    """Echo the query and cite passages without any network / model call.

    Parameters
    ----------
    refusal_message : str, optional
        Returned verbatim (with ``refused=True``) when ``hits`` is empty.
    max_chars_per_hit : int, optional
        Context truncation budget passed to :func:`build_citations`.
    """

    def __init__(
        self,
        *,
        refusal_message: str = DEFAULT_REFUSAL_MESSAGE,
        max_chars_per_hit: int = 1200,
    ) -> None:
        """Persist refusal string and context-truncation budget.

        Parameters
        ----------
        refusal_message : str, optional
            Used when the caller supplies zero evidence.
        max_chars_per_hit : int, optional
            Per-passage char limit forwarded to citation builder.
        """
        self._refusal = refusal_message
        self._max_chars = int(max_chars_per_hit)

    def generate(
        self,
        query: str,
        hits: list[RetrievalHit],
    ) -> GeneratedAnswer:
        """Produce a stable, KB-grounded stub reply for tests.

        Parameters
        ----------
        query : str
            User question text.
        hits : list of RetrievalHit
            Retrieved passages; numbered 1..N in citation order.

        Returns
        -------
        GeneratedAnswer
            Refusal when ``hits`` is empty; otherwise an answer of the form
            ``"Based on sources [1][2][3]: {query}"``.
        """
        cits = build_citations(hits, max_chars_per_hit=self._max_chars)
        if not cits:
            return GeneratedAnswer(
                answer=self._refusal,
                citations=[],
                refused=True,
                backend="fake",
                model="fake",
                usage={},
            )
        markers = "".join(f"[{c.index}]" for c in cits)
        answer = f"Based on sources {markers}: {query.strip()}"
        return GeneratedAnswer(
            answer=answer,
            citations=cits,
            refused=False,
            backend="fake",
            model="fake",
            usage={},
        )
