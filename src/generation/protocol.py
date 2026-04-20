from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..retrieval.models import RetrievalHit
from .models import GeneratedAnswer

__all__ = ["AnswerGenerator"]

__doc__ = """Protocol for Phase 2.4 grounded answer generators."""


@runtime_checkable
class AnswerGenerator(Protocol):
    """Produce a cited answer strictly from a list of retrieved passages.

    Implementations must not rely on world knowledge beyond ``hits``; the
    system prompt should instruct the model to refuse when the context is
    insufficient.
    """

    def generate(
        self,
        query: str,
        hits: list[RetrievalHit],
    ) -> GeneratedAnswer:
        """Return a grounded answer or a refusal.

        Parameters
        ----------
        query : str
            Natural language user question.
        hits : list of RetrievalHit
            Context passages (already retrieved / reranked / sliced by the
            caller). The generator numbers them 1..N for citation.

        Returns
        -------
        GeneratedAnswer
            Reply plus structured citations and provider metadata.
        """
        ...
