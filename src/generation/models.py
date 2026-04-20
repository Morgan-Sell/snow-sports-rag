from __future__ import annotations

from dataclasses import dataclass, field

__all__ = ["SourceCitation", "GeneratedAnswer"]

__doc__ = """Dataclasses for Phase 2.4 grounded answer generation."""


@dataclass(frozen=True)
class SourceCitation:
    """One passage rendered into the prompt as evidence.

    Attributes
    ----------
    index : int
        1-based citation marker (``[1]``, ``[2]``, ...) shown to the LLM.
    chunk_id : str
        Vector store row id of the passage.
    doc_id : str
        Source document id under the knowledge base root.
    section_path : str
        Human-readable section label (may be empty).
    chunk_index : int
        Zero-based chunk index within the source document.
    similarity : float
        Retrieval / rerank score for logging (higher is better).
    text : str
        (Possibly truncated) passage body placed in the context block.
    """

    index: int
    chunk_id: str
    doc_id: str
    section_path: str
    chunk_index: int
    similarity: float
    text: str


@dataclass(frozen=True)
class GeneratedAnswer:
    """Final user-facing reply plus structured provenance for UI / eval.

    Attributes
    ----------
    answer : str
        Natural language reply produced by the LLM (may be a refusal).
    citations : list of SourceCitation
        Passages passed into the prompt, in citation order.
    refused : bool
        ``True`` when the generator believes the KB does not support an answer
        (matched ``refusal_message`` or empty context).
    backend : str
        Concrete generator backend that produced the answer
        (``openai``, ``anthropic``, ``huggingface``, ``fake``).
    model : str
        Model identifier reported by the backend (may be empty for local).
    usage : dict
        Free-form provider metadata (token counts when available).
    """

    answer: str
    citations: list[SourceCitation]
    refused: bool
    backend: str
    model: str
    usage: dict = field(default_factory=dict)
