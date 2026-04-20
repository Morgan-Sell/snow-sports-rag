from __future__ import annotations

from ..retrieval.models import RetrievalHit
from .models import SourceCitation

__all__ = [
    "DEFAULT_SYSTEM_PROMPT",
    "DEFAULT_REFUSAL_MESSAGE",
    "build_citations",
    "format_context_block",
    "build_user_prompt",
]

__doc__ = """System prompt + context formatting for KB-grounded generation.

The prompt is deliberately strict: the LLM must answer ONLY from the numbered
``[SOURCE n]`` blocks and must refuse when evidence is insufficient, so that
responses cannot silently leak general web knowledge.
"""


DEFAULT_REFUSAL_MESSAGE = "I don't know based on the provided knowledge base."


DEFAULT_SYSTEM_PROMPT = (
    "You are a retrieval-augmented assistant for a snow-sports knowledge "
    "base.\n"
    "\n"
    "STRICT GROUNDING RULES:\n"
    "1. Answer ONLY using facts stated in the numbered [SOURCE n] blocks "
    "provided in the user message.\n"
    "2. Do NOT use outside or general knowledge, even if you are confident.\n"
    "3. If the sources do not contain enough information to answer, reply "
    'EXACTLY with: "{refusal}"\n'
    "4. Cite every non-trivial claim with bracketed markers like [1], [2] "
    "matching the [SOURCE n] labels. Multiple citations are allowed: [1][3].\n"
    "5. Prefer concise, specific answers. Do not speculate, do not invent "
    "numbers, names, or section titles that are not in the sources.\n"
    "6. Quote short spans only when a direct quote adds clarity; paraphrase "
    "otherwise.\n"
)


def _truncate(text: str, limit: int) -> str:
    """Truncate ``text`` to ``limit`` characters with an ellipsis marker.

    Parameters
    ----------
    text : str
        Passage body as stored in the vector index.
    limit : int
        Maximum characters to keep (``<= 0`` disables truncation).

    Returns
    -------
    str
        ``text`` itself when short enough, otherwise a truncated copy ending
        with ``" …"``.
    """
    if limit <= 0 or len(text) <= limit:
        return text
    cut = max(0, limit - 2)
    return text[:cut].rstrip() + " …"


def build_citations(
    hits: list[RetrievalHit],
    *,
    max_chars_per_hit: int = 1200,
) -> list[SourceCitation]:
    """Number retrieval hits into :class:`SourceCitation` records.

    Parameters
    ----------
    hits : list of RetrievalHit
        Ranked passages in the order they should appear in the prompt.
    max_chars_per_hit : int, optional
        Per-passage character budget. ``<= 0`` keeps the full text.

    Returns
    -------
    list of SourceCitation
        One entry per input hit with 1-based ``index`` and truncated ``text``.
    """
    out: list[SourceCitation] = []
    for i, h in enumerate(hits, start=1):
        out.append(
            SourceCitation(
                index=i,
                chunk_id=h.chunk_id,
                doc_id=h.doc_id,
                section_path=h.section_path,
                chunk_index=h.chunk_index,
                similarity=h.similarity,
                text=_truncate(h.text, max_chars_per_hit),
            )
        )
    return out


def format_context_block(
    citations: list[SourceCitation],
    *,
    include_section_path: bool = True,
) -> str:
    """Render citations as numbered ``[SOURCE n]`` evidence blocks.

    Parameters
    ----------
    citations : list of SourceCitation
        Output of :func:`build_citations`.
    include_section_path : bool, optional
        Whether to show ``section_path`` alongside ``doc_id`` in the header.

    Returns
    -------
    str
        Multi-line string; empty when ``citations`` is empty.
    """
    if not citations:
        return ""
    parts: list[str] = []
    for c in citations:
        header = f"[SOURCE {c.index}] doc_id={c.doc_id}"
        if include_section_path and c.section_path:
            header += f" | section={c.section_path}"
        parts.append(f"{header}\n{c.text}")
    return "\n\n".join(parts)


def build_user_prompt(
    query: str,
    citations: list[SourceCitation],
    *,
    include_section_path: bool = True,
) -> str:
    """Assemble the user message shown to the LLM (context + question).

    Parameters
    ----------
    query : str
        Original user question (stripped).
    citations : list of SourceCitation
        Numbered evidence to include verbatim in the prompt.
    include_section_path : bool, optional
        Passed through to :func:`format_context_block`.

    Returns
    -------
    str
        Prompt body; when ``citations`` is empty, a short no-context notice so
        the system-prompt refusal rule triggers cleanly.
    """
    q = query.strip()
    if not citations:
        return (
            "CONTEXT:\n(no sources retrieved)\n\n"
            f"QUESTION: {q}\n\n"
            "If the CONTEXT is empty, refuse per the grounding rules."
        )
    ctx = format_context_block(citations, include_section_path=include_section_path)
    return f"CONTEXT:\n{ctx}\n\nQUESTION: {q}\n"
