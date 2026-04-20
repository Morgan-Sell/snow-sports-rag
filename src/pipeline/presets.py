from __future__ import annotations

from dataclasses import dataclass

__all__ = ["RetrievalPreset", "PRESETS", "resolve_preset"]

__doc__ = """User-facing retrieval presets.

The UI exposes three choices (Fast / Balanced / Deep) to keep the settings
panel clean. Each preset maps to the concrete knobs that would otherwise be
exposed as sliders: ``top_k``, ``l1_shortlist_m``, ``max_chunks_per_doc``,
``top_n_in`` (rerank input), ``top_k_out`` (rerank output).
"""


@dataclass(frozen=True)
class RetrievalPreset:
    """A named bundle of retrieval/rerank parameters.

    Attributes
    ----------
    name : str
        Human label displayed in the UI (``Fast``, ``Balanced``, ``Deep``).
    description : str
        One-line explanation shown as a caption under the radio.
    top_k : int
        Target L2 chunk count (before rerank).
    l1_top_m : int
        Number of documents to keep in the L1 shortlist.
    max_chunks_per_doc : int
        Upper bound on chunks from the same ``doc_id``.
    top_n_in : int
        Candidates fed into the reranker.
    top_k_out : int
        Final passages returned to the user (also number of citations).
    top_n_pre_rerank : int
        Cap on fused multi-query candidates before the reranker.
    """

    name: str
    description: str
    top_k: int
    l1_top_m: int
    max_chunks_per_doc: int
    top_n_in: int
    top_k_out: int
    top_n_pre_rerank: int


PRESETS: dict[str, RetrievalPreset] = {
    "Fast": RetrievalPreset(
        name="Fast",
        description="Snappy answers, narrower context.",
        top_k=4,
        l1_top_m=3,
        max_chunks_per_doc=2,
        top_n_in=12,
        top_k_out=3,
        top_n_pre_rerank=24,
    ),
    "Balanced": RetrievalPreset(
        name="Balanced",
        description="Default: good quality/latency trade-off.",
        top_k=8,
        l1_top_m=5,
        max_chunks_per_doc=2,
        top_n_in=30,
        top_k_out=5,
        top_n_pre_rerank=48,
    ),
    "Deep": RetrievalPreset(
        name="Deep",
        description="More documents, more chunks, slower.",
        top_k=12,
        l1_top_m=8,
        max_chunks_per_doc=3,
        top_n_in=48,
        top_k_out=8,
        top_n_pre_rerank=80,
    ),
}


def resolve_preset(name: str | None) -> RetrievalPreset:
    """Return the preset for ``name`` or :data:`PRESETS['Balanced']`.

    Parameters
    ----------
    name : str or None
        Case-insensitive preset label. Unknown or missing values fall back
        to ``Balanced`` instead of raising, so the UI cannot wedge the
        pipeline with a bad selection.

    Returns
    -------
    RetrievalPreset
        One of the entries in :data:`PRESETS`.
    """
    if not name:
        return PRESETS["Balanced"]
    for key, preset in PRESETS.items():
        if key.lower() == name.strip().lower():
            return preset
    return PRESETS["Balanced"]
