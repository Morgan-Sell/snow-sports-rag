from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping, Protocol

from ..vectorstore.models import VectorQueryHit
from .models import RetrievalHit
from .scoring import chroma_cosine_distance_to_similarity

__all__ = [
    "ExpansionRequest",
    "SupportsGetByDocId",
    "expand_retrieval_hits",
    "plan_expansion_requests",
]

__doc__ = """Post-retrieval document expansion helpers."""


ExpansionMode = Literal["neighbors", "same_section", "anchor_sections"]


class SupportsGetByDocId(Protocol):
    """Vector-store read surface needed for document expansion."""

    def get_by_doc_id(self, doc_id: str) -> list[VectorQueryHit]:
        """Return all indexed L2 chunks for ``doc_id``."""
        ...


@dataclass(frozen=True)
class ExpansionRequest:
    """One bounded same-document expansion request."""

    mode: ExpansionMode
    doc_id: str
    chunk_index: int | None = None
    section_path: str | None = None
    section_paths: tuple[str, ...] = ()
    reason: str = ""


_FIELD_TERMS = (
    "home resort",
    "hometown",
    "born",
    "discipline",
    "sponsor",
    "sponsors",
    "train",
    "training",
    "venue",
    "located",
    "hosts",
    "hosted",
    "sanctioning",
    "frequency",
)


def _entity_type_from_doc_id(doc_id: str) -> str:
    return doc_id.split("/", 1)[0] if "/" in doc_id else ""


def _as_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, (list, tuple)):
        return tuple(str(x) for x in value if str(x).strip())
    return ()


def _query_needs_anchor(query: str) -> bool:
    q = query.lower()
    return any(t in q for t in _FIELD_TERMS)


def plan_expansion_requests(
    hit: RetrievalHit,
    query: str,
    *,
    modes: tuple[str, ...],
    window: int,
    anchor_sections_by_entity_type: Mapping[str, Any],
) -> list[ExpansionRequest]:
    """Plan same-document expansion routes for one retrieved hit.

    The planner intentionally returns multiple routes: neighbors work for all
    chunking strategies, same-section uses Markdown-header metadata when
    present, and anchors recover entity facts such as athlete ``Summary`` rows.
    """
    out: list[ExpansionRequest] = []
    enabled = {m.strip().lower() for m in modes}
    if "neighbors" in enabled and hit.chunk_index >= 0 and window > 0:
        out.append(
            ExpansionRequest(
                mode="neighbors",
                doc_id=hit.doc_id,
                chunk_index=hit.chunk_index,
                reason="local_context",
            )
        )

    if "same_section" in enabled and hit.section_path:
        out.append(
            ExpansionRequest(
                mode="same_section",
                doc_id=hit.doc_id,
                section_path=hit.section_path,
                reason="same_section",
            )
        )

    if "anchor_sections" in enabled and _query_needs_anchor(query):
        entity_type = _entity_type_from_doc_id(hit.doc_id)
        anchors = _as_tuple(anchor_sections_by_entity_type.get(entity_type))
        if anchors:
            out.append(
                ExpansionRequest(
                    mode="anchor_sections",
                    doc_id=hit.doc_id,
                    section_paths=anchors,
                    reason=f"{entity_type}_anchor_sections",
                )
            )
    return out


def _vector_to_retrieval(hit: VectorQueryHit) -> RetrievalHit:
    meta = hit.metadata
    raw_ci = meta.get("chunk_index", -1)
    chunk_index = int(raw_ci) if raw_ci is not None else -1
    return RetrievalHit(
        chunk_id=hit.id,
        text=hit.document,
        doc_id=str(meta.get("doc_id", "")),
        section_path=str(meta.get("section_path", "")),
        chunk_index=chunk_index,
        similarity=chroma_cosine_distance_to_similarity(hit.distance),
        distance=hit.distance,
    )


def _matches_request(
    candidate: RetrievalHit,
    req: ExpansionRequest,
    *,
    window: int,
) -> bool:
    if candidate.doc_id != req.doc_id:
        return False
    if req.mode == "neighbors":
        if req.chunk_index is None or candidate.chunk_index < 0:
            return False
        return 0 < abs(candidate.chunk_index - req.chunk_index) <= window
    if req.mode == "same_section":
        return bool(req.section_path) and candidate.section_path == req.section_path
    if req.mode == "anchor_sections":
        return candidate.section_path in set(req.section_paths)
    return False


def _penalized_hit(
    candidate: RetrievalHit,
    seed: RetrievalHit,
    penalty: float,
) -> RetrievalHit:
    sim = float(seed.similarity) - float(penalty)
    dist = float(seed.distance) + float(penalty)
    return RetrievalHit(
        chunk_id=candidate.chunk_id,
        text=candidate.text,
        doc_id=candidate.doc_id,
        section_path=candidate.section_path,
        chunk_index=candidate.chunk_index,
        similarity=sim,
        distance=dist,
    )


def expand_retrieval_hits(
    hits: list[RetrievalHit],
    *,
    query: str,
    store: SupportsGetByDocId,
    config: Mapping[str, Any],
) -> tuple[list[RetrievalHit], list[RetrievalHit], list[ExpansionRequest]]:
    """Expand fused retrieval hits with same-document sibling/anchor chunks.

    Returns
    -------
    tuple
        ``(expanded_hits, added_hits, requests)``. ``expanded_hits`` preserves
        original order and inserts added chunks immediately after their seed.
    """
    if not bool(config.get("enabled", False)) or not hits:
        return hits, [], []

    modes = _as_tuple(config.get("modes", ("neighbors",)))
    window = max(0, int(config.get("window", 1)))
    max_per_doc = max(0, int(config.get("max_extra_chunks_per_doc", 3)))
    max_total = max(0, int(config.get("max_total_extra_chunks", 8)))
    max_seed_hits = max(1, int(config.get("max_seed_hits", len(hits))))
    penalty = float(config.get("expansion_score_penalty", 0.05))
    anchors = config.get("anchor_sections_by_entity_type", {})
    if not isinstance(anchors, Mapping):
        anchors = {}

    doc_cache: dict[str, list[RetrievalHit]] = {}
    requests_by_seed: dict[str, list[ExpansionRequest]] = {}
    all_requests: list[ExpansionRequest] = []
    for seed in hits[:max_seed_hits]:
        reqs = plan_expansion_requests(
            seed,
            query,
            modes=modes,
            window=window,
            anchor_sections_by_entity_type=anchors,
        )
        requests_by_seed[seed.chunk_id] = reqs
        all_requests.extend(reqs)
        if reqs and seed.doc_id not in doc_cache:
            doc_cache[seed.doc_id] = [
                _vector_to_retrieval(vh) for vh in store.get_by_doc_id(seed.doc_id)
            ]

    seen = {h.chunk_id for h in hits}
    added: list[RetrievalHit] = []
    expanded: list[RetrievalHit] = []
    per_doc_counts: dict[str, int] = {}

    for seed in hits:
        expanded.append(seed)
        if len(added) >= max_total:
            continue
        reqs = requests_by_seed.get(seed.chunk_id, [])
        if not reqs:
            continue
        doc_hits = doc_cache.get(seed.doc_id, [])
        for req in reqs:
            if len(added) >= max_total:
                break
            for cand in doc_hits:
                if cand.chunk_id in seen:
                    continue
                if per_doc_counts.get(cand.doc_id, 0) >= max_per_doc:
                    continue
                if not _matches_request(cand, req, window=window):
                    continue
                new_hit = _penalized_hit(cand, seed, penalty)
                expanded.append(new_hit)
                added.append(new_hit)
                seen.add(new_hit.chunk_id)
                per_doc_counts[new_hit.doc_id] = (
                    per_doc_counts.get(new_hit.doc_id, 0) + 1
                )
                if len(added) >= max_total:
                    break

    return expanded, added, all_requests
