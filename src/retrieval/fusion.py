from __future__ import annotations

from collections import defaultdict

from .models import RetrievalHit

__all__ = ["fuse_retrieval_hits_max_score", "fuse_retrieval_hits_rrf"]

__doc__ = """Merge ranked hit lists from multi-query dense retrieval (Phase 2.2)."""


def fuse_retrieval_hits_max_score(
    ranked_lists: list[list[RetrievalHit]],
    *,
    top_n: int,
) -> list[RetrievalHit]:
    """Keep, per ``chunk_id``, the hit with highest ``similarity``; sort by that score.

    Parameters
    ----------
    ranked_lists : list of list of RetrievalHit
        One ranked list per query variant (best-first within each list).
    top_n : int
        Maximum number of fused hits to return.

    Returns
    -------
    list of RetrievalHit
        Fused ordering by descending best similarity seen for each chunk.
    """
    cap = max(1, int(top_n))
    best: dict[str, RetrievalHit] = {}
    for lst in ranked_lists:
        for h in lst:
            cur = best.get(h.chunk_id)
            if cur is None or h.similarity > cur.similarity:
                best[h.chunk_id] = h
    fused = sorted(best.values(), key=lambda x: x.similarity, reverse=True)
    return fused[:cap]


def fuse_retrieval_hits_rrf(
    ranked_lists: list[list[RetrievalHit]],
    *,
    top_n: int,
    rrf_k: int = 60,
) -> list[RetrievalHit]:
    """Reciprocal-rank fusion across query-variant hit lists.

    Uses ``score(chunk) = sum_i 1 / (k + rank_i)`` where ``rank_i`` is the
    zero-based rank of the chunk in list ``i`` (skipped if absent). The
    returned :class:`RetrievalHit` row is the one with highest ``similarity``
    among occurrences.

    Parameters
    ----------
    ranked_lists : list of list of RetrievalHit
        One ranked list per query variant.
    top_n : int
        Truncate to this many rows after sorting by RRF score.
    rrf_k : int, optional
        RRF smoothing constant (typical values 50--60).

    Returns
    -------
    list of RetrievalHit
        Best-first by fused RRF score.
    """
    cap = max(1, int(top_n))
    k = max(1, int(rrf_k))
    scores: dict[str, float] = defaultdict(float)
    best_hit: dict[str, RetrievalHit] = {}
    for lst in ranked_lists:
        for rank, h in enumerate(lst):
            cid = h.chunk_id
            scores[cid] += 1.0 / (k + rank + 1)
            prev = best_hit.get(cid)
            if prev is None or h.similarity > prev.similarity:
                best_hit[cid] = h
    ordered_ids = sorted(scores.keys(), key=lambda cid: scores[cid], reverse=True)
    return [best_hit[cid] for cid in ordered_ids[:cap]]
