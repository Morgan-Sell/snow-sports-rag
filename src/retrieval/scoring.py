from __future__ import annotations

__all__ = ["chroma_cosine_distance_to_similarity"]

__doc__ = """Similarity helpers for Chroma cosine-space distances."""


def chroma_cosine_distance_to_similarity(distance: float) -> float:
    """Convert Chroma cosine **distance** to a **similarity** score for ranking.

    For L2-normalized embeddings and Chroma's cosine space, distance is
    typically ``1 - cos_sim``; therefore ``similarity = 1.0 - distance`` recovers
    cosine similarity in ``[0, 1]`` when vectors are unit length.

    Parameters
    ----------
    distance : float
        Value returned by Chroma for the ``cosine`` space (smaller is closer).

    Returns
    -------
    float
        ``1.0 - distance`` for downstream ranking and logging.
    """
    return 1.0 - float(distance)
