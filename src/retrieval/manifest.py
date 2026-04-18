from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from ..embedding.model import EmbeddingModel

__all__ = ["ManifestReadableStore", "validate_embedder_against_manifest"]

__doc__ = """Optional embedding manifest checks for index/query parity."""


@runtime_checkable
class ManifestReadableStore(Protocol):
    """Stores that expose :meth:`read_embedding_manifest`."""

    def read_embedding_manifest(self) -> dict[str, Any] | None:
        """Load ``embedding_manifest.json`` adjacent to the vector store, if present.

        Returns
        -------
        dict or None
            Parsed JSON with at least ``model_name`` / ``dimension`` keys, or
            ``None`` when no manifest file exists.
        """
        ...


def validate_embedder_against_manifest(
    embedder: EmbeddingModel,
    manifest: dict[str, Any] | None,
) -> None:
    """Ensure ``embedder`` matches a non-null index manifest.

    Parameters
    ----------
    embedder : EmbeddingModel
        Model used to embed the query.
    manifest : dict or None
        Parsed ``embedding_manifest.json`` contents, or ``None`` if absent.

    Raises
    ------
    ValueError
        If ``manifest`` is present but ``model_name`` or ``dimension`` disagrees
        with ``embedder``.
    """
    if manifest is None:
        return
    exp_name = str(manifest.get("model_name", ""))
    if exp_name and exp_name != embedder.model_name:
        msg = (
            f"embedder model_name {embedder.model_name!r} does not match "
            f"index manifest {exp_name!r}"
        )
        raise ValueError(msg)
    exp_dim = manifest.get("dimension")
    if exp_dim is not None and int(exp_dim) != embedder.dimension:
        msg = (
            f"embedder dimension {embedder.dimension} does not match "
            f"index manifest dimension {exp_dim!r}"
        )
        raise ValueError(msg)
