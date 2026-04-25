from __future__ import annotations

import hashlib
import json
import threading
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from ..retrieval.models import RetrievalHit
from .models import PipelineResult

__all__ = [
    "compute_config_hash",
    "new_trace_id",
    "TraceLogger",
]

__doc__ = """Append-only JSONL trace logging for later offline analysis."""


_SENSITIVE_KEYS = frozenset(
    {
        "api_key",
        "openai_api_key",
        "anthropic_api_key",
        "hf_token",
    }
)


def _strip_secrets(section: Mapping[str, Any] | Any) -> Any:
    """Drop obvious API keys from a config mapping before hashing.

    Parameters
    ----------
    section : Mapping[str, Any] or Any
        Config subsection; non-mappings are returned unchanged.

    Returns
    -------
    Any
        Shallow-cleaned mapping (nested mappings cleaned recursively) with
        keys in :data:`_SENSITIVE_KEYS` removed.
    """
    if not isinstance(section, Mapping):
        return section
    out: dict[str, Any] = {}
    for k, v in section.items():
        if k.lower() in _SENSITIVE_KEYS:
            continue
        out[k] = _strip_secrets(v) if isinstance(v, Mapping) else v
    return out


def compute_config_hash(cfg: Any) -> str:
    """Return a short, stable hash of the query-affecting config subset.

    Only sections that materially change retrieval / generation behaviour are
    included (chunking, embedding, vector_store, retrieval, rerank,
    query_expansion, document_expansion, generation). Secrets are stripped.

    Parameters
    ----------
    cfg : Any
        An :class:`~snow_sports_rag.config.AppConfig` or a plain dict-like
        object with the same attribute / key names.

    Returns
    -------
    str
        12-hex-char SHA1 prefix.
    """

    def _get(key: str) -> Any:
        if isinstance(cfg, Mapping):
            return cfg.get(key, {})
        return getattr(cfg, key, {})

    relevant = {
        k: _strip_secrets(_get(k))
        for k in (
            "chunking",
            "embedding",
            "vector_store",
            "retrieval",
            "rerank",
            "query_expansion",
            "document_expansion",
            "generation",
        )
    }
    if isinstance(relevant.get("vector_store"), Mapping):
        vs = dict(relevant["vector_store"])
        pd = vs.get("persist_directory")
        if pd is not None:
            vs["persist_directory"] = str(pd)
        relevant["vector_store"] = vs

    blob = json.dumps(relevant, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha1(blob).hexdigest()[:12]


def new_trace_id() -> str:
    """Return an opaque 16-hex-char correlation id.

    Returns
    -------
    str
        Truncated uuid4 hex; unique enough for a single-user UI session.
    """
    return uuid.uuid4().hex[:16]


def _hit_to_dict(h: RetrievalHit) -> dict[str, Any]:
    """Serialize a :class:`RetrievalHit` to a trace-safe dict.

    Parameters
    ----------
    h : RetrievalHit
        Hit as produced by the retriever / reranker.

    Returns
    -------
    dict
        JSON-serializable subset (``text`` body is omitted to keep traces
        compact; chunk text is already reproducible from the index).
    """
    return {
        "chunk_id": h.chunk_id,
        "doc_id": h.doc_id,
        "section_path": h.section_path,
        "chunk_index": h.chunk_index,
        "similarity": round(float(h.similarity), 6),
    }


@dataclass
class TraceRecord:
    """One row in ``traces.jsonl``.

    Attributes
    ----------
    type : str
        ``"query"`` for pipeline runs, ``"feedback"`` for thumbs records.
    trace_id : str
        Correlation id. Feedback rows reuse the query's id.
    timestamp : str
        ISO 8601 UTC timestamp.
    config_hash : str
        Hash produced by :func:`compute_config_hash`.
    payload : dict
        Free-form content (different shapes for ``query`` vs ``feedback``).
    """

    type: str
    trace_id: str
    timestamp: str
    config_hash: str
    payload: dict[str, Any] = field(default_factory=dict)


class TraceLogger:
    """Append JSONL rows under a lock so concurrent Gradio handlers don't tear.

    Parameters
    ----------
    path : pathlib.Path
        Destination file; parent directories are created on first write.
    enabled : bool, default True
        When false, :meth:`log_query` / :meth:`log_feedback` are no-ops.
    """

    def __init__(self, path: Path, *, enabled: bool = True) -> None:
        """Record target path; defer any filesystem I/O until the first write.

        Parameters
        ----------
        path : pathlib.Path
            Absolute or relative path to the ``traces.jsonl`` file.
        enabled : bool, optional
            Short-circuit flag for tests or read-only environments.
        """
        self._path = Path(path)
        self._enabled = bool(enabled)
        self._lock = threading.Lock()

    @property
    def path(self) -> Path:
        """Return the JSONL file path regardless of ``enabled`` state.

        Returns
        -------
        pathlib.Path
            The constructor argument (resolved at write time).
        """
        return self._path

    def _append(self, record: TraceRecord) -> None:
        """Serialize and append a single :class:`TraceRecord` atomically.

        Parameters
        ----------
        record : TraceRecord
            Row to write; ``asdict`` turns it into a plain mapping.
        """
        if not self._enabled:
            return
        line = json.dumps(asdict(record), default=str)
        with self._lock:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with self._path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")

    def log_query(
        self,
        result: PipelineResult,
        *,
        preset: str,
        rerank_enabled: bool,
        expansion_enabled: bool,
        generation_enabled: bool,
    ) -> None:
        """Append a ``"query"`` row summarising a completed pipeline run.

        Parameters
        ----------
        result : PipelineResult
            Output of :meth:`RAGPipeline.run`.
        preset : str
            Preset label selected by the user.
        rerank_enabled, expansion_enabled, generation_enabled : bool
            Toggles that were active for the run; recorded verbatim for
            later bucketed analysis.
        """
        tr = result.trace
        payload: dict[str, Any] = {
            "query": result.query,
            "preset": preset,
            "rerank_enabled": bool(rerank_enabled),
            "expansion_enabled": bool(expansion_enabled),
            "generation_enabled": bool(generation_enabled),
            "expansions": list(tr.expansions),
            "variants": list(tr.variants),
            "l1_shortlist": list(tr.l1_shortlist),
            "l2_pre_rerank": [_hit_to_dict(h) for h in tr.l2_pre_rerank],
            "document_expansion_added": [
                _hit_to_dict(h) for h in tr.document_expansion_added
            ],
            "reranked": [_hit_to_dict(h) for h in tr.reranked],
            "final_sources": [
                {
                    "index": c.index,
                    "chunk_id": c.chunk_id,
                    "doc_id": c.doc_id,
                    "section_path": c.section_path,
                    "chunk_index": c.chunk_index,
                    "similarity": round(float(c.similarity), 6),
                }
                for c in result.cards
            ],
            "latency_ms": asdict(tr.latency),
        }
        if result.answer is not None:
            payload["answer"] = result.answer.answer
            payload["refused"] = bool(result.answer.refused)
            payload["backend"] = result.answer.backend
            payload["model"] = result.answer.model
        self._append(
            TraceRecord(
                type="query",
                trace_id=result.trace_id,
                timestamp=datetime.now(timezone.utc).isoformat(),
                config_hash=result.config_hash,
                payload=payload,
            )
        )

    def log_feedback(
        self,
        *,
        trace_id: str,
        config_hash: str,
        feedback: str,
    ) -> None:
        """Append a ``"feedback"`` row correlated to a prior query record.

        Parameters
        ----------
        trace_id : str
            Id returned by the earlier :meth:`log_query` call.
        config_hash : str
            Same hash as the query row, copied for easy offline joins.
        feedback : str
            ``"up"`` or ``"down"``; other strings are accepted and stored
            verbatim for future extension (e.g. free-text comments).
        """
        self._append(
            TraceRecord(
                type="feedback",
                trace_id=trace_id,
                timestamp=datetime.now(timezone.utc).isoformat(),
                config_hash=config_hash,
                payload={"feedback": feedback},
            )
        )
