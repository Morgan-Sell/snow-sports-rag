from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import replace
from typing import Any

from ..retrieval.models import RetrievalHit
from .listwise_utils import parse_ranked_indices, passage_snippet

__all__ = ["OpenAIListwiseReranker"]

__doc__ = """Listwise passage reranking via OpenAI-compatible chat API."""


def _build_prompt(query: str, hits: list[RetrievalHit]) -> str:
    """Format numbered passages and instructions for the listwise chat user message.

    Parameters
    ----------
    query : str
        Original search string.
    hits : list of RetrievalHit
        Candidates to rank (order defines stable indices).

    Returns
    -------
    str
        Multi-line user prompt including JSON response instructions.
    """
    lines = [
        "Rank these passages by relevance to the query.",
        'Return ONLY JSON: {"ranked_indices": [<int>, ...]} — a permutation '
        "of all indices 0..n-1 from most to least relevant.",
        "",
        f"Query: {query.strip()}",
        "",
        "Passages:",
    ]
    for i, h in enumerate(hits):
        meta = f"{h.doc_id} | {h.section_path}".strip()
        lines.append(f"[{i}] ({meta}) {passage_snippet(h.text)}")
    return "\n".join(lines)


class OpenAIListwiseReranker:
    """One chat call asks the model to emit a full ranking of candidate indices.

    Uses the OpenAI-compatible ``/v1/chat/completions`` endpoint (same as
    :mod:`snow_sports_rag.llm.openai_compatible`).

    Parameters
    ----------
    model : str
        Chat model id (e.g. ``gpt-4o-mini``).
    api_key : str or None
        Bearer token; if None, read from ``api_key_env``.
    base_url : str
        API root without trailing slash.
    temperature : float
        Sampling temperature (low recommended).
    api_key_env : str
        Environment variable for API key when ``api_key`` is unset.
    timeout_s : float
        HTTP timeout.
    """

    def __init__(
        self,
        *,
        model: str,
        api_key: str | None = None,
        base_url: str = "https://api.openai.com/v1",
        temperature: float = 0.0,
        api_key_env: str = "OPENAI_API_KEY",
        timeout_s: float = 120.0,
    ) -> None:
        """Store OpenAI-compatible endpoint settings (no network I/O yet).

        Parameters
        ----------
        model : str
            Chat completions model id.
        api_key : str or None, optional
            Static bearer token; if omitted, :meth:`_resolve_api_key` reads env.
        base_url : str, optional
            API root (no trailing slash).
        temperature : float, optional
            Sampling temperature for the ranking call.
        api_key_env : str, optional
            Environment variable name consulted when ``api_key`` is unset.
        timeout_s : float, optional
            Per-request HTTP timeout in seconds.
        """
        self._model = model
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._temperature = float(temperature)
        self._api_key_env = api_key_env
        self._timeout_s = float(timeout_s)

    def _resolve_api_key(self) -> str:
        """Return the bearer token from the constructor or environment.

        Returns
        -------
        str
            Non-empty API key.

        Raises
        ------
        ValueError
            If neither ``api_key`` nor the configured env var is set.
        """
        if self._api_key is not None and str(self._api_key).strip():
            return str(self._api_key).strip()
        env = os.environ.get(self._api_key_env, "")
        if not env.strip():
            msg = f"Missing API key: set {self._api_key_env!r} or pass api_key="
            raise ValueError(msg)
        return env.strip()

    def rerank(
        self,
        query: str,
        hits: list[RetrievalHit],
        *,
        top_k: int,
    ) -> list[RetrievalHit]:
        """Ask the chat model for a permutation of indices; return top rows.

        Parameters
        ----------
        query : str
            Same query string used for retrieval.
        hits : list of RetrievalHit
            Candidate list (typically truncated to ``rerank.top_n_in``).
        top_k : int
            How many reranked rows to return.

        Returns
        -------
        list of RetrievalHit
            Reordered hits; ``similarity`` encodes listwise rank (higher is better).

        Raises
        ------
        ValueError
            On HTTP failures, malformed JSON, or unexpected API payloads.
        """
        if not hits:
            return []
        k = max(1, int(top_k))
        n = len(hits)
        system = (
            "You are a search relevance expert. Output valid JSON only, "
            "no markdown fences."
        )
        user = _build_prompt(query, hits)
        body: dict[str, Any] = {
            "model": self._model,
            "temperature": self._temperature,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        url = f"{self._base_url}/chat/completions"
        req = urllib.request.Request(
            url,
            data=json.dumps(body).encode("utf-8"),
            method="POST",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._resolve_api_key()}",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=self._timeout_s) as resp:
                raw = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", errors="replace")[:500]
            msg = f"OpenAI rerank HTTP {e.code}: {detail}"
            raise ValueError(msg) from e

        try:
            content = raw["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as e:
            msg = f"Unexpected chat completions shape: {raw!r}"
            raise ValueError(msg) from e

        order = parse_ranked_indices(str(content), n_passages=n)
        if len(order) < n:
            rest = [i for i in range(n) if i not in order]
            order = order + rest
        rank_score = float(n)
        out: list[RetrievalHit] = []
        for idx in order[:k]:
            hit = hits[idx]
            out.append(
                replace(
                    hit,
                    similarity=rank_score,
                    distance=-rank_score,
                )
            )
            rank_score -= 1.0
        return out
