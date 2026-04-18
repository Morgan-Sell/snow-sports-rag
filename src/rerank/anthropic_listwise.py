from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import replace
from typing import Any

from ..retrieval.models import RetrievalHit
from .listwise_utils import parse_ranked_indices, passage_snippet

__all__ = ["AnthropicListwiseReranker"]

__doc__ = """Listwise passage reranking via Anthropic Messages API."""


def _build_user_block(query: str, hits: list[RetrievalHit]) -> str:
    """Build the sole user message for Anthropic listwise reranking.

    Parameters
    ----------
    query : str
        Search query text.
    hits : list of RetrievalHit
        Passages to rank in stable index order.

    Returns
    -------
    str
        Instructional block including numbered snippets.
    """
    lines = [
        "Rank these passages by relevance to the query.",
        "Return ONLY JSON: {\"ranked_indices\": [<int>, ...]} — a permutation "
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


class AnthropicListwiseReranker:
    """One Messages call; model returns ``ranked_indices`` JSON.

    Parameters
    ----------
    model : str
        Anthropic model id (e.g. ``claude-3-5-haiku-20241022``).
    api_key : str or None
        ``x-api-key`` header value; if None, read from ``api_key_env``.
    api_key_env : str
        Environment variable holding the Anthropic API key.
    max_tokens : int
        Max tokens for the assistant reply (JSON only).
    timeout_s : float
        HTTP timeout.
    """

    def __init__(
        self,
        *,
        model: str,
        api_key: str | None = None,
        api_key_env: str = "ANTHROPIC_API_KEY",
        max_tokens: int = 1024,
        timeout_s: float = 120.0,
    ) -> None:
        """Persist Anthropic model id, auth, and HTTP limits.

        Parameters
        ----------
        model : str
            Messages API model id.
        api_key : str or None, optional
            Static ``x-api-key`` value; if omitted, read from ``api_key_env``.
        api_key_env : str, optional
            Environment variable consulted when ``api_key`` is unset.
        max_tokens : int, optional
            Budget for the assistant JSON reply.
        timeout_s : float, optional
            HTTP timeout for the Messages request.
        """
        self._model = model
        self._api_key = api_key
        self._api_key_env = api_key_env
        self._max_tokens = int(max_tokens)
        self._timeout_s = float(timeout_s)

    def _resolve_api_key(self) -> str:
        """Return Anthropic API key from arguments or the environment.

        Returns
        -------
        str
            Non-empty API key string.

        Raises
        ------
        ValueError
            If no key is available.
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
        """POST to Messages API, parse ``ranked_indices`` JSON, slice to ``top_k``.

        Parameters
        ----------
        query : str
            User question for relevance judging.
        hits : list of RetrievalHit
            Candidate passages in stable index order.
        top_k : int
            Number of reranked hits to return.

        Returns
        -------
        list of RetrievalHit
            Reordered subset with rank-derived ``similarity`` scores.

        Raises
        ------
        ValueError
            On HTTP errors, malformed JSON, or unexpected response shapes.
        """
        if not hits:
            return []
        k = max(1, int(top_k))
        n = len(hits)
        user_text = _build_user_block(query, hits)
        body: dict[str, Any] = {
            "model": self._model,
            "max_tokens": self._max_tokens,
            "messages": [{"role": "user", "content": user_text}],
        }
        url = "https://api.anthropic.com/v1/messages"
        req = urllib.request.Request(
            url,
            data=json.dumps(body).encode("utf-8"),
            method="POST",
            headers={
                "Content-Type": "application/json",
                "x-api-key": self._resolve_api_key(),
                "anthropic-version": "2023-06-01",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=self._timeout_s) as resp:
                raw = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", errors="replace")[:500]
            msg = f"Anthropic rerank HTTP {e.code}: {detail}"
            raise ValueError(msg) from e

        try:
            blocks = raw["content"]
            text_parts: list[str] = []
            for b in blocks:
                if isinstance(b, dict) and b.get("type") == "text":
                    text_parts.append(str(b.get("text", "")))
            content = "".join(text_parts)
        except (KeyError, TypeError) as e:
            msg = f"Unexpected Anthropic messages shape: {raw!r}"
            raise ValueError(msg) from e

        order = parse_ranked_indices(content, n_passages=n)
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
