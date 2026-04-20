from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any

from ..retrieval.models import RetrievalHit
from .models import GeneratedAnswer
from .prompt import (
    DEFAULT_REFUSAL_MESSAGE,
    DEFAULT_SYSTEM_PROMPT,
    build_citations,
    build_user_prompt,
)

__all__ = ["AnthropicAnswerGenerator"]

__doc__ = """Grounded generation via Anthropic Messages API."""


class AnthropicAnswerGenerator:
    """Call Anthropic ``/v1/messages`` with strict KB-grounding instructions.

    Parameters
    ----------
    model : str
        Messages model id (e.g. ``claude-3-5-sonnet-20241022``).
    api_key : str or None, optional
        ``x-api-key`` header; if ``None``, read from ``api_key_env``.
    api_key_env : str, optional
        Environment variable consulted when ``api_key`` is unset.
    temperature : float, optional
        Sampling temperature.
    max_tokens : int, optional
        Maximum output tokens.
    timeout_s : float, optional
        HTTP timeout in seconds.
    system_prompt : str, optional
        Override the built-in grounding instructions.
    refusal_message : str, optional
        Injected into the system prompt and used to detect refusals.
    max_chars_per_hit : int, optional
        Per-passage truncation budget in the rendered prompt.
    include_section_path : bool, optional
        Whether to show ``section_path`` in ``[SOURCE n]`` headers.
    """

    def __init__(
        self,
        *,
        model: str,
        api_key: str | None = None,
        api_key_env: str = "ANTHROPIC_API_KEY",
        temperature: float = 0.1,
        max_tokens: int = 1024,
        timeout_s: float = 120.0,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        refusal_message: str = DEFAULT_REFUSAL_MESSAGE,
        max_chars_per_hit: int = 1200,
        include_section_path: bool = True,
    ) -> None:
        """Persist Anthropic endpoint, auth, and prompt-formatting config.

        Parameters
        ----------
        model, api_key, api_key_env, temperature, max_tokens, timeout_s : \
see class docstring.
        system_prompt : str
            Raw template; ``{refusal}`` is substituted with ``refusal_message``.
        refusal_message : str
            Both the grounding-rule placeholder and refusal detection marker.
        max_chars_per_hit : int
            Passed to :func:`build_citations`.
        include_section_path : bool
            Passed to :func:`build_user_prompt`.
        """
        self._model = model
        self._api_key = api_key
        self._api_key_env = api_key_env
        self._temperature = float(temperature)
        self._max_tokens = int(max_tokens)
        self._timeout_s = float(timeout_s)
        self._refusal = refusal_message
        self._system = system_prompt.replace("{refusal}", refusal_message)
        self._max_chars = int(max_chars_per_hit)
        self._include_section_path = bool(include_section_path)

    def _resolve_api_key(self) -> str:
        """Return an Anthropic key from attributes or the environment.

        Returns
        -------
        str
            Non-empty API key used in the ``x-api-key`` header.

        Raises
        ------
        ValueError
            If neither ``api_key`` nor ``api_key_env`` resolves.
        """
        if self._api_key is not None and str(self._api_key).strip():
            return str(self._api_key).strip()
        env = os.environ.get(self._api_key_env, "")
        if not env.strip():
            msg = (
                f"Missing API key: set {self._api_key_env!r} or pass api_key= "
                "to AnthropicAnswerGenerator"
            )
            raise ValueError(msg)
        return env.strip()

    def generate(
        self,
        query: str,
        hits: list[RetrievalHit],
    ) -> GeneratedAnswer:
        """POST to Anthropic Messages and wrap the reply.

        Parameters
        ----------
        query : str
            User question text.
        hits : list of RetrievalHit
            Context passages; empty triggers a local refusal.

        Returns
        -------
        GeneratedAnswer
            Concatenated text blocks with attached citations and ``usage``.

        Raises
        ------
        ValueError
            On HTTP errors or unexpected response shapes.
        """
        cits = build_citations(hits, max_chars_per_hit=self._max_chars)
        user = build_user_prompt(
            query, cits, include_section_path=self._include_section_path
        )
        if not cits:
            return GeneratedAnswer(
                answer=self._refusal,
                citations=[],
                refused=True,
                backend="anthropic",
                model=self._model,
                usage={},
            )

        body: dict[str, Any] = {
            "model": self._model,
            "system": self._system,
            "temperature": self._temperature,
            "max_tokens": self._max_tokens,
            "messages": [{"role": "user", "content": user}],
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
            msg = f"Anthropic generation HTTP {e.code}: {detail}"
            raise ValueError(msg) from e

        try:
            blocks = raw["content"]
            text_parts: list[str] = []
            for b in blocks:
                if isinstance(b, dict) and b.get("type") == "text":
                    text_parts.append(str(b.get("text", "")))
            content = "".join(text_parts).strip()
        except (KeyError, TypeError) as e:
            msg = f"Unexpected Anthropic messages shape: {raw!r}"
            raise ValueError(msg) from e

        usage = raw.get("usage") if isinstance(raw, dict) else None
        usage_dict = dict(usage) if isinstance(usage, dict) else {}
        refused = content.strip() == self._refusal.strip()
        return GeneratedAnswer(
            answer=content,
            citations=cits,
            refused=refused,
            backend="anthropic",
            model=self._model,
            usage=usage_dict,
        )
