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

__all__ = ["OpenAIAnswerGenerator"]

__doc__ = """Grounded generation via OpenAI-compatible chat completions."""


class OpenAIAnswerGenerator:
    """Call ``/v1/chat/completions`` with a strict KB-grounding system prompt.

    Parameters
    ----------
    model : str
        Chat model id (e.g. ``gpt-4o-mini``).
    api_key : str or None, optional
        Bearer token; if ``None``, read from ``api_key_env``.
    base_url : str, optional
        OpenAI-compatible API root without trailing slash.
    temperature : float, optional
        Sampling temperature (keep low for factual grounding).
    max_tokens : int, optional
        Upper bound on generated tokens.
    api_key_env : str, optional
        Environment variable consulted when ``api_key`` is unset.
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
        base_url: str = "https://api.openai.com/v1",
        temperature: float = 0.1,
        max_tokens: int = 1024,
        api_key_env: str = "OPENAI_API_KEY",
        timeout_s: float = 120.0,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        refusal_message: str = DEFAULT_REFUSAL_MESSAGE,
        max_chars_per_hit: int = 1200,
        include_section_path: bool = True,
    ) -> None:
        """Store model, auth, and prompt-formatting configuration.

        Parameters
        ----------
        model, api_key, base_url, temperature, max_tokens, api_key_env, \
timeout_s : see class docstring.
        system_prompt : str
            Raw template; ``{refusal}`` is substituted with ``refusal_message``.
        refusal_message : str
            Used as both a substitution into the system prompt and the marker
            that identifies refusals in the returned :class:`GeneratedAnswer`.
        max_chars_per_hit : int
            Forwarded to :func:`build_citations`.
        include_section_path : bool
            Forwarded to :func:`build_user_prompt`.
        """
        self._model = model
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._temperature = float(temperature)
        self._max_tokens = int(max_tokens)
        self._api_key_env = api_key_env
        self._timeout_s = float(timeout_s)
        self._refusal = refusal_message
        self._system = system_prompt.replace("{refusal}", refusal_message)
        self._max_chars = int(max_chars_per_hit)
        self._include_section_path = bool(include_section_path)

    def _resolve_api_key(self) -> str:
        """Return the bearer token from the constructor or environment.

        Returns
        -------
        str
            Non-empty API key.

        Raises
        ------
        ValueError
            If neither ``api_key`` nor ``api_key_env`` yield a value.
        """
        if self._api_key is not None and str(self._api_key).strip():
            return str(self._api_key).strip()
        env = os.environ.get(self._api_key_env, "")
        if not env.strip():
            msg = (
                f"Missing API key: set {self._api_key_env!r} or pass api_key= "
                "to OpenAIAnswerGenerator"
            )
            raise ValueError(msg)
        return env.strip()

    def generate(
        self,
        query: str,
        hits: list[RetrievalHit],
    ) -> GeneratedAnswer:
        """POST chat completions and wrap the reply into :class:`GeneratedAnswer`.

        Parameters
        ----------
        query : str
            User question text.
        hits : list of RetrievalHit
            Context passages to cite; may be empty (forces a refusal).

        Returns
        -------
        GeneratedAnswer
            Provider-reported text with attached citations and token usage.

        Raises
        ------
        ValueError
            On HTTP failures or unexpected response payloads.
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
                backend="openai",
                model=self._model,
                usage={},
            )

        body: dict[str, Any] = {
            "model": self._model,
            "temperature": self._temperature,
            "max_tokens": self._max_tokens,
            "messages": [
                {"role": "system", "content": self._system},
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
            msg = f"OpenAI generation HTTP {e.code}: {detail}"
            raise ValueError(msg) from e

        try:
            content = str(raw["choices"][0]["message"]["content"]).strip()
        except (KeyError, IndexError, TypeError) as e:
            msg = f"Unexpected chat completions shape: {raw!r}"
            raise ValueError(msg) from e

        usage = raw.get("usage") if isinstance(raw, dict) else None
        usage_dict = dict(usage) if isinstance(usage, dict) else {}
        refused = content.strip() == self._refusal.strip()
        return GeneratedAnswer(
            answer=content,
            citations=cits,
            refused=refused,
            backend="openai",
            model=self._model,
            usage=usage_dict,
        )
