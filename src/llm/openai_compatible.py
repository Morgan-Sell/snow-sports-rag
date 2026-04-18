from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from typing import Any

__all__ = ["OpenAICompatibleLLMClient"]

__doc__ = """OpenAI-compatible Chat Completions API for query expansion."""

_JSON_ARRAY = re.compile(r"\[[\s\S]*\]")


def _extract_json_array(text: str) -> list[Any]:
    """Parse the first JSON array from model output."""
    raw = text.strip()
    m = _JSON_ARRAY.search(raw)
    if not m:
        msg = f"Expected a JSON array in LLM response, got: {raw[:200]!r}"
        raise ValueError(msg)
    data = json.loads(m.group())
    if not isinstance(data, list):
        msg = f"JSON top-level value must be a list, got {type(data)}"
        raise ValueError(msg)
    return data


class OpenAICompatibleLLMClient:
    """Query expansion via ``/v1/chat/completions`` (OpenAI-compatible servers).

    Parameters
    ----------
    model : str
        Chat model id.
    api_key : str or None, optional
        Bearer token. If ``None``, reads from the environment variable named
        by ``api_key_env``.
    base_url : str, optional
        API root without trailing slash (default OpenAI v1 base).
    temperature : float, optional
        Sampling temperature for the expansion call.
    api_key_env : str, optional
        Environment variable holding the API key when ``api_key`` is omitted.
    timeout_s : float, optional
        HTTP timeout in seconds.
    """

    def __init__(
        self,
        *,
        model: str,
        api_key: str | None = None,
        base_url: str = "https://api.openai.com/v1",
        temperature: float = 0.2,
        api_key_env: str = "OPENAI_API_KEY",
        timeout_s: float = 60.0,
    ) -> None:
        self._model = model
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._temperature = float(temperature)
        self._api_key_env = api_key_env
        self._timeout_s = float(timeout_s)

    def _resolve_api_key(self) -> str:
        if self._api_key is not None and str(self._api_key).strip():
            return str(self._api_key).strip()
        env = os.environ.get(self._api_key_env, "")
        if not env.strip():
            msg = (
                f"Missing API key: set {self._api_key_env!r} or pass api_key= "
                "to OpenAICompatibleLLMClient"
            )
            raise ValueError(msg)
        return env.strip()

    def expand_query(self, query: str, *, num_paraphrases: int = 3) -> list[str]:
        """Ask the chat model for JSON-array paraphrases."""
        n = max(0, int(num_paraphrases))
        if not query.strip() or n == 0:
            return []

        system = (
            "You rewrite search queries for semantic retrieval. "
            "Return ONLY a JSON array of strings: short paraphrases or alternate "
            "wordings of the user's query. No markdown fences, no commentary. "
            f"At most {n} strings; each must be non-empty and distinct."
        )
        user = f"Original query:\n{query.strip()}\n\nProduce up to {n} paraphrases."

        body: dict[str, Any] = {
            "model": self._model,
            "temperature": self._temperature,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        url = f"{self._base_url}/chat/completions"
        payload = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=payload,
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
            msg = f"OpenAI-compatible HTTP {e.code}: {detail}"
            raise ValueError(msg) from e

        try:
            content = raw["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as e:
            msg = f"Unexpected chat completions shape: {raw!r}"
            raise ValueError(msg) from e

        arr = _extract_json_array(str(content))
        out: list[str] = []
        seen: set[str] = set()
        for item in arr:
            if len(out) >= n:
                break
            if not isinstance(item, str):
                continue
            s = item.strip()
            if not s or s in seen:
                continue
            seen.add(s)
            out.append(s)
        return out
