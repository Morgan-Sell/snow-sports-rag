from __future__ import annotations

from typing import Any, Mapping

from .fake import FakeLLMClient
from .openai_compatible import OpenAICompatibleLLMClient
from .protocol import LLMClient

__all__ = ["llm_client_from_config"]

__doc__ = """Construct :class:`LLMClient` from config mappings."""


def llm_client_from_config(
    llm: Mapping[str, Any],
) -> LLMClient:
    """Build an :class:`LLMClient` from the ``llm`` config subsection.

    Parameters
    ----------
    llm : Mapping[str, Any]
        Keys include ``provider`` (``fake`` or ``openai_compatible``),
        ``model``, ``temperature``, optional ``base_url``, ``api_key_env``.

    Returns
    -------
    LLMClient

    Raises
    ------
    ValueError
        If ``provider`` is unknown.
    """
    prov = str(llm.get("provider", "fake")).strip().lower().replace("-", "_")
    if prov == "fake":
        return FakeLLMClient()

    if prov == "openai_compatible":
        model = str(llm.get("model", "gpt-4o-mini")).strip()
        temp = float(llm.get("temperature", 0.2))
        base = str(llm.get("base_url", "https://api.openai.com/v1")).strip()
        api_key_env = str(llm.get("api_key_env", "OPENAI_API_KEY")).strip()
        raw_key = llm.get("api_key")
        if isinstance(raw_key, str) and raw_key.strip():
            api_key = str(raw_key).strip()
        else:
            api_key = None
        timeout = float(llm.get("timeout_s", 60.0))
        return OpenAICompatibleLLMClient(
            model=model,
            api_key=api_key,
            base_url=base,
            temperature=temp,
            api_key_env=api_key_env,
            timeout_s=timeout,
        )

    msg = f"Unknown llm.provider: {prov!r}"
    raise ValueError(msg)
