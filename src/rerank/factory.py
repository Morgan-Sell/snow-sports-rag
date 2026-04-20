from __future__ import annotations

from typing import Any, Mapping

from ..config.coalesce import coalesce_openai_timeout_s
from .anthropic_listwise import AnthropicListwiseReranker
from .cross_encoder import CrossEncoderReranker
from .identity import IdentityReranker
from .openai_listwise import OpenAIListwiseReranker
from .protocol import Reranker

__all__ = ["reranker_from_config"]

__doc__ = """Construct a :class:`Reranker` from config."""


def reranker_from_config(
    rerank: Mapping[str, Any],
    *,
    llm: Mapping[str, Any] | None = None,
) -> Reranker:
    """Build reranker from ``rerank`` (and optionally ``llm``) config sections.

    Parameters
    ----------
    rerank : Mapping[str, Any]
        Keys: ``enabled``, ``backend`` (``cross_encoder`` | ``openai`` |
        ``anthropic`` | ``noop``), ``model_name``, ``top_n_in``, ``top_k_out``,
        backend-specific options.
    llm : Mapping[str, Any] or None
        When provided, ``openai`` reranker inherits ``base_url``,
        ``api_key_env``, ``timeout_s`` from here if not set under ``rerank``.

    Returns
    -------
    Reranker
        ``IdentityReranker`` when ``enabled`` is false or backend is ``noop``.
    """
    if not bool(rerank.get("enabled", False)):
        return IdentityReranker()

    raw_backend = str(rerank.get("backend", "cross_encoder")).strip().lower()
    backend = raw_backend.replace("-", "_")
    if backend in ("noop", "none", "identity"):
        return IdentityReranker()

    llm = llm or {}

    if backend == "cross_encoder":
        model = str(rerank.get("model_name", "cross-encoder/ms-marco-MiniLM-L-6-v2"))
        dev = rerank.get("device")
        device = str(dev) if isinstance(dev, str) and dev.strip() else None
        return CrossEncoderReranker(model, device=device)

    if backend == "openai":
        model = str(rerank.get("openai_model") or llm.get("model", "gpt-4o-mini"))
        base_default = "https://api.openai.com/v1"
        base = str(rerank.get("openai_base_url") or llm.get("base_url", base_default))
        key_env_default = "OPENAI_API_KEY"
        api_key_env = str(
            rerank.get("openai_api_key_env") or llm.get("api_key_env", key_env_default)
        )
        temp = float(rerank.get("openai_temperature", llm.get("temperature", 0.0)))
        raw_key = rerank.get("openai_api_key")
        api_key = (
            str(raw_key).strip()
            if isinstance(raw_key, str) and raw_key.strip()
            else None
        )
        timeout = coalesce_openai_timeout_s(rerank, llm)
        return OpenAIListwiseReranker(
            model=model,
            api_key=api_key,
            base_url=base,
            temperature=temp,
            api_key_env=api_key_env,
            timeout_s=timeout,
        )

    if backend == "anthropic":
        model = str(rerank.get("anthropic_model", "claude-3-5-haiku-20241022")).strip()
        api_key_env = str(rerank.get("anthropic_api_key_env", "ANTHROPIC_API_KEY"))
        raw_key = rerank.get("anthropic_api_key")
        api_key = (
            str(raw_key).strip()
            if isinstance(raw_key, str) and raw_key.strip()
            else None
        )
        max_tokens = int(rerank.get("anthropic_max_tokens", 1024))
        timeout = float(rerank.get("anthropic_timeout_s", 120.0))
        return AnthropicListwiseReranker(
            model=model,
            api_key=api_key,
            api_key_env=api_key_env,
            max_tokens=max_tokens,
            timeout_s=timeout,
        )

    msg = f"Unknown rerank.backend: {backend!r}"
    raise ValueError(msg)
