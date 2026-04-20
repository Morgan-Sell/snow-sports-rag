from __future__ import annotations

from typing import Any, Mapping

from ..config.coalesce import coalesce_openai_timeout_s
from .anthropic import AnthropicAnswerGenerator
from .fake import FakeAnswerGenerator
from .huggingface import HuggingFaceAnswerGenerator
from .openai import OpenAIAnswerGenerator
from .prompt import DEFAULT_REFUSAL_MESSAGE, DEFAULT_SYSTEM_PROMPT
from .protocol import AnswerGenerator

__all__ = ["answer_generator_from_config"]

__doc__ = """Construct an :class:`AnswerGenerator` from config mappings."""


def _opt_str(mapping: Mapping[str, Any], key: str) -> str | None:
    """Return a stripped string value from ``mapping`` or ``None``.

    Parameters
    ----------
    mapping : Mapping[str, Any]
        Config subsection.
    key : str
        Key to look up.

    Returns
    -------
    str or None
        ``None`` when the value is missing, not a string, or empty after strip.
    """
    v = mapping.get(key)
    if isinstance(v, str) and v.strip():
        return v.strip()
    return None


def answer_generator_from_config(
    generation: Mapping[str, Any],
    *,
    llm: Mapping[str, Any] | None = None,
) -> AnswerGenerator:
    """Build an :class:`AnswerGenerator` from the ``generation`` config section.

    Parameters
    ----------
    generation : Mapping[str, Any]
        Keys: ``enabled``, ``backend`` (``openai`` | ``anthropic`` |
        ``huggingface`` | ``fake``), ``temperature``, ``max_tokens``,
        ``max_chars_per_hit``, ``include_section_path``, ``refusal_message``,
        ``system_prompt``, and backend-specific options.
    llm : Mapping[str, Any] or None, optional
        Shared LLM settings. The ``openai`` generator falls back to
        ``llm.base_url`` / ``llm.api_key_env`` / ``llm.timeout_s`` /
        ``llm.model`` when the equivalent keys are not set under
        ``generation``.

    Returns
    -------
    AnswerGenerator
        :class:`FakeAnswerGenerator` when ``enabled`` is false or backend is
        ``fake``; concrete backend otherwise.

    Raises
    ------
    ValueError
        If ``backend`` is not recognised.
    """
    llm = llm or {}

    refusal = _opt_str(generation, "refusal_message") or DEFAULT_REFUSAL_MESSAGE
    system = _opt_str(generation, "system_prompt") or DEFAULT_SYSTEM_PROMPT
    max_chars = int(generation.get("max_chars_per_hit", 1200))
    include_sp = bool(generation.get("include_section_path", True))
    temperature = float(generation.get("temperature", 0.1))
    max_tokens = int(generation.get("max_tokens", 1024))

    common: dict[str, Any] = {
        "system_prompt": system,
        "refusal_message": refusal,
        "max_chars_per_hit": max_chars,
        "include_section_path": include_sp,
    }

    if not bool(generation.get("enabled", False)):
        return FakeAnswerGenerator(refusal_message=refusal, max_chars_per_hit=max_chars)

    raw_backend = str(generation.get("backend", "openai")).strip().lower()
    backend = raw_backend.replace("-", "_")

    if backend == "fake":
        return FakeAnswerGenerator(refusal_message=refusal, max_chars_per_hit=max_chars)

    if backend == "openai":
        model = (
            _opt_str(generation, "openai_model")
            or _opt_str(llm, "model")
            or "gpt-4o-mini"
        )
        base_url = (
            _opt_str(generation, "openai_base_url")
            or _opt_str(llm, "base_url")
            or "https://api.openai.com/v1"
        )
        api_key_env = (
            _opt_str(generation, "openai_api_key_env")
            or _opt_str(llm, "api_key_env")
            or "OPENAI_API_KEY"
        )
        api_key = _opt_str(generation, "openai_api_key")
        timeout_s = coalesce_openai_timeout_s(generation, llm)
        return OpenAIAnswerGenerator(
            model=model,
            api_key=api_key,
            base_url=base_url,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key_env=api_key_env,
            timeout_s=timeout_s,
            **common,
        )

    if backend == "anthropic":
        model = _opt_str(generation, "anthropic_model") or "claude-3-5-sonnet-20241022"
        api_key_env = (
            _opt_str(generation, "anthropic_api_key_env") or "ANTHROPIC_API_KEY"
        )
        api_key = _opt_str(generation, "anthropic_api_key")
        timeout_s = float(generation.get("anthropic_timeout_s", 120.0))
        return AnthropicAnswerGenerator(
            model=model,
            api_key=api_key,
            api_key_env=api_key_env,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout_s=timeout_s,
            **common,
        )

    if backend == "huggingface":
        model = _opt_str(generation, "hf_model") or "Qwen/Qwen2.5-3B-Instruct"
        device = _opt_str(generation, "hf_device")
        dtype = _opt_str(generation, "hf_dtype")
        max_new_tokens = int(generation.get("hf_max_new_tokens", max_tokens))
        ds_raw = generation.get("hf_do_sample")
        do_sample = bool(ds_raw) if ds_raw is not None else None
        return HuggingFaceAnswerGenerator(
            model_name=model,
            device=device,
            dtype=dtype,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
            **common,
        )

    msg = f"Unknown generation.backend: {backend!r}"
    raise ValueError(msg)
