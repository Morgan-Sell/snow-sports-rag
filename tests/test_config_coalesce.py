"""Tests for optional config key coalescing (explicit ``None`` vs missing key)."""

from __future__ import annotations

from snow_sports_rag.config.coalesce import coalesce_openai_timeout_s
from snow_sports_rag.generation.factory import answer_generator_from_config
from snow_sports_rag.generation.openai import OpenAIAnswerGenerator


def test_coalesce_openai_timeout_s_prefers_section_when_set() -> None:
    assert (
        coalesce_openai_timeout_s(
            {"openai_timeout_s": 30.0},
            {"timeout_s": 60.0},
        )
        == 30.0
    )


def test_coalesce_openai_timeout_s_inherits_llm_when_section_none() -> None:
    """Merged defaults use ``openai_timeout_s: None``; must fall back to ``llm``."""
    assert (
        coalesce_openai_timeout_s(
            {"openai_timeout_s": None},
            {"timeout_s": 60.0},
        )
        == 60.0
    )


def test_coalesce_openai_timeout_s_default_when_both_missing() -> None:
    assert coalesce_openai_timeout_s({}, {}) == 120.0


def test_answer_generator_openai_accepts_merged_none_openai_timeout_s() -> None:
    """Regression: ``float(None)`` must not occur when building OpenAI generator."""
    gen = {
        "enabled": True,
        "backend": "openai",
        "openai_timeout_s": None,
    }
    llm = {
        "timeout_s": 60.0,
        "model": "gpt-4o-mini",
        "base_url": "https://api.openai.com/v1",
        "api_key_env": "OPENAI_API_KEY",
    }
    g = answer_generator_from_config(gen, llm=llm)
    assert isinstance(g, OpenAIAnswerGenerator)
    assert g._timeout_s == 60.0
