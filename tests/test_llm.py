from __future__ import annotations

import pytest

from snow_sports_rag.llm import FakeLLMClient, llm_client_from_config
from snow_sports_rag.llm.openai_compatible import (
    OpenAICompatibleLLMClient,
    _extract_json_array,
)


def test_fake_llm_expand_query_respects_count() -> None:
    llm = FakeLLMClient()
    out = llm.expand_query("ski racing", num_paraphrases=2)
    assert len(out) == 2
    assert all("ski racing" in x for x in out)


def test_llm_client_from_config_fake() -> None:
    c = llm_client_from_config({"provider": "fake"})
    assert isinstance(c, FakeLLMClient)


def test_llm_client_from_config_unknown() -> None:
    with pytest.raises(ValueError, match="Unknown"):
        llm_client_from_config({"provider": "nosuch"})


def test_extract_json_array_tolerates_prefix_text() -> None:
    assert _extract_json_array('Sure: ["a", "b"]') == ["a", "b"]


def test_openai_expand_query_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    client = OpenAICompatibleLLMClient(
        model="gpt-4o-mini",
        api_key_env="OPENAI_API_KEY",
    )
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        client.expand_query("hello", num_paraphrases=1)
