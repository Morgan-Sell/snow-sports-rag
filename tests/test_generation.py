from __future__ import annotations

import json
from unittest.mock import patch

import pytest
from snow_sports_rag.generation import (
    DEFAULT_REFUSAL_MESSAGE,
    DEFAULT_SYSTEM_PROMPT,
    AnthropicAnswerGenerator,
    FakeAnswerGenerator,
    HuggingFaceAnswerGenerator,
    OpenAIAnswerGenerator,
    answer_generator_from_config,
    build_citations,
    build_user_prompt,
    format_context_block,
)
from snow_sports_rag.retrieval.models import RetrievalHit


def _hit(cid: str, text: str = "body", sim: float = 0.9) -> RetrievalHit:
    return RetrievalHit(
        chunk_id=cid,
        text=text,
        doc_id=f"{cid}.md",
        section_path="S",
        chunk_index=0,
        similarity=sim,
        distance=1.0 - sim,
    )


def test_build_citations_truncates_and_numbers() -> None:
    hits = [_hit("a", text="x" * 2000), _hit("b", text="short")]
    cits = build_citations(hits, max_chars_per_hit=50)
    assert [c.index for c in cits] == [1, 2]
    assert cits[0].text.endswith("…")
    assert len(cits[0].text) <= 50
    assert cits[1].text == "short"


def test_format_context_block_includes_section_path() -> None:
    hits = [_hit("a"), _hit("b")]
    cits = build_citations(hits, max_chars_per_hit=0)
    block = format_context_block(cits, include_section_path=True)
    assert "[SOURCE 1]" in block
    assert "[SOURCE 2]" in block
    assert "section=S" in block


def test_format_context_block_can_hide_section_path() -> None:
    cits = build_citations([_hit("a")], max_chars_per_hit=0)
    block = format_context_block(cits, include_section_path=False)
    assert "section=" not in block


def test_build_user_prompt_empty_context_note() -> None:
    prompt = build_user_prompt("why?", citations=[])
    assert "(no sources retrieved)" in prompt
    assert "QUESTION: why?" in prompt


def test_fake_generator_refuses_on_empty_hits() -> None:
    gen = FakeAnswerGenerator()
    ans = gen.generate("what is the snowpack?", [])
    assert ans.refused is True
    assert ans.answer == DEFAULT_REFUSAL_MESSAGE
    assert ans.citations == []
    assert ans.backend == "fake"


def test_fake_generator_cites_sources() -> None:
    gen = FakeAnswerGenerator()
    hits = [_hit("a"), _hit("b")]
    ans = gen.generate("a test question", hits)
    assert ans.refused is False
    assert "[1][2]" in ans.answer
    assert [c.index for c in ans.citations] == [1, 2]


def test_default_system_prompt_contains_grounding_rules() -> None:
    assert "ONLY" in DEFAULT_SYSTEM_PROMPT
    assert "{refusal}" in DEFAULT_SYSTEM_PROMPT


def test_answer_generator_from_config_disabled_is_fake() -> None:
    gen = answer_generator_from_config({"enabled": False, "backend": "openai"})
    assert isinstance(gen, FakeAnswerGenerator)


def test_answer_generator_from_config_backend_fake() -> None:
    gen = answer_generator_from_config({"enabled": True, "backend": "fake"})
    assert isinstance(gen, FakeAnswerGenerator)


def test_answer_generator_from_config_unknown() -> None:
    with pytest.raises(ValueError, match="Unknown"):
        answer_generator_from_config({"enabled": True, "backend": "magic"})


def test_answer_generator_from_config_openai_inherits_llm() -> None:
    gen = answer_generator_from_config(
        {"enabled": True, "backend": "openai"},
        llm={
            "model": "gpt-4o-mini",
            "base_url": "https://api.openai.com/v1",
            "api_key_env": "OPENAI_API_KEY",
            "timeout_s": 45.0,
        },
    )
    assert isinstance(gen, OpenAIAnswerGenerator)


def test_answer_generator_from_config_anthropic() -> None:
    gen = answer_generator_from_config(
        {
            "enabled": True,
            "backend": "anthropic",
            "anthropic_model": "claude-3-5-sonnet-20241022",
        }
    )
    assert isinstance(gen, AnthropicAnswerGenerator)


def test_answer_generator_from_config_huggingface() -> None:
    gen = answer_generator_from_config(
        {
            "enabled": True,
            "backend": "huggingface",
            "hf_model": "Qwen/Qwen2.5-3B-Instruct",
        }
    )
    assert isinstance(gen, HuggingFaceAnswerGenerator)


def test_openai_generator_parses_reply() -> None:
    hits = [_hit("a"), _hit("b")]
    api_body = {
        "choices": [{"message": {"content": "Yes, per [1][2]."}}],
        "usage": {"prompt_tokens": 100, "completion_tokens": 12},
    }
    raw = json.dumps(api_body).encode("utf-8")

    class _FakeResp:
        def read(self) -> bytes:
            return raw

        def __enter__(self) -> _FakeResp:
            return self

        def __exit__(self, *args: object) -> None:
            return None

    def fake_urlopen(req: object, timeout: float = 0) -> _FakeResp:
        return _FakeResp()

    with patch.object(OpenAIAnswerGenerator, "_resolve_api_key", return_value="sk"):
        with patch("urllib.request.urlopen", fake_urlopen):
            gen = OpenAIAnswerGenerator(model="gpt-4o-mini", api_key="x")
            ans = gen.generate("question", hits)
    assert ans.backend == "openai"
    assert ans.answer == "Yes, per [1][2]."
    assert ans.refused is False
    assert [c.index for c in ans.citations] == [1, 2]
    assert ans.usage["prompt_tokens"] == 100


def test_openai_generator_refuses_on_empty_hits_without_http() -> None:
    def boom(req: object, timeout: float = 0) -> object:
        raise AssertionError("should not call urlopen when hits is empty")

    with patch.object(OpenAIAnswerGenerator, "_resolve_api_key", return_value="sk"):
        with patch("urllib.request.urlopen", boom):
            gen = OpenAIAnswerGenerator(model="gpt-4o-mini", api_key="x")
            ans = gen.generate("question", [])
    assert ans.refused is True
    assert ans.answer == DEFAULT_REFUSAL_MESSAGE
    assert ans.citations == []


def test_openai_generator_detects_refusal_reply() -> None:
    hits = [_hit("a")]
    api_body = {
        "choices": [{"message": {"content": DEFAULT_REFUSAL_MESSAGE}}],
    }
    raw = json.dumps(api_body).encode("utf-8")

    class _FakeResp:
        def read(self) -> bytes:
            return raw

        def __enter__(self) -> _FakeResp:
            return self

        def __exit__(self, *args: object) -> None:
            return None

    def fake_urlopen(req: object, timeout: float = 0) -> _FakeResp:
        return _FakeResp()

    with patch.object(OpenAIAnswerGenerator, "_resolve_api_key", return_value="sk"):
        with patch("urllib.request.urlopen", fake_urlopen):
            gen = OpenAIAnswerGenerator(model="gpt-4o-mini", api_key="x")
            ans = gen.generate("q", hits)
    assert ans.refused is True


def test_anthropic_generator_parses_reply() -> None:
    hits = [_hit("a")]
    api_body = {
        "content": [{"type": "text", "text": "Short answer [1]."}],
        "usage": {"input_tokens": 50, "output_tokens": 5},
    }
    raw = json.dumps(api_body).encode("utf-8")

    class _FakeRespB:
        def read(self) -> bytes:
            return raw

        def __enter__(self) -> _FakeRespB:
            return self

        def __exit__(self, *args: object) -> None:
            return None

    def fake_urlopen_b(req: object, timeout: float = 0) -> _FakeRespB:
        return _FakeRespB()

    with patch.object(AnthropicAnswerGenerator, "_resolve_api_key", return_value="k"):
        with patch("urllib.request.urlopen", fake_urlopen_b):
            gen = AnthropicAnswerGenerator(
                model="claude-3-5-sonnet-20241022", api_key="x"
            )
            ans = gen.generate("q", hits)
    assert ans.backend == "anthropic"
    assert ans.answer == "Short answer [1]."
    assert ans.refused is False
    assert ans.usage["output_tokens"] == 5


def test_huggingface_generator_uses_pipeline() -> None:
    hits = [_hit("a")]

    class _FakePipe:
        def __call__(self, prompt_text: str, **kwargs: object) -> list[dict]:
            assert "QUESTION" in prompt_text
            return [{"generated_text": "grounded answer [1]"}]

    class _FakeTok:
        def apply_chat_template(
            self,
            messages: list,
            *,
            tokenize: bool = False,
            add_generation_prompt: bool = False,
        ) -> str:
            parts = [f"{m['role']}: {m['content']}" for m in messages]
            return "\n".join(parts)

    gen = HuggingFaceAnswerGenerator(model_name="stub")
    gen._pipeline = _FakePipe()
    gen._tokenizer = _FakeTok()

    ans = gen.generate("q", hits)
    assert ans.backend == "huggingface"
    assert ans.model == "stub"
    assert ans.answer == "grounded answer [1]"
    assert ans.refused is False
    assert len(ans.citations) == 1


def test_huggingface_generator_refuses_without_loading_model() -> None:
    gen = HuggingFaceAnswerGenerator(model_name="stub")
    ans = gen.generate("q", [])
    assert ans.refused is True
    assert gen._pipeline is None
