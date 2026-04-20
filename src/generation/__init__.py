"""Phase 2.4 grounded answer generation.

Concrete backends (``openai``, ``anthropic``, ``huggingface``) share the
:class:`AnswerGenerator` protocol and a strict system prompt so that responses
are derived only from retrieved knowledge-base passages.
"""

from .anthropic import AnthropicAnswerGenerator
from .factory import answer_generator_from_config
from .fake import FakeAnswerGenerator
from .huggingface import HuggingFaceAnswerGenerator
from .models import GeneratedAnswer, SourceCitation
from .openai import OpenAIAnswerGenerator
from .prompt import (
    DEFAULT_REFUSAL_MESSAGE,
    DEFAULT_SYSTEM_PROMPT,
    build_citations,
    build_user_prompt,
    format_context_block,
)
from .protocol import AnswerGenerator

__all__ = [
    "AnswerGenerator",
    "AnthropicAnswerGenerator",
    "DEFAULT_REFUSAL_MESSAGE",
    "DEFAULT_SYSTEM_PROMPT",
    "FakeAnswerGenerator",
    "GeneratedAnswer",
    "HuggingFaceAnswerGenerator",
    "OpenAIAnswerGenerator",
    "SourceCitation",
    "answer_generator_from_config",
    "build_citations",
    "build_user_prompt",
    "format_context_block",
]
