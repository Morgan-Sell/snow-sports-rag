from __future__ import annotations

from typing import Any

from ..retrieval.models import RetrievalHit
from .models import GeneratedAnswer
from .prompt import (
    DEFAULT_REFUSAL_MESSAGE,
    DEFAULT_SYSTEM_PROMPT,
    build_citations,
    build_user_prompt,
)

__all__ = ["HuggingFaceAnswerGenerator"]

__doc__ = """Local grounded generation using ``transformers`` text generation.

The ``transformers`` and ``torch`` packages are imported lazily inside
:meth:`_ensure_pipeline` so that importing this module never requires them.
"""


class HuggingFaceAnswerGenerator:
    """Run a local causal LM through the ``transformers`` pipeline API.

    Parameters
    ----------
    model_name : str
        Hub id or local path (e.g. ``meta-llama/Llama-3.1-8B-Instruct``).
    device : str or None, optional
        Torch device override (``cuda``, ``mps``, ``cpu``). ``None`` defers to
        ``transformers`` auto-placement.
    dtype : str or None, optional
        Torch dtype name (``float16``, ``bfloat16``, ...). ``None`` keeps the
        model default.
    max_new_tokens : int, optional
        Upper bound on tokens generated per call.
    temperature : float, optional
        Sampling temperature (``<=0`` disables sampling).
    do_sample : bool, optional
        Explicit override; when ``None`` it is inferred from ``temperature``.
    system_prompt : str, optional
        Override the built-in grounding template.
    refusal_message : str, optional
        Substituted into the system prompt and used to detect refusals.
    max_chars_per_hit : int, optional
        Per-passage truncation budget.
    include_section_path : bool, optional
        Whether to show ``section_path`` in ``[SOURCE n]`` headers.
    """

    def __init__(
        self,
        *,
        model_name: str,
        device: str | None = None,
        dtype: str | None = None,
        max_new_tokens: int = 512,
        temperature: float = 0.1,
        do_sample: bool | None = None,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        refusal_message: str = DEFAULT_REFUSAL_MESSAGE,
        max_chars_per_hit: int = 1200,
        include_section_path: bool = True,
    ) -> None:
        """Record model/runtime options without loading weights.

        Parameters
        ----------
        See class docstring for each argument. The pipeline is constructed
        lazily on the first call to :meth:`generate` via :meth:`_ensure_pipeline`.
        """
        self._model_name = model_name
        self._device = device
        self._dtype = dtype
        self._max_new_tokens = int(max_new_tokens)
        self._temperature = float(temperature)
        self._do_sample = do_sample
        self._refusal = refusal_message
        self._system = system_prompt.replace("{refusal}", refusal_message)
        self._max_chars = int(max_chars_per_hit)
        self._include_section_path = bool(include_section_path)
        self._pipeline: Any | None = None
        self._tokenizer: Any | None = None

    def _ensure_pipeline(self) -> Any:
        """Construct and cache a ``text-generation`` pipeline on first use.

        Returns
        -------
        Any
            A ``transformers.pipelines.TextGenerationPipeline`` instance.

        Raises
        ------
        RuntimeError
            When ``transformers`` (and its backend) cannot be imported.
        """
        if self._pipeline is not None:
            return self._pipeline
        try:
            import torch  # type: ignore[import-untyped]
            from transformers import (  # type: ignore[import-untyped]
                AutoModelForCausalLM,
                AutoTokenizer,
                pipeline,
            )
        except Exception as e:  # pragma: no cover - import failure path
            msg = (
                "HuggingFaceAnswerGenerator requires 'transformers' and "
                "'torch' to be installed."
            )
            raise RuntimeError(msg) from e

        torch_dtype = None
        if self._dtype:
            torch_dtype = getattr(torch, self._dtype, None)

        tokenizer = AutoTokenizer.from_pretrained(self._model_name)
        model_kwargs: dict[str, Any] = {}
        if torch_dtype is not None:
            model_kwargs["torch_dtype"] = torch_dtype
        model = AutoModelForCausalLM.from_pretrained(self._model_name, **model_kwargs)

        pipe_kwargs: dict[str, Any] = {"task": "text-generation"}
        if self._device is not None:
            pipe_kwargs["device"] = self._device

        self._tokenizer = tokenizer
        self._pipeline = pipeline(model=model, tokenizer=tokenizer, **pipe_kwargs)
        return self._pipeline

    def _format_messages(self, user: str) -> str:
        """Render chat messages through the tokenizer's chat template.

        Parameters
        ----------
        user : str
            The user-turn text including context and the question.

        Returns
        -------
        str
            Prompt string ready for the text-generation pipeline. Falls back
            to a plain ``"System:\\nUser:\\nAssistant:"`` layout when the
            tokenizer does not define a chat template.
        """
        messages = [
            {"role": "system", "content": self._system},
            {"role": "user", "content": user},
        ]
        tok = self._tokenizer
        apply = getattr(tok, "apply_chat_template", None)
        if apply is not None:
            try:
                return apply(messages, tokenize=False, add_generation_prompt=True)
            except Exception:
                pass
        return f"System:\n{self._system}\n\nUser:\n{user}\n\nAssistant:\n"

    def generate(
        self,
        query: str,
        hits: list[RetrievalHit],
    ) -> GeneratedAnswer:
        """Run the local pipeline and strip the prompt from its output.

        Parameters
        ----------
        query : str
            User question text.
        hits : list of RetrievalHit
            Context passages; empty triggers a local refusal without running
            the model.

        Returns
        -------
        GeneratedAnswer
            Local model output with citations attached.
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
                backend="huggingface",
                model=self._model_name,
                usage={},
            )

        pipe = self._ensure_pipeline()
        prompt_text = self._format_messages(user)

        do_sample = self._do_sample
        if do_sample is None:
            do_sample = self._temperature > 0.0

        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": self._max_new_tokens,
            "do_sample": bool(do_sample),
            "return_full_text": False,
        }
        if do_sample:
            gen_kwargs["temperature"] = float(self._temperature)

        out = pipe(prompt_text, **gen_kwargs)
        text = ""
        if isinstance(out, list) and out:
            first = out[0]
            if isinstance(first, dict):
                text = str(first.get("generated_text", "")).strip()
        refused = text.strip() == self._refusal.strip()
        return GeneratedAnswer(
            answer=text,
            citations=cits,
            refused=refused,
            backend="huggingface",
            model=self._model_name,
            usage={},
        )
