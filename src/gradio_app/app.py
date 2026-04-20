from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

from dotenv import find_dotenv, load_dotenv

from ..chunking import chunk_strategy_from_config
from ..config import AppConfig, load_config
from ..embedding import embedding_model_from_config
from ..ingest import KnowledgeBaseLoader
from ..pipeline import PRESETS, RAGPipeline, TraceLogger
from ..retrieval import IndexBuilder
from ..vectorstore import chroma_l2_l1_stores_from_config
from .components import (
    DEFAULT_LOW_EVIDENCE_THRESHOLD,
    EMPTY_EVIDENCE_HTML,
    EMPTY_INDEX_BANNER_HTML,
    EMPTY_SOURCES_HTML,
    EXAMPLE_QUESTIONS,
    HEADER_HTML,
    HERO_HTML,
    render_debug_panel,
    render_evidence_banner,
    render_preset_caption,
    render_settings_readonly,
    render_source_cards,
)
from .css import ALPINE_CSS
from .theme import make_alpine_theme

__all__ = ["build_demo", "launch", "_auto_index_if_empty"]

__doc__ = """Gradio Blocks app for the Snow Sports RAG system.

Three layout layers:

1. **User UI** (always visible): chat + sources panel + feedback buttons.
2. **Settings** (accordion, collapsed): preset, toggles, read-only config.
3. **Debug** (accordion, hidden unless Debug toggle is on): expansions,
   L1 shortlist, L2 pre-rerank, reranked, latencies.
"""


_DEFAULT_TRACE_PATH = Path(".rag_traces/traces.jsonl")


def _auto_index_if_empty(
    cfg: AppConfig,
    *,
    log: Any = print,
) -> dict[str, int]:
    """Build the L1/L2 indexes from ``cfg`` when either collection is empty.

    This is the implementation behind ``snow-sports-rag-ui --auto-index``.
    It reuses the same helpers as the ``snow_sports_rag index`` CLI command
    so the resulting on-disk layout is identical.

    Parameters
    ----------
    cfg : AppConfig
        Parsed application configuration.
    log : callable, optional
        Called with a single string for progress messages. Defaults to
        :func:`print`; tests pass a no-op.

    Returns
    -------
    dict
        Always includes an ``action`` key (``"skipped"`` when both stores
        already have rows, or ``"indexed"`` when a build ran). When a build
        runs, also includes ``"chunks"``, ``"docs"``, ``"l2_before"``, and
        ``"l1_before"`` for diagnostics and tests.
    """
    l2_store, l1_store = chroma_l2_l1_stores_from_config(cfg.vector_store)
    l2_before = l2_store.count()
    l1_before = l1_store.count()
    if l2_before > 0 and l1_before > 0:
        return {"action": "skipped", "l2_before": l2_before, "l1_before": l1_before}

    log(
        f"[auto-index] collections empty (l2={l2_before}, l1={l1_before}); "
        "building index from knowledge base…"
    )
    loader = KnowledgeBaseLoader(cfg)
    docs = loader.load_all()
    strategy = chunk_strategy_from_config(cfg.chunking)
    embedder = embedding_model_from_config(cfg.embedding)
    builder = IndexBuilder(strategy, embedder, l2_store, l1_store=l1_store)
    n_chunks = builder.build(docs)
    n_docs = len(docs)
    log(f"[auto-index] indexed {n_chunks} L2 chunks and {n_docs} L1 summaries")
    return {
        "action": "indexed",
        "chunks": int(n_chunks),
        "docs": int(n_docs),
        "l2_before": l2_before,
        "l1_before": l1_before,
    }


def _load_env_file() -> None:
    """Populate ``os.environ`` from a ``.env`` file if one is present.

    Mirrors :func:`snow_sports_rag.cli._load_env_file` so the Gradio app
    also picks up API keys without the user having to export them manually.
    """
    path = find_dotenv(usecwd=True)
    if path:
        load_dotenv(path, override=False)


def build_demo(
    cfg: AppConfig,
    *,
    pipeline: RAGPipeline | None = None,
    trace_logger: TraceLogger | None = None,
    low_evidence_threshold: float = DEFAULT_LOW_EVIDENCE_THRESHOLD,
) -> Any:
    """Construct the Gradio ``Blocks`` app without launching it.

    Parameters
    ----------
    cfg : AppConfig
        Parsed application configuration.
    pipeline : RAGPipeline or None, optional
        Pre-built pipeline (useful for tests). A new one is constructed from
        ``cfg`` when omitted.
    trace_logger : TraceLogger or None, optional
        Override trace file destination. Defaults to a logger at
        ``.rag_traces/traces.jsonl``.
    low_evidence_threshold : float, optional
        Top-source similarity below which the evidence banner triggers.

    Returns
    -------
    gradio.Blocks
        The fully wired demo; call ``.launch()`` to serve it.
    """
    import gradio as gr

    if pipeline is None:
        pipeline = RAGPipeline(cfg)
    if trace_logger is None:
        trace_logger = TraceLogger(_DEFAULT_TRACE_PATH)

    preset_names = list(PRESETS.keys())
    default_preset = "Balanced"
    default_rerank = bool(cfg.rerank.get("enabled", False))
    default_expand = bool(cfg.query_expansion.get("enabled", False))
    default_generate = bool(cfg.generation.get("enabled", False))

    theme = make_alpine_theme()

    demo_kwargs: dict[str, Any] = {
        "title": "Snow Sports RAG",
        "fill_height": True,
    }
    with gr.Blocks(**demo_kwargs) as demo:
        demo._alpine_theme = theme
        demo._alpine_css = ALPINE_CSS
        gr.HTML(HEADER_HTML)

        last_trace_id = gr.State("")
        last_config_hash = gr.State(pipeline.config_hash)

        initial_banner = EMPTY_EVIDENCE_HTML
        try:
            if pipeline.index_empty:
                initial_banner = EMPTY_INDEX_BANNER_HTML
        except Exception:
            initial_banner = EMPTY_EVIDENCE_HTML
        evidence_banner = gr.HTML(value=initial_banner)

        with gr.Row(equal_height=False):
            with gr.Column(scale=3, min_width=420):
                with gr.Column(visible=True) as hero_column:
                    gr.HTML(value=HERO_HTML)
                    with gr.Row(elem_classes=["alpine-hero-row"]):
                        example_buttons = [
                            gr.Button(q, variant="secondary", size="sm")
                            for q in EXAMPLE_QUESTIONS
                        ]
                chatbot = gr.Chatbot(
                    label="Chat",
                    height=460,
                    buttons=["copy"],
                    value=[],
                )
                with gr.Row():
                    msg = gr.Textbox(
                        placeholder="Ask about routes, resorts, conditions, gear…",
                        label="Your question",
                        lines=2,
                        scale=5,
                    )
                    send_btn = gr.Button("Send", variant="primary", scale=1)
                with gr.Row():
                    thumbs_up = gr.Button("👍  Helpful", variant="secondary")
                    thumbs_down = gr.Button("👎  Not helpful", variant="secondary")
                    feedback_status = gr.Markdown("", elem_id="alpine-feedback-status")

            with gr.Column(scale=2, min_width=320):
                gr.Markdown("### Sources")
                sources_html = gr.HTML(value=EMPTY_SOURCES_HTML)

        with gr.Accordion("Settings", open=False):
            with gr.Row():
                with gr.Column(scale=1):
                    preset_radio = gr.Radio(
                        choices=preset_names,
                        value=default_preset,
                        label="Retrieval preset",
                    )
                    preset_caption = gr.HTML(
                        value=render_preset_caption(default_preset)
                    )
                    rerank_toggle = gr.Checkbox(value=default_rerank, label="Reranker")
                    expand_toggle = gr.Checkbox(
                        value=default_expand, label="Query expansion"
                    )
                    generate_toggle = gr.Checkbox(
                        value=default_generate,
                        label="Generate grounded answer",
                    )
                    debug_toggle = gr.Checkbox(
                        value=False,
                        label="Debug mode (show pipeline internals)",
                    )
                with gr.Column(scale=1):
                    gr.Markdown("**Active configuration** (read-only)")
                    gr.HTML(
                        value=render_settings_readonly(
                            cfg, config_hash=pipeline.config_hash
                        )
                    )

        debug_accordion = gr.Accordion(
            "Debug · pipeline internals",
            open=False,
            visible=False,
        )
        with debug_accordion:
            debug_html = gr.HTML(
                value=(
                    '<div class="alpine-empty">'
                    "Run a query with Debug mode enabled to see internals."
                    "</div>"
                )
            )

        def _on_submit(
            message: str,
            history: list[dict],
            preset: str,
            rerank_on: bool,
            expand_on: bool,
            generate_on: bool,
            debug_on: bool,
        ):
            history = list(history or [])
            user_text = (message or "").strip()
            if not user_text:
                return (
                    history,
                    "",
                    EMPTY_SOURCES_HTML,
                    "",
                    "",
                    "",
                    EMPTY_EVIDENCE_HTML,
                    gr.update(),
                )

            history.append({"role": "user", "content": user_text})

            result = pipeline.run(
                user_text,
                preset=preset,
                rerank_enabled=rerank_on,
                expansion_enabled=expand_on,
                generation_enabled=generate_on,
            )

            if result.answer is not None:
                assistant_text = result.answer.answer
            else:
                if result.cards:
                    ids = ", ".join(f"[{c.index}] {c.doc_id}" for c in result.cards)
                    assistant_text = (
                        "Generation is disabled. Retrieved sources:\n\n" + ids
                    )
                else:
                    assistant_text = "No sources were retrieved for this query."
            history.append({"role": "assistant", "content": assistant_text})

            sources_markup = render_source_cards(result.cards, debug=debug_on)
            debug_markup = render_debug_panel(result.trace) if debug_on else ""
            banner_markup = render_evidence_banner(
                result, threshold=low_evidence_threshold
            )

            try:
                trace_logger.log_query(
                    result,
                    preset=preset,
                    rerank_enabled=rerank_on,
                    expansion_enabled=expand_on,
                    generation_enabled=generate_on,
                )
            except OSError:
                pass

            return (
                history,
                "",
                sources_markup,
                debug_markup,
                result.trace_id,
                result.config_hash,
                banner_markup,
                gr.update(visible=False),
            )

        submit_outputs = [
            chatbot,
            msg,
            sources_html,
            debug_html,
            last_trace_id,
            last_config_hash,
            evidence_banner,
            hero_column,
        ]
        submit_inputs = [
            msg,
            chatbot,
            preset_radio,
            rerank_toggle,
            expand_toggle,
            generate_toggle,
            debug_toggle,
        ]
        send_btn.click(_on_submit, inputs=submit_inputs, outputs=submit_outputs)
        msg.submit(_on_submit, inputs=submit_inputs, outputs=submit_outputs)

        for btn, question in zip(example_buttons, EXAMPLE_QUESTIONS):
            btn.click(lambda q=question: q, inputs=None, outputs=msg)

        def _on_debug_toggle(enabled: bool) -> dict:
            return gr.update(visible=bool(enabled), open=bool(enabled))

        debug_toggle.change(
            _on_debug_toggle, inputs=debug_toggle, outputs=debug_accordion
        )

        def _on_preset_change(name: str) -> str:
            return render_preset_caption(name)

        preset_radio.change(
            _on_preset_change, inputs=preset_radio, outputs=preset_caption
        )

        def _feedback(vote: str, trace_id: str, config_hash: str) -> str:
            if not trace_id:
                return "Ask a question first — nothing to rate yet."
            try:
                trace_logger.log_feedback(
                    trace_id=trace_id,
                    config_hash=config_hash,
                    feedback=vote,
                )
            except OSError as e:
                return f"Could not record feedback: {e}"
            label = "👍 helpful" if vote == "up" else "👎 not helpful"
            return f"Thanks — recorded **{label}** for this answer."

        thumbs_up.click(
            lambda tid, ch: _feedback("up", tid, ch),
            inputs=[last_trace_id, last_config_hash],
            outputs=feedback_status,
        )
        thumbs_down.click(
            lambda tid, ch: _feedback("down", tid, ch),
            inputs=[last_trace_id, last_config_hash],
            outputs=feedback_status,
        )

    return demo


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    """Parse CLI arguments for the Gradio entry point.

    Parameters
    ----------
    argv : list of str or None
        Raw argv excluding the program name; ``None`` uses :data:`sys.argv`.

    Returns
    -------
    argparse.Namespace
        Parsed arguments: ``config``, ``base_dir``, ``host``, ``port``,
        ``share``.
    """
    parser = argparse.ArgumentParser(
        prog="snow-sports-rag-ui",
        description="Launch the Snow Sports RAG Gradio UI (Phase 3).",
    )
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--base-dir", type=Path, default=None)
    parser.add_argument(
        "--host",
        type=str,
        default=os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1"),
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("GRADIO_SERVER_PORT", "7860")),
    )
    parser.add_argument("--share", action="store_true")
    parser.add_argument(
        "--auto-index",
        action="store_true",
        help=(
            "Build the vector index from the knowledge base at startup when "
            "either Chroma collection is empty. No-op when an index already "
            "exists. Off by default so launches stay fast and reproducible."
        ),
    )
    return parser.parse_args(argv)


def launch(argv: list[str] | None = None) -> None:
    """CLI-friendly entry point: load config + env, build demo, serve.

    Parameters
    ----------
    argv : list of str or None, optional
        Override argv for tests. ``None`` delegates to :data:`sys.argv`.
    """
    _load_env_file()
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    base = args.base_dir.resolve() if args.base_dir else Path.cwd()
    cfg_path = args.config
    if cfg_path is None:
        cfg_path = base / "configs/default.yaml"
    elif not cfg_path.is_absolute():
        cfg_path = (base / cfg_path).resolve()
    cfg = load_config(cfg_path, base_dir=base)
    if getattr(args, "auto_index", False):
        try:
            _auto_index_if_empty(cfg)
        except Exception as exc:
            print(f"[auto-index] failed: {exc}", file=sys.stderr)
    demo = build_demo(cfg)
    theme = getattr(demo, "_alpine_theme", None)
    css = getattr(demo, "_alpine_css", None)
    demo.queue().launch(
        server_name=args.host,
        server_port=args.port,
        share=bool(args.share),
        theme=theme,
        css=css,
    )
