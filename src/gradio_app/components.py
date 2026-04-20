from __future__ import annotations

from html import escape

from ..pipeline.models import PipelineResult, PipelineTrace, SourceCard
from ..pipeline.presets import PRESETS

__all__ = [
    "DEFAULT_LOW_EVIDENCE_THRESHOLD",
    "EMPTY_EVIDENCE_HTML",
    "EMPTY_INDEX_BANNER_HTML",
    "EMPTY_SOURCES_HTML",
    "EXAMPLE_QUESTIONS",
    "HERO_HTML",
    "HEADER_HTML",
    "build_card_tooltip",
    "render_debug_panel",
    "render_empty_index_banner",
    "render_evidence_banner",
    "render_preset_caption",
    "render_settings_readonly",
    "render_source_cards",
]

__doc__ = """HTML renderers for the alpine Gradio UI.

These functions intentionally return simple HTML strings so the main app
module can stay focused on layout and event wiring.
"""


HEADER_HTML = """
<div class="alpine-header">
  <span class="alpine-mark" aria-hidden="true">⛰</span>
  <div>
    <div class="alpine-title">Trailhead</div>
    <div class="alpine-subtitle">
      Your starting point for trusted snow sports answers.
    </div>
  </div>
</div>
""".strip()


EMPTY_SOURCES_HTML = (
    '<div class="alpine-sources">'
    '<div class="alpine-empty">Sources will appear here after you ask a question.</div>'
    "</div>"
)


EMPTY_EVIDENCE_HTML = ""


DEFAULT_LOW_EVIDENCE_THRESHOLD = 0.25


EXAMPLE_QUESTIONS: tuple[str, ...] = (
    "Which athletes have won gold in alpine events?",
    "What resorts have gondolas?",
    "Summarize recent World Cup race results.",
    "How is a slalom course structured?",
)


HERO_HTML = (
    '<div class="alpine-hero">'
    '  <div class="alpine-hero-mark" aria-hidden="true">❄</div>'
    '  <div class="alpine-hero-title">Welcome to Trailhead</div>'
    '  <div class="alpine-hero-sub">'
    "    Answers are grounded in retrieved sources. Try one of the example "
    "    questions below, or type your own."
    "  </div>"
    "</div>"
)


def build_card_tooltip(card: SourceCard) -> str:
    """Return the ``title=`` tooltip text for a :class:`SourceCard`.

    Parameters
    ----------
    card : SourceCard
        Card to describe; ``pre_rerank_similarity`` drives the rerank delta.

    Returns
    -------
    str
        Plain multi-line string suitable for an HTML ``title`` attribute
        (newlines render as line breaks in most browsers' native tooltip).
    """
    lines = [f"similarity: {card.similarity:+.3f}"]
    if (
        card.pre_rerank_similarity is not None
        and card.pre_rerank_similarity != card.similarity
    ):
        delta = card.similarity - card.pre_rerank_similarity
        lines.append(f"pre-rerank: {card.pre_rerank_similarity:+.3f}")
        lines.append(f"rerank Δ:   {delta:+.3f}")
    lines.append(f"doc_id: {card.doc_id}")
    if card.section_path:
        lines.append(f"section: {card.section_path}")
    lines.append(f"chunk: {card.chunk_index}")
    return "\n".join(lines)


def render_source_cards(
    cards: list[SourceCard],
    *,
    debug: bool = False,
) -> str:
    """Render the Sources panel as a vertical stack of cards.

    Parameters
    ----------
    cards : list of SourceCard
        Output of :func:`RAGPipeline.run` (``result.cards``).
    debug : bool, optional
        Whether to show numeric similarity scores on each card. Hidden by
        default per product requirements.

    Returns
    -------
    str
        Single HTML ``<div>`` element; safe for ``gr.HTML``.
    """
    if not cards:
        return EMPTY_SOURCES_HTML

    parts: list[str] = ['<div class="alpine-sources">']
    for c in cards:
        doc = escape(c.doc_id)
        section = escape(c.section_path) if c.section_path else ""
        snippet = escape(c.snippet)
        tooltip = escape(build_card_tooltip(c), quote=True)
        score_html = (
            f'<span class="alpine-score">sim {c.similarity:+.3f}</span>'
            if debug
            else '<span class="alpine-hint" aria-hidden="true">ⓘ</span>'
        )
        section_html = (
            f'<div class="alpine-section">§ {section} · chunk {c.chunk_index}</div>'
            if section
            else f'<div class="alpine-section">chunk {c.chunk_index}</div>'
        )
        parts.append(
            f"""
            <div class="alpine-card" title="{tooltip}">
              <div class="alpine-card-head">
                <span class="alpine-badge">{c.index}</span>
                <div>
                  <div class="alpine-doc">{doc}</div>
                  {section_html}
                </div>
                {score_html}
              </div>
              <div class="alpine-snippet">{snippet}</div>
            </div>
            """.strip()
        )
    parts.append("</div>")
    return "\n".join(parts)


def render_empty_index_banner() -> str:
    """Return the glacier-blue "knowledge base is not indexed" banner.

    Called both at UI startup (via :func:`build_demo`) and per-query (via
    :func:`render_evidence_banner`) so the message is identical in both
    places. The command strings below intentionally match the ones in the
    project ``README`` so copy/paste works.

    Returns
    -------
    str
        HTML fragment using the ``alpine-banner alpine-banner-info``
        classes defined in :mod:`snow_sports_rag.gradio_app.css`.
    """
    return (
        '<div class="alpine-banner alpine-banner-info" role="status">'
        '  <span class="alpine-banner-icon" aria-hidden="true">❄</span>'
        "  <div>"
        '    <div class="alpine-banner-title">'
        "      Knowledge base is not indexed yet"
        "    </div>"
        '    <div class="alpine-banner-body">'
        "      Run <code>uv run python -m snow_sports_rag index</code> from the"
        "      project root, then reload this page. To auto-build on launch"
        "      instead, start the UI with"
        "      <code>uv run snow-sports-rag-ui --auto-index</code>."
        "    </div>"
        "  </div>"
        "</div>"
    )


EMPTY_INDEX_BANNER_HTML = render_empty_index_banner()


def render_evidence_banner(
    result: PipelineResult,
    *,
    threshold: float = DEFAULT_LOW_EVIDENCE_THRESHOLD,
) -> str:
    """Return an HTML banner when the answer is weak, else an empty string.

    The banner fires in three cases, in priority order:

    1. No passages were retrieved at all.
    2. The generator explicitly refused due to insufficient evidence.
    3. The top source similarity is below ``threshold`` (likely out-of-domain
       or unfamiliar vocabulary).

    Parameters
    ----------
    result : PipelineResult
        Output of :meth:`RAGPipeline.run`.
    threshold : float, optional
        Similarity floor for "high confidence". Defaults to
        :data:`DEFAULT_LOW_EVIDENCE_THRESHOLD` (``0.25``).

    Returns
    -------
    str
        Banner ``<div>`` or the empty string when confidence is acceptable.
    """
    cards = result.cards
    top_sim = max((c.similarity for c in cards), default=None)

    if getattr(result, "index_empty", False):
        return render_empty_index_banner()

    if not cards:
        title = "No matching sources"
        body = (
            "The knowledge base returned no passages for this question. "
            "Try different keywords or a broader phrasing."
        )
    elif result.answer is not None and result.answer.refused:
        title = "Limited evidence"
        body = (
            "The assistant refused because the retrieved sources don't "
            "contain enough information to answer confidently. "
            "Try rephrasing your question."
        )
    elif top_sim is not None and top_sim < threshold:
        title = "Low-confidence sources"
        body = (
            f"Top similarity is {top_sim:+.3f} (below {threshold:+.2f}). "
            "Results may be off-topic — consider refining your question."
        )
    else:
        return EMPTY_EVIDENCE_HTML

    return (
        '<div class="alpine-banner alpine-banner-warn" role="status">'
        f'  <span class="alpine-banner-icon" aria-hidden="true">⚠</span>'
        "  <div>"
        f'    <div class="alpine-banner-title">{escape(title)}</div>'
        f'    <div class="alpine-banner-body">{escape(body)}</div>'
        "  </div>"
        "</div>"
    )


def _render_string_list(items: list[str]) -> str:
    """Render a list of plain strings as an HTML table with one column.

    Parameters
    ----------
    items : list of str
        Rows to display (for example expansions or L1 doc_ids).

    Returns
    -------
    str
        ``<table>`` markup, or an empty-state message.
    """
    if not items:
        return '<div class="alpine-empty">(none)</div>'
    rows = "".join(
        f"<tr><td>{i + 1}</td><td>{escape(s)}</td></tr>" for i, s in enumerate(items)
    )
    return (
        "<table><thead><tr><th>#</th><th>value</th></tr></thead>"
        f"<tbody>{rows}</tbody></table>"
    )


def _render_hit_table(trace_rows: list[dict]) -> str:
    """Render a list of ``{index, doc_id, section, sim}`` dicts as a table.

    Parameters
    ----------
    trace_rows : list of dict
        Pre-shaped rows (see :func:`render_debug_panel`).

    Returns
    -------
    str
        Table markup or an empty-state message.
    """
    if not trace_rows:
        return '<div class="alpine-empty">(none)</div>'
    head = (
        "<thead><tr><th>#</th><th>doc_id</th><th>section</th>"
        "<th>chunk</th><th>score</th></tr></thead>"
    )
    body = []
    for r in trace_rows:
        body.append(
            "<tr>"
            f"<td>{r['index']}</td>"
            f"<td>{escape(r['doc_id'])}</td>"
            f"<td>{escape(r['section_path'])}</td>"
            f"<td>{r['chunk_index']}</td>"
            f"<td>{r['similarity']:+.3f}</td>"
            "</tr>"
        )
    return f"<table>{head}<tbody>{''.join(body)}</tbody></table>"


def render_debug_panel(trace: PipelineTrace) -> str:
    """Render the full Debug panel (expansions, L1/L2, rerank, latency).

    Parameters
    ----------
    trace : PipelineTrace
        Trace produced by :meth:`RAGPipeline.run`.

    Returns
    -------
    str
        Single HTML fragment suitable for ``gr.HTML``.
    """

    def _rows_from_hits(hits: list, limit: int = 50) -> list[dict]:
        out: list[dict] = []
        for i, h in enumerate(hits[:limit], start=1):
            out.append(
                {
                    "index": i,
                    "doc_id": h.doc_id,
                    "section_path": h.section_path,
                    "chunk_index": h.chunk_index,
                    "similarity": float(h.similarity),
                }
            )
        return out

    lat = trace.latency
    latency_html = (
        '<div class="alpine-latency">'
        f"<span>expansion {lat.expansion_ms:.1f} ms</span>"
        f"<span>retrieval {lat.retrieval_ms:.1f} ms</span>"
        f"<span>rerank {lat.rerank_ms:.1f} ms</span>"
        f"<span>generation {lat.generation_ms:.1f} ms</span>"
        f"<span>total {lat.total_ms:.1f} ms</span>"
        "</div>"
    )
    return "\n".join(
        [
            '<div class="alpine-debug">',
            "<h4>Expanded queries</h4>",
            _render_string_list(trace.expansions),
            "<h4>L1 document shortlist</h4>",
            _render_string_list(trace.l1_shortlist),
            "<h4>L2 candidates (pre-rerank)</h4>",
            _render_hit_table(_rows_from_hits(trace.l2_pre_rerank)),
            "<h4>Reranked / final</h4>",
            _render_hit_table(_rows_from_hits(trace.reranked)),
            "<h4>Latency</h4>",
            latency_html,
            "</div>",
        ]
    )


def render_settings_readonly(
    cfg,
    *,
    config_hash: str,
) -> str:
    """Render the read-only Settings box (models, store type, chunking).

    Parameters
    ----------
    cfg : AppConfig
        Parsed config; only the non-secret display fields are pulled.
    config_hash : str
        Short hash from :func:`compute_config_hash`.

    Returns
    -------
    str
        ``<dl>`` markup describing the active configuration.
    """

    def _get(section_name: str, key: str, default: str = "—") -> str:
        section = getattr(cfg, section_name, None) or {}
        return escape(str(section.get(key, default)))

    return "\n".join(
        [
            '<dl class="alpine-info">',
            "<dt>embedding model</dt>",
            f"<dd>{_get('embedding', 'model_name')}</dd>",
            "<dt>reranker</dt>",
            f"<dd>{_get('rerank', 'backend')} · {_get('rerank', 'model_name')}</dd>",
            "<dt>vector store</dt>",
            f"<dd>{_get('vector_store', 'backend')}"
            f" ({_get('vector_store', 'collection_name')})</dd>",
            "<dt>chunking</dt>",
            f"<dd>{_get('chunking', 'strategy')} · size "
            f"{_get('chunking', 'chunk_size')} · overlap "
            f"{_get('chunking', 'chunk_overlap')}</dd>",
            "<dt>config_hash</dt>",
            f"<dd>{escape(config_hash)}</dd>",
            "</dl>",
        ]
    )


def render_preset_caption(preset_name: str) -> str:
    """Return a short italic caption for the selected preset.

    Parameters
    ----------
    preset_name : str
        Label currently selected in the UI radio.

    Returns
    -------
    str
        Inline HTML caption; empty string for unknown presets.
    """
    preset = PRESETS.get(preset_name)
    if preset is None:
        return ""
    return (
        f'<span style="color:var(--alpine-slate);font-size:12px;">'
        f"{escape(preset.description)}</span>"
    )
