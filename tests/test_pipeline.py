from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from snow_sports_rag.chunking import MarkdownHeaderChunkStrategy
from snow_sports_rag.config import load_config
from snow_sports_rag.embedding import FakeEmbeddingModel
from snow_sports_rag.ingest.models import SourceDocument
from snow_sports_rag.pipeline import (
    PRESETS,
    RAGPipeline,
    TraceLogger,
    compute_config_hash,
    resolve_preset,
)
from snow_sports_rag.pipeline.trace import new_trace_id
from snow_sports_rag.retrieval import IndexBuilder
from snow_sports_rag.retrieval.models import RetrievalHit
from snow_sports_rag.vectorstore import ChromaVectorStore

# ---------------------------------------------------------------------------
# presets
# ---------------------------------------------------------------------------


def test_factory_openai_when_merged_enabled_true() -> None:
    """YAML may disable generation; merged ``enabled: True`` selects real backend."""
    from snow_sports_rag.generation.factory import answer_generator_from_config
    from snow_sports_rag.generation.fake import FakeAnswerGenerator
    from snow_sports_rag.generation.openai import OpenAIAnswerGenerator

    yaml_style = {"enabled": False, "backend": "openai"}
    fake = answer_generator_from_config(yaml_style, llm={})
    assert isinstance(fake, FakeAnswerGenerator)

    merged = {**yaml_style, "enabled": True}
    real = answer_generator_from_config(merged, llm={})
    assert isinstance(real, OpenAIAnswerGenerator)


def test_resolve_preset_case_insensitive_and_fallback() -> None:
    assert resolve_preset("fast").name == "Fast"
    assert resolve_preset("BALANCED").name == "Balanced"
    assert resolve_preset(None).name == "Balanced"
    assert resolve_preset("nonsense").name == "Balanced"


def test_preset_has_all_three_named_tiers() -> None:
    assert set(PRESETS) == {"Fast", "Balanced", "Deep"}
    assert PRESETS["Fast"].top_k < PRESETS["Balanced"].top_k < PRESETS["Deep"].top_k
    assert (
        PRESETS["Fast"].top_k_out
        <= PRESETS["Balanced"].top_k_out
        <= PRESETS["Deep"].top_k_out
    )


# ---------------------------------------------------------------------------
# config hash
# ---------------------------------------------------------------------------


def test_compute_config_hash_is_stable_and_strips_secrets() -> None:
    base = {
        "chunking": {"strategy": "x"},
        "embedding": {"model_name": "m"},
        "vector_store": {"collection_name": "c"},
        "retrieval": {"top_k": 8},
        "rerank": {"enabled": False},
        "query_expansion": {"enabled": False},
        "generation": {"enabled": False},
    }
    h1 = compute_config_hash(base)
    with_secret = dict(base)
    with_secret["rerank"] = {
        "enabled": False,
        "openai_api_key": "sk-xxx",
    }
    h2 = compute_config_hash(with_secret)
    assert h1 == h2, "config hash must ignore API keys"

    bumped = dict(base)
    bumped["retrieval"] = {"top_k": 16}
    h3 = compute_config_hash(bumped)
    assert h1 != h3
    assert len(h1) == 12


# ---------------------------------------------------------------------------
# trace logger
# ---------------------------------------------------------------------------


def test_trace_logger_feedback_writes_valid_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "traces.jsonl"
    logger = TraceLogger(path)
    logger.log_feedback(trace_id="abc", config_hash="h1", feedback="up")
    logger.log_feedback(trace_id="abc", config_hash="h1", feedback="down")

    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    rows = [json.loads(ln) for ln in lines]
    assert all(r["type"] == "feedback" for r in rows)
    assert rows[0]["trace_id"] == "abc"
    assert rows[0]["payload"]["feedback"] == "up"
    assert rows[1]["payload"]["feedback"] == "down"


def test_trace_logger_disabled_does_not_create_file(tmp_path: Path) -> None:
    path = tmp_path / "nope.jsonl"
    logger = TraceLogger(path, enabled=False)
    logger.log_feedback(trace_id="t", config_hash="h", feedback="up")
    assert not path.exists()


def test_new_trace_id_unique() -> None:
    ids = {new_trace_id() for _ in range(50)}
    assert len(ids) == 50


# ---------------------------------------------------------------------------
# pipeline (integration using fake embedder + real Chroma in tmp_path)
# ---------------------------------------------------------------------------


def _write_config(tmp_path: Path) -> Path:
    """Write a minimal YAML config pointing embedding=fake and a tmp chroma dir."""
    cfg = {
        "knowledge_base_path": "kb",
        "embedding": {
            "backend": "fake",
            "model_name": "pipeline-fake",
            "dimension": 8,
        },
        "vector_store": {
            "backend": "chroma",
            "persist_directory": str(tmp_path / "db"),
            "collection_name": "snow_sports_kb",
            "l1_collection_name": "snow_sports_kb_l1",
        },
        "retrieval": {
            "top_k": 8,
            "mode": "hierarchical",
            "hierarchical_global_fallback": True,
        },
        "rerank": {"enabled": False},
        "query_expansion": {"enabled": False},
        "generation": {"enabled": False},
    }
    (tmp_path / "kb").mkdir(exist_ok=True)
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.dump(cfg), encoding="utf-8")
    return cfg_path


@pytest.fixture
def indexed_cfg(tmp_path: Path):
    """Build a tiny hierarchical index and return an AppConfig for it."""
    cfg_path = _write_config(tmp_path)
    cfg = load_config(cfg_path, base_dir=tmp_path)

    embedder = FakeEmbeddingModel(
        dimension=int(cfg.embedding.get("dimension", 8)),
        model_name=str(cfg.embedding["model_name"]),
    )
    root = cfg.vector_store["persist_directory"]
    l2 = ChromaVectorStore(root, cfg.vector_store["collection_name"])
    l1 = ChromaVectorStore(root, cfg.vector_store["l1_collection_name"])
    docs = [
        SourceDocument(
            doc_id="athletes/a.md",
            entity_type="athletes",
            title="Alpine Annie",
            raw_markdown=(
                "# Alpine Annie\n\n## Bio\n\n"
                + "zebra " * 40
                + "\n\n## Career\n\n"
                + "mountain ridge " * 30
                + "\n"
            ),
            headings=("Bio", "Career"),
        ),
        SourceDocument(
            doc_id="resorts/pine.md",
            entity_type="resorts",
            title="Pine Resort",
            raw_markdown=(
                "# Pine Resort\n\n## Overview\n\n"
                + "pine trails " * 40
                + "\n\n## Facilities\n\n"
                + "gondola lodge " * 30
                + "\n"
            ),
            headings=("Overview", "Facilities"),
        ),
    ]
    strategy = MarkdownHeaderChunkStrategy(chunk_size=120, chunk_overlap=20)
    IndexBuilder(strategy, embedder, l2, l1_store=l1).build(docs)
    return cfg


def test_pipeline_run_returns_cards_trace_and_config_hash(indexed_cfg) -> None:
    pipe = RAGPipeline(indexed_cfg)
    result = pipe.run(
        "pine trails gondola lodge",
        preset="Balanced",
        rerank_enabled=False,
        expansion_enabled=False,
        generation_enabled=False,
    )

    assert result.query == "pine trails gondola lodge"
    assert result.trace_id
    assert result.config_hash == pipe.config_hash
    assert result.cards, "expected at least one source card"
    assert [c.index for c in result.cards] == list(range(1, len(result.cards) + 1))
    for c in result.cards:
        assert c.chunk_id
        assert isinstance(c.snippet, str) and c.snippet
    assert result.trace.l1_shortlist, "L1 shortlist should not be empty"
    assert result.trace.l2_pre_rerank, "L2 candidates should not be empty"
    assert result.trace.reranked, "final list should not be empty"
    assert result.trace.latency.total_ms >= 0.0
    assert result.answer is None


def test_pipeline_run_empty_query_is_noop(indexed_cfg) -> None:
    pipe = RAGPipeline(indexed_cfg)
    result = pipe.run("   ")
    assert result.cards == []
    assert result.trace.l1_shortlist == []
    assert result.trace.l2_pre_rerank == []


def test_pipeline_run_respects_preset_top_k(indexed_cfg) -> None:
    pipe = RAGPipeline(indexed_cfg)
    fast = pipe.run("pine", preset="Fast")
    deep = pipe.run("pine", preset="Deep")
    assert len(fast.cards) <= PRESETS["Fast"].top_k_out
    assert len(deep.cards) <= PRESETS["Deep"].top_k_out


def test_trace_logger_log_query_roundtrip(tmp_path: Path, indexed_cfg) -> None:
    pipe = RAGPipeline(indexed_cfg)
    result = pipe.run(
        "pine trails",
        preset="Fast",
        rerank_enabled=False,
        expansion_enabled=False,
        generation_enabled=False,
    )

    tr_path = tmp_path / "traces.jsonl"
    logger = TraceLogger(tr_path)
    logger.log_query(
        result,
        preset="Fast",
        rerank_enabled=False,
        expansion_enabled=False,
        generation_enabled=False,
    )

    rows = [json.loads(ln) for ln in tr_path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    row = rows[0]
    assert row["type"] == "query"
    assert row["trace_id"] == result.trace_id
    assert row["config_hash"] == result.config_hash
    payload = row["payload"]
    assert payload["preset"] == "Fast"
    assert payload["query"] == "pine trails"
    assert isinstance(payload["l1_shortlist"], list)
    assert isinstance(payload["l2_pre_rerank"], list)
    assert isinstance(payload["final_sources"], list)
    assert isinstance(payload["latency_ms"], dict)


# ---------------------------------------------------------------------------
# component renderers
# ---------------------------------------------------------------------------


def _stub_hit(doc_id: str = "a.md", idx: int = 0, sim: float = 0.5) -> RetrievalHit:
    return RetrievalHit(
        chunk_id=f"{doc_id}::{idx}",
        text="alpine body text" * 3,
        doc_id=doc_id,
        section_path="Overview",
        chunk_index=idx,
        similarity=sim,
        distance=1.0 - sim,
    )


def test_render_source_cards_escapes_and_hides_scores() -> None:
    from snow_sports_rag.gradio_app.components import render_source_cards
    from snow_sports_rag.pipeline.models import SourceCard

    cards = [
        SourceCard(
            index=1,
            chunk_id="a.md::0",
            doc_id="<evil>.md",
            section_path="Bio",
            chunk_index=0,
            similarity=0.42,
            snippet="hello <script>",
            pre_rerank_similarity=0.30,
        )
    ]
    html_default = render_source_cards(cards, debug=False)
    assert "&lt;evil&gt;" in html_default
    assert "&lt;script&gt;" in html_default
    assert "sim " not in html_default
    assert 'title="' in html_default
    assert "pre-rerank:" in html_default
    assert "rerank " in html_default

    html_debug = render_source_cards(cards, debug=True)
    assert "sim " in html_debug


def test_build_card_tooltip_omits_pre_rerank_when_missing() -> None:
    from snow_sports_rag.gradio_app.components import build_card_tooltip
    from snow_sports_rag.pipeline.models import SourceCard

    card = SourceCard(
        index=1,
        chunk_id="a.md::0",
        doc_id="a.md",
        section_path="",
        chunk_index=0,
        similarity=0.7,
        snippet="body",
        pre_rerank_similarity=None,
    )
    tip = build_card_tooltip(card)
    assert "similarity:" in tip
    assert "pre-rerank" not in tip
    assert "rerank " not in tip


def test_render_evidence_banner_low_similarity() -> None:
    from snow_sports_rag.gradio_app.components import render_evidence_banner
    from snow_sports_rag.pipeline.models import (
        PipelineResult,
        PipelineTrace,
        SourceCard,
    )

    weak = PipelineResult(
        query="foo",
        cards=[
            SourceCard(
                index=1,
                chunk_id="a.md::0",
                doc_id="a.md",
                section_path="",
                chunk_index=0,
                similarity=0.05,
                snippet="…",
            )
        ],
        answer=None,
        trace=PipelineTrace(query="foo"),
        trace_id="t",
        config_hash="h",
    )
    html = render_evidence_banner(weak, threshold=0.25)
    assert "alpine-banner" in html
    assert "Low-confidence" in html


def test_render_evidence_banner_empty_when_strong() -> None:
    from snow_sports_rag.gradio_app.components import render_evidence_banner
    from snow_sports_rag.pipeline.models import (
        PipelineResult,
        PipelineTrace,
        SourceCard,
    )

    strong = PipelineResult(
        query="foo",
        cards=[
            SourceCard(
                index=1,
                chunk_id="a.md::0",
                doc_id="a.md",
                section_path="",
                chunk_index=0,
                similarity=0.9,
                snippet="…",
            )
        ],
        answer=None,
        trace=PipelineTrace(query="foo"),
        trace_id="t",
        config_hash="h",
    )
    assert render_evidence_banner(strong, threshold=0.25) == ""


def test_render_evidence_banner_no_sources() -> None:
    from snow_sports_rag.gradio_app.components import render_evidence_banner
    from snow_sports_rag.pipeline.models import PipelineResult, PipelineTrace

    empty = PipelineResult(
        query="foo",
        cards=[],
        answer=None,
        trace=PipelineTrace(query="foo"),
        trace_id="t",
        config_hash="h",
    )
    html = render_evidence_banner(empty)
    assert "No matching sources" in html


def test_pipeline_populates_pre_rerank_similarity(indexed_cfg) -> None:
    pipe = RAGPipeline(indexed_cfg)
    result = pipe.run(
        "pine trails gondola lodge",
        preset="Balanced",
        rerank_enabled=False,
        expansion_enabled=False,
        generation_enabled=False,
    )
    assert result.cards
    assert all(c.pre_rerank_similarity is not None for c in result.cards), (
        "pipeline should copy the pre-rerank score for each final card"
    )
    for c in result.cards:
        assert abs(c.similarity - c.pre_rerank_similarity) < 1e-6


def test_render_debug_panel_contains_all_sections() -> None:
    from snow_sports_rag.gradio_app.components import render_debug_panel
    from snow_sports_rag.pipeline.models import PipelineTrace, StageLatency

    trace = PipelineTrace(
        query="q",
        expansions=["alt phrasing"],
        variants=["q", "alt phrasing"],
        l1_shortlist=["a.md", "b.md"],
        l2_pre_rerank=[_stub_hit("a.md", 0, 0.4), _stub_hit("b.md", 1, 0.3)],
        reranked=[_stub_hit("a.md", 0, 0.9)],
        latency=StageLatency(
            expansion_ms=1.0,
            retrieval_ms=2.0,
            rerank_ms=0.0,
            generation_ms=0.0,
            total_ms=3.0,
        ),
    )
    html = render_debug_panel(trace)
    assert "Expanded queries" in html
    assert "L1 document shortlist" in html
    assert "L2 candidates" in html
    assert "Reranked" in html
    assert "Latency" in html
    assert "alt phrasing" in html
    assert "a.md" in html


def test_render_settings_readonly_shows_config_hash(indexed_cfg) -> None:
    from snow_sports_rag.gradio_app.components import render_settings_readonly

    html = render_settings_readonly(indexed_cfg, config_hash="deadbeefcafe")
    assert "deadbeefcafe" in html
    assert "pipeline-fake" in html
    assert "chroma" in html
    assert "markdown_header" in html


# ---------------------------------------------------------------------------
# smoke: gradio Blocks is constructable (skip when gradio is not installed)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# empty-index short-circuit + --auto-index
# ---------------------------------------------------------------------------


def _empty_cfg(tmp_path: Path):
    """Return an AppConfig whose Chroma persist dir has zero rows in it."""
    cfg_path = _write_config(tmp_path)
    return load_config(cfg_path, base_dir=tmp_path)


def test_chroma_vector_store_count_zero_on_fresh_dir(tmp_path: Path) -> None:
    store = ChromaVectorStore(tmp_path / "db", "snow_sports_kb")
    assert store.count() == 0


def test_pipeline_short_circuits_when_index_empty(tmp_path: Path) -> None:
    cfg = _empty_cfg(tmp_path)
    pipe = RAGPipeline(cfg)
    assert pipe.index_empty is True

    result = pipe.run("anything at all")
    assert result.index_empty is True
    assert result.cards == []
    assert result.answer is None
    assert result.trace.latency.total_ms >= 0.0


def test_render_evidence_banner_index_empty_takes_priority(tmp_path: Path) -> None:
    from snow_sports_rag.gradio_app.components import (
        render_empty_index_banner,
        render_evidence_banner,
    )
    from snow_sports_rag.pipeline.models import PipelineResult, PipelineTrace

    res = PipelineResult(
        query="anything",
        cards=[],
        answer=None,
        trace=PipelineTrace(query="anything"),
        trace_id="t",
        config_hash="h",
        index_empty=True,
    )
    html = render_evidence_banner(res)
    assert html == render_empty_index_banner()
    assert "not indexed yet" in html
    assert "alpine-banner-info" in html
    assert "snow_sports_rag index" in html
    assert "--auto-index" in html


def test_auto_index_if_empty_builds_then_becomes_noop(tmp_path: Path) -> None:
    from snow_sports_rag.gradio_app.app import _auto_index_if_empty
    from snow_sports_rag.ingest.loader import KnowledgeBaseLoader

    cfg = _empty_cfg(tmp_path)
    kb = Path(cfg.knowledge_base_path)
    (kb / "athletes").mkdir(parents=True, exist_ok=True)
    (kb / "athletes" / "annie.md").write_text(
        "# Annie\n\n## Bio\n\n"
        + ("alpine skier " * 30)
        + "\n\n## Career\n\n"
        + ("mountain ridge " * 30)
        + "\n",
        encoding="utf-8",
    )

    assert KnowledgeBaseLoader(cfg).load_all(), "sanity: fixture doc must load"

    logs: list[str] = []
    first = _auto_index_if_empty(cfg, log=logs.append)
    assert first["action"] == "indexed"
    assert first["chunks"] > 0
    assert first["docs"] >= 1

    pipe = RAGPipeline(cfg)
    assert pipe.index_empty is False

    second = _auto_index_if_empty(cfg, log=logs.append)
    assert second["action"] == "skipped"
    assert second["l2_before"] > 0
    assert second["l1_before"] > 0


def test_launch_parse_args_auto_index_flag(tmp_path: Path) -> None:
    from snow_sports_rag.gradio_app.app import _parse_args

    ns_off = _parse_args([])
    assert ns_off.auto_index is False

    ns_on = _parse_args(["--auto-index"])
    assert ns_on.auto_index is True


def test_build_demo_constructs_blocks(indexed_cfg) -> None:
    pytest.importorskip("gradio")
    from snow_sports_rag.gradio_app import build_demo

    pipe = RAGPipeline(indexed_cfg)
    demo = build_demo(
        indexed_cfg,
        pipeline=pipe,
        trace_logger=TraceLogger(
            Path("/tmp/should-not-be-written.jsonl"),
            enabled=False,
        ),
    )
    assert demo is not None
