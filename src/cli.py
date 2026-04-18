from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .chunking import chunk_strategy_from_config
from .config import load_config
from .embedding import embedding_model_from_config
from .ingest import KnowledgeBaseLoader
from .llm import llm_client_from_config
from .rerank import reranker_from_config
from .retrieval import (
    BaselineRetriever,
    HierarchicalRetriever,
    IndexBuilder,
    QueryExpander,
)
from .vectorstore import chroma_l2_l1_stores_from_config, vector_store_from_config


def _cmd_ingest(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(description="Ingest knowledge base (Phase 0)")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to YAML config (default: configs/default.yaml)",
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=None,
        help="Resolve relative paths from this directory (default: cwd)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Print doc_id and entity_type for each document",
    )
    args = parser.parse_args(argv)
    base = args.base_dir.resolve() if args.base_dir else Path.cwd()
    cfg_path = args.config
    if cfg_path is None:
        cfg_path = base / "configs/default.yaml"
    elif not cfg_path.is_absolute():
        cfg_path = (base / cfg_path).resolve()
    cfg = load_config(cfg_path, base_dir=base)
    loader = KnowledgeBaseLoader(cfg)
    docs = loader.load_all()

    print(f"Loaded {len(docs)} documents from {cfg.knowledge_base_path}")
    if args.list:
        for d in docs:
            print(f"{d.doc_id}\t{d.entity_type}\t{d.title}")


def _cmd_index(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(description="Rebuild vector index (Phase 1.4)")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to YAML config (default: configs/default.yaml)",
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=None,
        help="Resolve relative paths from this directory (default: cwd)",
    )
    args = parser.parse_args(argv)
    base = args.base_dir.resolve() if args.base_dir else Path.cwd()
    cfg_path = args.config
    if cfg_path is None:
        cfg_path = base / "configs/default.yaml"
    elif not cfg_path.is_absolute():
        cfg_path = (base / cfg_path).resolve()
    cfg = load_config(cfg_path, base_dir=base)
    loader = KnowledgeBaseLoader(cfg)
    docs = loader.load_all()
    strategy = chunk_strategy_from_config(cfg.chunking)
    embedder = embedding_model_from_config(cfg.embedding)
    l2_store, l1_store = chroma_l2_l1_stores_from_config(cfg.vector_store)
    builder = IndexBuilder(strategy, embedder, l2_store, l1_store=l1_store)
    n = builder.build(docs)
    nd = len(docs)
    print(f"Indexed {n} L2 chunks and {nd} L1 summaries from {nd} documents")


def _cmd_query(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(description="Dense baseline query (Phase 1.4)")
    parser.add_argument(
        "query_text",
        type=str,
        help="Natural language query",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to YAML config (default: configs/default.yaml)",
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=None,
        help="Resolve relative paths from this directory (default: cwd)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Override retrieval.top_k from config",
    )
    parser.add_argument(
        "--hierarchical",
        action="store_true",
        help="Use Phase 2.1 hierarchical retrieval (L1 shortlist → L2)",
    )
    parser.add_argument(
        "--expand-queries",
        action="store_true",
        help=(
            "Use Phase 2.2 LLM query expansion + fusion "
            "(see query_expansion in config)"
        ),
    )
    parser.add_argument(
        "--no-rerank",
        action="store_true",
        help="Disable Phase 2.3 reranking even if rerank.enabled is true",
    )
    args = parser.parse_args(argv)
    base = args.base_dir.resolve() if args.base_dir else Path.cwd()
    cfg_path = args.config
    if cfg_path is None:
        cfg_path = base / "configs/default.yaml"
    elif not cfg_path.is_absolute():
        cfg_path = (base / cfg_path).resolve()
    cfg = load_config(cfg_path, base_dir=base)
    embedder = embedding_model_from_config(cfg.embedding)
    display_top_k = (
        args.top_k if args.top_k is not None else int(cfg.retrieval["top_k"])
    )
    rer_cfg = cfg.rerank
    rerank_on = bool(rer_cfg.get("enabled")) and not args.no_rerank
    n_in = int(rer_cfg.get("top_n_in", 30))
    pre = int(rer_cfg.get("top_n_pre_rerank", 48))
    qe_cfg = cfg.query_expansion
    use_expand = args.expand_queries or bool(qe_cfg.get("enabled"))
    retrieve_pool = display_top_k
    if rerank_on:
        retrieve_pool = max(retrieve_pool, n_in)
    if use_expand:
        retrieve_pool = max(retrieve_pool, pre)

    mode = str(cfg.retrieval.get("mode", "")).lower()
    hierarchical = args.hierarchical or mode == "hierarchical"
    if hierarchical:
        l2_store, l1_store = chroma_l2_l1_stores_from_config(cfg.vector_store)
        retriever = HierarchicalRetriever(
            embedder,
            l2_store,
            l1_store,
            top_k=retrieve_pool,
            l1_top_m=int(cfg.retrieval.get("l1_shortlist_m", 5)),
            max_chunks_per_doc=int(cfg.retrieval.get("max_chunks_per_doc", 2)),
            global_fallback=bool(
                cfg.retrieval.get("hierarchical_global_fallback", True)
            ),
        )
    else:
        store = vector_store_from_config(cfg.vector_store)
        retriever = BaselineRetriever(embedder, store, top_k=retrieve_pool)

    if use_expand:
        llm = llm_client_from_config(cfg.llm)
        retriever = QueryExpander(
            retriever,
            llm,
            enabled=True,
            num_paraphrases=int(qe_cfg.get("num_paraphrases", 3)),
            fusion=str(qe_cfg.get("fusion", "max_score")),
            rrf_k=int(qe_cfg.get("rrf_k", 60)),
            top_n_fused=int(cfg.rerank.get("top_n_pre_rerank", 48)),
            default_inner_k=retrieve_pool,
        )

    hits = retriever.retrieve(args.query_text)
    if rerank_on:
        k_out = int(rer_cfg.get("top_k_out", 5))
        reranker = reranker_from_config(cfg.rerank, llm=cfg.llm)
        hits = reranker.rerank(args.query_text, hits[:n_in], top_k=k_out)
    else:
        hits = hits[:display_top_k]
    for i, h in enumerate(hits, start=1):
        print(
            f"{i}\t{h.similarity:.6f}\t{h.doc_id}\t{h.section_path}\t{h.chunk_index}"
        )
        print(h.text[:500] + ("…" if len(h.text) > 500 else ""))


def main() -> None:
    """CLI entry: ``ingest`` (default), ``index``, or ``query``."""
    argv = sys.argv[1:]
    if argv and argv[0] == "index":
        _cmd_index(argv[1:])
    elif argv and argv[0] == "query":
        _cmd_query(argv[1:])
    else:
        _cmd_ingest(argv)
