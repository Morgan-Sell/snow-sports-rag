"""Phase 4.2 sweep runner: grid of configs, gold-set retrieval metrics."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import time
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping

import yaml

from ..chunking import chunk_strategy_from_config
from ..config.loader import AppConfig, load_config
from ..embedding import embedding_model_from_config
from ..embedding.model import EmbeddingModel
from ..ingest import KnowledgeBaseLoader
from ..pipeline.trace import compute_config_hash
from ..retrieval import BaselineRetriever, HierarchicalRetriever, IndexBuilder
from ..vectorstore import chroma_l2_l1_stores_from_config, vector_store_from_config
from .config_merge import merged_config_from_flat
from .gold import GoldItem, load_gold_qa
from .metrics import aggregate_query_metrics, per_query_metrics

__all__ = [
    "SweepConfig",
    "SweepResult",
    "load_sweep_grid_yaml",
    "iter_grid_cells",
    "run_sweep_retrieval",
    "sweep_main",
]


@dataclass(frozen=True)
class SweepConfig:
    """Parsed sweep grid definition."""

    seed: int
    axes: dict[str, list[Any]]


@dataclass(frozen=True)
class SweepResult:
    """One completed sweep directory and in-memory summary."""

    run_dir: Path
    metrics_path: Path
    csv_path: Path | None
    rows: list[dict[str, Any]]


def load_sweep_grid_yaml(path: Path) -> SweepConfig:
    """Load ``seed`` and ``axes`` from a YAML sweep grid file.

    Expected shape::

        seed: 0
        axes:
          chunking.strategy: [markdown_header, recursive_char]
          embedding.model_name: [...]
    """
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: root must be a mapping")
    seed = int(raw.get("seed", 0))
    axes = raw.get("axes")
    if not isinstance(axes, dict) or not axes:
        raise ValueError(f"{path}: 'axes' must be a non-empty mapping")
    out_axes: dict[str, list[Any]] = {}
    for k, v in axes.items():
        if not isinstance(k, str) or not k.strip():
            raise ValueError(f"{path}: invalid axis key {k!r}")
        if not isinstance(v, list) or not v:
            raise ValueError(f"{path}: axis {k!r} must be a non-empty list")
        out_axes[k.strip()] = list(v)
    return SweepConfig(seed=seed, axes=out_axes)


def iter_grid_cells(axes: Mapping[str, list[Any]]) -> Iterator[dict[str, Any]]:
    """Yield Cartesian product of axis values as flat override dicts."""
    keys = list(axes.keys())
    vals = [axes[k] for k in keys]
    for combo in itertools.product(*vals):
        yield dict(zip(keys, combo))


def _with_chroma_dir(cfg: AppConfig, cell_dir: Path) -> AppConfig:
    """Point ``vector_store.persist_directory`` at ``cell_dir/chroma``."""
    vs = dict(cfg.vector_store)
    vs["persist_directory"] = str((cell_dir / "chroma").resolve())
    return replace(cfg, vector_store=vs)


def _make_retriever(cfg: AppConfig, embedder: EmbeddingModel) -> Any:
    mode = str(cfg.retrieval.get("mode", "baseline")).strip().lower()
    top_k = int(cfg.retrieval.get("top_k", 8))

    if mode == "hierarchical":
        l2, l1 = chroma_l2_l1_stores_from_config(cfg.vector_store)
        return HierarchicalRetriever(
            embedder,
            l2,
            l1,
            top_k=top_k,
            l1_top_m=int(cfg.retrieval.get("l1_shortlist_m", 5)),
            max_chunks_per_doc=int(cfg.retrieval.get("max_chunks_per_doc", 2)),
            global_fallback=bool(
                cfg.retrieval.get("hierarchical_global_fallback", True)
            ),
        )

    l2 = vector_store_from_config(cfg.vector_store)
    return BaselineRetriever(embedder, l2, top_k=top_k)


def _build_index(cfg: AppConfig, documents: list[Any]) -> int:
    strategy = chunk_strategy_from_config(cfg.chunking)
    embedder = embedding_model_from_config(cfg.embedding)

    mode = str(cfg.retrieval.get("mode", "baseline")).strip().lower()
    if mode == "hierarchical":
        l2, l1 = chroma_l2_l1_stores_from_config(cfg.vector_store)
        builder = IndexBuilder(strategy, embedder, l2, l1_store=l1)
    else:
        l2 = vector_store_from_config(cfg.vector_store)
        builder = IndexBuilder(strategy, embedder, l2, l1_store=None)

    return builder.build(documents)


def eval_retrieval_cell(
    cfg: AppConfig,
    documents: list[Any],
    gold: list[GoldItem],
    cell_dir: Path,
    *,
    k_eval: int | None = None,
) -> dict[str, Any]:
    """Rebuild index under ``cell_dir`` and score ``gold`` retrieval-only."""
    cell_dir.mkdir(parents=True, exist_ok=True)
    cfg_cell = _with_chroma_dir(cfg, cell_dir)
    n_chunks = _build_index(cfg_cell, documents)

    embedder = embedding_model_from_config(cfg_cell.embedding)
    retriever = _make_retriever(cfg_cell, embedder)
    k = int(k_eval if k_eval is not None else cfg_cell.retrieval.get("top_k", 8))

    qrows: list[Any] = []
    latencies: list[float] = []
    for item in gold:
        t0 = time.perf_counter()
        hits = retriever.retrieve(item.question, k=k)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        latencies.append(dt_ms)
        qrows.append(per_query_metrics(item, hits, k=k, latency_ms=dt_ms))

    metrics = aggregate_query_metrics(qrows)
    metrics["n_chunks_indexed"] = n_chunks
    metrics["top_k_eval"] = k
    return metrics


def run_sweep_retrieval(
    base: AppConfig,
    sweep: SweepConfig,
    gold: list[GoldItem],
    documents: list[Any],
    run_dir: Path,
    *,
    write_csv: bool = True,
    k_eval: int | None = None,
) -> SweepResult:
    """Execute each grid cell; write ``metrics.json`` and optional ``summary.csv``."""
    run_dir.mkdir(parents=True, exist_ok=True)
    cells_out: list[dict[str, Any]] = []
    flat_cells = list(iter_grid_cells(sweep.axes))

    for i, flat in enumerate(flat_cells):
        merged = merged_config_from_flat(base, flat)
        cell_id = f"{i:03d}_{compute_config_hash(merged)[:10]}"
        cell_dir = run_dir / "cells" / cell_id
        row: dict[str, Any] = {
            "cell_id": cell_id,
            "flat_overrides": flat,
            "config_hash": compute_config_hash(merged),
        }
        try:
            m = eval_retrieval_cell(
                merged,
                documents,
                gold,
                cell_dir,
                k_eval=k_eval,
            )
            row["status"] = "ok"
            row["metrics"] = m
        except Exception as e:  # noqa: BLE001 — record sweep failures
            row["status"] = "error"
            row["error"] = f"{type(e).__name__}: {e}"
            row["metrics"] = None
        cells_out.append(row)

    payload: dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seed": sweep.seed,
        "axes": sweep.axes,
        "n_cells": len(flat_cells),
        "gold_count": len(gold),
        "knowledge_base": str(base.knowledge_base_path),
        "cells": cells_out,
    }
    metrics_path = run_dir / "metrics.json"
    metrics_path.write_text(
        json.dumps(payload, indent=2, default=str),
        encoding="utf-8",
    )

    csv_path: Path | None = None
    if write_csv and cells_out:
        csv_path = run_dir / "summary.csv"
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    "cell_id",
                    "status",
                    "recall_at_k",
                    "mrr",
                    "ndcg_at_k",
                    "latency_ms_p50",
                    "latency_ms_p95",
                    "n_queries",
                    "n_chunks_indexed",
                    "config_hash",
                ]
            )
            for row in cells_out:
                m = row.get("metrics") or {}
                w.writerow(
                    [
                        row["cell_id"],
                        row.get("status", ""),
                        f'{m.get("recall_at_k", "")}',
                        f'{m.get("mrr", "")}',
                        f'{m.get("ndcg_at_k", "")}',
                        f'{m.get("latency_ms_p50", "")}',
                        f'{m.get("latency_ms_p95", "")}',
                        f'{m.get("n_queries", "")}',
                        f'{m.get("n_chunks_indexed", "")}',
                        row.get("config_hash", ""),
                    ]
                )

    return SweepResult(
        run_dir=run_dir,
        metrics_path=metrics_path,
        csv_path=csv_path,
        rows=cells_out,
    )


def sweep_main(argv: list[str] | None = None) -> int:
    """CLI entry for ``uv run snow-sports-rag-sweep``."""
    parser = argparse.ArgumentParser(
        description="Run retrieval metrics over a grid of RAG configs (Phase 4.2).",
    )
    parser.add_argument(
        "--grid",
        type=Path,
        default=Path("evaluation/sweep_grid.yaml"),
        help="YAML file with seed and axes (default: evaluation/sweep_grid.yaml)",
    )
    parser.add_argument(
        "--base-config",
        type=Path,
        default=None,
        help="Base YAML config (default: configs/default.yaml under --base-dir)",
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=None,
        help="Working directory for resolving relative paths (default: cwd)",
    )
    parser.add_argument(
        "--gold",
        type=Path,
        default=None,
        help="Gold JSONL path (default: evaluation/gold_qa.jsonl under base-dir)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Run output directory (default: evaluation/runs/<timestamp>)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Override retrieval top-k for eval (default: each cell's retrieval.top_k)",
    )
    parser.add_argument(
        "--no-csv",
        action="store_true",
        help="Do not write summary.csv",
    )
    args = parser.parse_args(argv)

    base = args.base_dir.resolve() if args.base_dir else Path.cwd()
    cfg_path = args.base_config
    if cfg_path is None:
        cfg_path = base / "configs/default.yaml"
    elif not cfg_path.is_absolute():
        cfg_path = (base / cfg_path).resolve()

    grid_path = args.grid if args.grid.is_absolute() else (base / args.grid).resolve()
    if not grid_path.is_file():
        raise SystemExit(f"Grid file not found: {grid_path}")

    gold_path = args.gold
    if gold_path is None:
        gold_path = base / "evaluation/gold_qa.jsonl"
    elif not gold_path.is_absolute():
        gold_path = (base / gold_path).resolve()

    sweep_cfg = load_sweep_grid_yaml(grid_path)
    app_cfg = load_config(cfg_path, base_dir=base)
    gold_items = load_gold_qa(gold_path)
    loader = KnowledgeBaseLoader(app_cfg)
    documents = loader.load_all()

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = args.output
    if run_dir is None:
        run_dir = base / "evaluation" / "runs" / ts
    else:
        run_dir = run_dir if run_dir.is_absolute() else (base / run_dir).resolve()

    res = run_sweep_retrieval(
        app_cfg,
        sweep_cfg,
        gold_items,
        documents,
        run_dir,
        write_csv=not args.no_csv,
        k_eval=args.top_k,
    )
    print(f"Wrote {res.metrics_path}")
    if res.csv_path:
        print(f"Wrote {res.csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(sweep_main())
