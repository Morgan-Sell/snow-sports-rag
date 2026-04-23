"""Phase 4.3: aggregate Gradio ``traces.jsonl`` by ``config_hash``.

Optional: diff two sweep ``metrics.json`` files.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

__all__ = [
    "compare_sweep_metrics",
    "iter_trace_records",
    "aggregate_traces_file",
    "trace_analyze_main",
]


def iter_trace_records(
    path: Path,
) -> Iterator[tuple[dict[str, Any] | None, str | None]]:
    """Yield ``(record, error)`` per line: ``(dict, None)`` or ``(None, err)``."""
    text = path.read_text(encoding="utf-8")
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        try:
            rec = json.loads(stripped)
        except json.JSONDecodeError as e:
            yield None, f"json: {e}"
            continue
        if not isinstance(rec, dict):
            yield None, f"expected object, got {type(rec).__name__}"
            continue
        yield rec, None


@dataclass
class _Bucket:
    n_queries: int = 0
    n_feedback: int = 0
    feedback_up: int = 0
    feedback_down: int = 0
    refused: int = 0
    latency_totals: list[float] = field(default_factory=list)
    trace_ids_with_feedback: set[str] = field(default_factory=set)

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "n_queries": self.n_queries,
            "n_feedback": self.n_feedback,
            "feedback_up": self.feedback_up,
            "feedback_down": self.feedback_down,
            "feedback_traces": len(self.trace_ids_with_feedback),
        }
        if self.n_queries:
            out["refused_rate"] = round(self.refused / self.n_queries, 6)
        if self.latency_totals:
            t = self.latency_totals
            out["latency_total_ms_mean"] = round(sum(t) / len(t), 4)
        return out


def aggregate_traces_file(path: Path) -> dict[str, Any]:
    """Build summary stats keyed by ``config_hash`` from a JSONL trace file."""
    if not path.is_file():
        raise FileNotFoundError(f"Traces file not found: {path}")

    n_malformed = 0
    buckets: dict[str, _Bucket] = {}
    n_lines = 0

    for rec, err in iter_trace_records(path):
        n_lines += 1
        if err is not None or rec is None:
            n_malformed += 1
            continue
        t = str(rec.get("type", ""))
        ch = str(rec.get("config_hash", ""))
        if not ch:
            n_malformed += 1
            continue
        b = buckets.setdefault(ch, _Bucket())
        if t == "query":
            b.n_queries += 1
            payload = rec.get("payload")
            if isinstance(payload, dict):
                if payload.get("refused") is True:
                    b.refused += 1
                lat = payload.get("latency_ms")
                if isinstance(lat, dict) and "total_ms" in lat:
                    try:
                        b.latency_totals.append(float(lat["total_ms"]))
                    except (TypeError, ValueError):
                        pass
        elif t == "feedback":
            tid = str(rec.get("trace_id", ""))
            payload = rec.get("payload")
            fb = ""
            if isinstance(payload, dict):
                fb = str(payload.get("feedback", "")).strip().lower()
            b.n_feedback += 1
            if tid:
                b.trace_ids_with_feedback.add(tid)
            if fb == "up":
                b.feedback_up += 1
            elif fb == "down":
                b.feedback_down += 1
        else:
            n_malformed += 1

    by_hash = {k: v.to_dict() for k, v in sorted(buckets.items())}
    return {
        "traces_path": str(path.resolve()),
        "n_lines_non_empty": n_lines,
        "n_malformed": n_malformed,
        "n_config_hashes": len(by_hash),
        "by_config_hash": by_hash,
    }


def _cell_metrics_key(cell: dict[str, Any]) -> str:
    """Stable key to match cells across two sweep runs."""
    fo = cell.get("flat_overrides")
    if isinstance(fo, dict) and fo:
        return json.dumps(fo, sort_keys=True, default=str)
    return str(cell.get("cell_id", ""))


def compare_sweep_metrics(
    baseline_path: Path,
    current_path: Path,
    *,
    min_delta: float = 0.0,
) -> dict[str, Any]:
    """Compare per-cell metrics between two sweep ``metrics.json`` files.

    A **regression** is recorded when a numeric metric in ``current`` is
    strictly lower than ``baseline`` by more than ``min_delta`` (default: any
    drop).

    Parameters
    ----------
    baseline_path, current_path : Path
        Each must be a ``metrics.json`` with a top-level ``cells`` list.
    min_delta : float, optional
        Ignores drops smaller than this (e.g. ``0.01`` for 1 point on [0,1] metrics).

    Returns
    -------
    dict
        ``matched``, ``unmatched_*``, ``cells`` (per-key deltas), ``regressions``.
    """
    base_raw = json.loads(baseline_path.read_text(encoding="utf-8"))
    cur_raw = json.loads(current_path.read_text(encoding="utf-8"))
    b_cells: list[dict[str, Any]] = list(base_raw.get("cells") or [])
    c_cells: list[dict[str, Any]] = list(cur_raw.get("cells") or [])

    b_map: dict[str, dict[str, Any]] = {}
    for c in b_cells:
        b_map[_cell_metrics_key(c)] = c
    c_map: dict[str, dict[str, Any]] = {}
    for c in c_cells:
        c_map[_cell_metrics_key(c)] = c

    keys = sorted(set(b_map) & set(c_map))
    only_b = sorted(set(b_map) - set(c_map))
    only_c = sorted(set(c_map) - set(b_map))

    metric_names = ("recall_at_k", "mrr", "ndcg_at_k")
    cells_out: list[dict[str, Any]] = []
    regressions: list[dict[str, Any]] = []

    for k in keys:
        br = b_map[k]
        cr = c_map[k]
        b_m = br.get("metrics") or {}
        c_m = cr.get("metrics") or {}
        row: dict[str, Any] = {
            "key": k,
            "cell_id_baseline": br.get("cell_id"),
            "cell_id_current": cr.get("cell_id"),
            "deltas": {},
        }
        for m in metric_names:
            try:
                bv = float(b_m.get(m, 0.0))
                cv = float(c_m.get(m, 0.0))
            except (TypeError, ValueError):
                continue
            delta = cv - bv
            row["deltas"][m] = round(delta, 6)
            if cv < bv - min_delta:
                regressions.append(
                    {
                        "key": k,
                        "metric": m,
                        "baseline": bv,
                        "current": cv,
                        "delta": round(delta, 6),
                    }
                )
        cells_out.append(row)

    return {
        "baseline": str(baseline_path.resolve()),
        "current": str(current_path.resolve()),
        "n_matched": len(keys),
        "unmatched_baseline": only_b,
        "unmatched_current": only_c,
        "cells": cells_out,
        "regressions": regressions,
        "regression_count": len(regressions),
    }


def trace_analyze_main(argv: list[str] | None = None) -> int:
    """CLI for trace aggregation and optional metrics comparison."""
    p = argparse.ArgumentParser(
        description="Analyze RAG JSONL traces (Phase 4.3) or compare sweep metrics.",
    )
    p.add_argument(
        "--traces",
        type=Path,
        default=Path(".rag_traces/traces.jsonl"),
        help="Path to traces.jsonl (default: .rag_traces/traces.jsonl)",
    )
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Write JSON to this path instead of stdout",
    )
    p.add_argument(
        "--compare-metrics",
        nargs=2,
        metavar=("BASELINE", "CURRENT"),
        type=Path,
        help=(
            "Compare two snow-sports-rag-sweep metrics.json files; writes JSON report."
        ),
    )
    p.add_argument(
        "--min-delta",
        type=float,
        default=0.0,
        help="Minimum drop to flag a regression in --compare-metrics (default: 0).",
    )
    p.add_argument(
        "--fail-on-regression",
        action="store_true",
        help="Exit 1 if --compare-metrics finds any regression.",
    )
    args = p.parse_args(argv)

    if args.compare_metrics:
        report = compare_sweep_metrics(
            args.compare_metrics[0],
            args.compare_metrics[1],
            min_delta=args.min_delta,
        )
        text = json.dumps(report, indent=2, default=str)
        if args.output is not None:
            args.output.write_text(text, encoding="utf-8")
        else:
            print(text)
        if args.fail_on_regression and report["regression_count"] > 0:
            return 1
        return 0

    agg = aggregate_traces_file(args.traces)
    text = json.dumps(agg, indent=2, default=str)
    if args.output is not None:
        args.output.write_text(text, encoding="utf-8")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(trace_analyze_main())
