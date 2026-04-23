"""Tests for Phase 4.3 trace aggregation and sweep metrics comparison."""

from __future__ import annotations

import json
from pathlib import Path

from snow_sports_rag.evaluation.trace_analyze import (
    aggregate_traces_file,
    compare_sweep_metrics,
    trace_analyze_main,
)


def test_aggregate_traces_by_config_hash(tmp_path: Path) -> None:
    f = tmp_path / "t.jsonl"
    f.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "type": "query",
                        "trace_id": "a1",
                        "config_hash": "h1",
                        "payload": {
                            "refused": False,
                            "latency_ms": {"total_ms": 100.0},
                        },
                    }
                ),
                json.dumps(
                    {
                        "type": "query",
                        "trace_id": "a2",
                        "config_hash": "h1",
                        "payload": {
                            "refused": True,
                            "latency_ms": {"total_ms": 50.0},
                        },
                    }
                ),
                json.dumps(
                    {
                        "type": "feedback",
                        "trace_id": "a1",
                        "config_hash": "h1",
                        "payload": {"feedback": "up"},
                    }
                ),
                "not valid json{",
            ]
        ),
        encoding="utf-8",
    )
    out = aggregate_traces_file(f)
    assert out["n_malformed"] == 1
    b = out["by_config_hash"]["h1"]
    assert b["n_queries"] == 2
    assert b["refused_rate"] == 0.5
    assert b["latency_total_ms_mean"] == 75.0
    assert b["feedback_up"] == 1
    assert b["n_feedback"] == 1


def test_compare_sweep_metrics_regression(tmp_path: Path) -> None:
    def write_metrics(name: str, recall: float) -> Path:
        p = tmp_path / name
        p.write_text(
            json.dumps(
                {
                    "cells": [
                        {
                            "cell_id": "0_x",
                            "flat_overrides": {"chunking.chunk_size": 400},
                            "status": "ok",
                            "metrics": {
                                "recall_at_k": recall,
                                "mrr": 0.5,
                                "ndcg_at_k": 0.5,
                            },
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        return p

    b = write_metrics("b.json", 0.8)
    c = write_metrics("c.json", 0.5)
    r = compare_sweep_metrics(b, c, min_delta=0.0)
    assert r["regression_count"] == 1
    assert r["regressions"][0]["metric"] == "recall_at_k"

    c2 = write_metrics("c2.json", 0.79)
    r2 = compare_sweep_metrics(b, c2, min_delta=0.05)
    assert r2["regression_count"] == 0


def test_compare_sweep_mismatched_keys(tmp_path: Path) -> None:
    a = tmp_path / "a.json"
    a.write_text(
        json.dumps(
            {"cells": [{"flat_overrides": {"x": 1}, "metrics": {"recall_at_k": 1.0}}]}
        ),
        encoding="utf-8",
    )
    b2 = tmp_path / "b.json"
    b2.write_text(
        json.dumps(
            {"cells": [{"flat_overrides": {"x": 2}, "metrics": {"recall_at_k": 1.0}}]}
        ),
        encoding="utf-8",
    )
    r = compare_sweep_metrics(a, b2, min_delta=0.0)
    assert r["n_matched"] == 0
    assert r["unmatched_baseline"]


def test_cli_trace_analyze_and_compare_exit(tmp_path: Path) -> None:
    t = tmp_path / "t.jsonl"
    t.write_text(
        '{"type":"query","config_hash":"z","trace_id":"1",'
        '"payload":{"latency_ms":{"total_ms":1.0},"refused":false}}\n',
        encoding="utf-8",
    )

    assert trace_analyze_main(["--traces", str(t)]) == 0

    b = tmp_path / "b.json"
    b.write_text(
        json.dumps(
            {
                "cells": [
                    {
                        "flat_overrides": {"k": 1},
                        "metrics": {
                            "recall_at_k": 0.9,
                            "mrr": 0.5,
                            "ndcg_at_k": 0.5,
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    c = tmp_path / "c.json"
    c.write_text(
        json.dumps(
            {
                "cells": [
                    {
                        "flat_overrides": {"k": 1},
                        "metrics": {
                            "recall_at_k": 0.1,
                            "mrr": 0.5,
                            "ndcg_at_k": 0.5,
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    assert (
        trace_analyze_main(
            [
                "--compare-metrics",
                str(b),
                str(c),
                "--fail-on-regression",
            ],
        )
        == 1
    )
