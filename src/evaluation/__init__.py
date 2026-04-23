"""Evaluation datasets, metrics, and sweep runner (Phase 4)."""

from .gold import GoldItem, default_gold_qa_path, load_gold_qa
from .sweep import SweepConfig, SweepResult, load_sweep_grid_yaml, run_sweep_retrieval
from .trace_analyze import (
    aggregate_traces_file,
    compare_sweep_metrics,
    trace_analyze_main,
)

__all__ = [
    "GoldItem",
    "SweepConfig",
    "SweepResult",
    "aggregate_traces_file",
    "compare_sweep_metrics",
    "default_gold_qa_path",
    "load_gold_qa",
    "load_sweep_grid_yaml",
    "run_sweep_retrieval",
    "trace_analyze_main",
]
