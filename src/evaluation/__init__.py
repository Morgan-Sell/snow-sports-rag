"""Evaluation datasets, metrics, and sweep runner (Phase 4)."""

from .gold import GoldItem, default_gold_qa_path, load_gold_qa
from .sweep import SweepConfig, SweepResult, load_sweep_grid_yaml, run_sweep_retrieval

__all__ = [
    "GoldItem",
    "SweepConfig",
    "SweepResult",
    "default_gold_qa_path",
    "load_gold_qa",
    "load_sweep_grid_yaml",
    "run_sweep_retrieval",
]
