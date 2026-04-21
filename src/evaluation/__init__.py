"""Evaluation datasets and metrics (Phase 4)."""

from .gold import GoldItem, default_gold_qa_path, load_gold_qa

__all__ = ["GoldItem", "default_gold_qa_path", "load_gold_qa"]
