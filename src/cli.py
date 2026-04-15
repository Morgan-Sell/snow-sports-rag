from __future__ import annotations

import argparse
from pathlib import Path

from .config import load_config
from .ingest import KnowledgeBaseLoader


def main() -> None:
    """Run the Phase 0 CLI: parse ``sys.argv``, load config, ingest Markdown.

    Prints a document count; use ``--list`` to print ``doc_id``, ``entity_type``,
    and ``title`` per file.
    """
    parser = argparse.ArgumentParser(description="Snow Sports RAG (Phase 0 — ingest)")
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
    args = parser.parse_args()

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
