"""
Deployment entrypoint for the Snow Sports RAG application.

Examples
--------
Run from the repository root after ``uv sync`` (editable install)::

    uv run python app/main.py --list

Notes
-----
Library implementation lives under ``src/`` with import name
``snow_sports_rag``. Additional process entrypoints (e.g. Gradio) can live in
``app/`` in later phases.
"""

from __future__ import annotations

from snow_sports_rag.cli import main

if __name__ == "__main__":
    main()
