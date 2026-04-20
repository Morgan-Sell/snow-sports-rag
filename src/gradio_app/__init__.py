"""Phase 3 Gradio frontend.

Launch with ``uv run python -m snow_sports_rag.gradio_app``.
"""

from .app import build_demo, launch

__all__ = ["build_demo", "launch"]
