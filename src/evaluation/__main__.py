"""``python -m snow_sports_rag.evaluation`` → sweep CLI."""

from __future__ import annotations

from .sweep import sweep_main

if __name__ == "__main__":
    raise SystemExit(sweep_main())
