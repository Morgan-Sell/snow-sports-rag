"""
Pytest plugin context: bootstrap the ``snow_sports_rag`` package for test collection.

If ``pytest`` is invoked without an editable install, Python would not resolve
``snow_sports_rag`` to the flat ``src/`` tree; this module registers the package
early so imports in tests succeed.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _ensure_package() -> None:
    """Load ``snow_sports_rag`` from ``src/`` when it is not already imported.

    Notes
    -----
    Mutates ``sys.modules`` so subsequent ``import snow_sports_rag`` succeeds.

    Raises
    ------
    ImportError
        If ``src/__init__.py`` is missing or cannot be executed.
    """
    if "snow_sports_rag" in sys.modules:
        return
    root = Path(__file__).resolve().parents[1]
    src = root / "src"
    init = src / "__init__.py"
    spec = importlib.util.spec_from_file_location(
        "snow_sports_rag",
        init,
        submodule_search_locations=[str(src)],
    )
    if spec is None or spec.loader is None:
        msg = f"Cannot load snow_sports_rag from {src}"
        raise ImportError(msg)
    module = importlib.util.module_from_spec(spec)
    sys.modules["snow_sports_rag"] = module
    spec.loader.exec_module(module)


_ensure_package()
