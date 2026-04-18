from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

_ENV_KB_PATH = "SNOW_SPORTS_RAG_KNOWLEDGE_BASE_PATH"

_DEFAULT_SUBSECTIONS: dict[str, Any] = {
    "chunking": {
        "strategy": "markdown_header",
        "chunk_size": 512,
        "chunk_overlap": 64,
        "min_section_chars": 0,
        "recursive_separators": ["\n\n", "\n", " ", ""],
    },
    "embedding": {
        "backend": "sentence_transformers",
        "model_name": "sentence-transformers/all-MiniLM-L6-v2",
        "normalize": True,
    },
    "vector_store": {
        "backend": "chroma",
        "persist_directory": ".rag_index/chroma",
        "collection_name": "snow_sports_kb",
        "l1_collection_name": None,
    },
    "retrieval": {
        "top_k": 8,
        "mode": "baseline",
        "l1_shortlist_m": 5,
        "max_chunks_per_doc": 2,
        "hierarchical_global_fallback": True,
    },
    "rerank": {
        "enabled": False,
        "backend": "cross_encoder",
        "model_name": "cross-encoder/ms-marco-MiniLM-L-6-v2",
        "top_n_pre_rerank": 48,
        "top_n_in": 30,
        "top_k_out": 5,
        "device": None,
        "openai_model": None,
        "openai_base_url": None,
        "openai_api_key": None,
        "openai_api_key_env": None,
        "openai_temperature": None,
        "openai_timeout_s": None,
        "anthropic_model": "claude-3-5-haiku-20241022",
        "anthropic_api_key": None,
        "anthropic_api_key_env": "ANTHROPIC_API_KEY",
        "anthropic_max_tokens": 1024,
        "anthropic_timeout_s": 120.0,
    },
    "llm": {
        "provider": "openai_compatible",
        "model": "gpt-4o-mini",
        "temperature": 0.1,
        "base_url": "https://api.openai.com/v1",
        "api_key_env": "OPENAI_API_KEY",
        "timeout_s": 60.0,
    },
    "query_expansion": {
        "enabled": False,
        "num_paraphrases": 3,
        "fusion": "max_score",
        "rrf_k": 60,
    },
    "logging": {"level": "INFO"},
}


@dataclass(frozen=True)
class AppConfig:
    """Immutable application settings after YAML load and environment overrides.

    Attributes
    ----------
    knowledge_base_path : pathlib.Path
        Absolute path to the Markdown corpus root directory.
    chunking : Mapping[str, Any]
        Chunking-related options for later RAG phases.
    embedding : Mapping[str, Any]
        Bi-encoder / embedding model options.
    vector_store : Mapping[str, Any]
        Vector index backend (Chroma path, collection name).
    retrieval : Mapping[str, Any]
        Vector search and hierarchical retrieval parameters.
    rerank : Mapping[str, Any]
        Phase 2.3 reranking (backend, ``top_n_in`` / ``top_k_out``, models).
    llm : Mapping[str, Any]
        LLM provider and generation parameters.
    query_expansion : Mapping[str, Any]
        Phase 2.2 multi-query expansion and fusion options.
    logging : Mapping[str, Any]
        Log level and related settings.
    """

    knowledge_base_path: Path
    chunking: Mapping[str, Any]
    embedding: Mapping[str, Any]
    vector_store: Mapping[str, Any]
    retrieval: Mapping[str, Any]
    rerank: Mapping[str, Any]
    llm: Mapping[str, Any]
    query_expansion: Mapping[str, Any]
    logging: Mapping[str, Any]


def _deep_merge_defaults(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Merge known config sections with built-in defaults (later keys win).

    Parameters
    ----------
    raw : Mapping[str, Any]
        Mapping loaded from YAML at the top level.

    Returns
    -------
    dict[str, Any]
        Copy of ``raw`` with each subsection shallow-merged into defaults.
    """
    merged: dict[str, Any] = dict(raw)
    for key, default_val in _DEFAULT_SUBSECTIONS.items():
        section = merged.get(key)
        if not isinstance(section, dict):
            merged[key] = dict(default_val)
        else:
            merged[key] = {**default_val, **section}
    return merged


def load_config(
    config_path: Path | None = None,
    *,
    base_dir: Path | None = None,
) -> AppConfig:
    """Load YAML configuration and resolve paths.

    Relative ``knowledge_base_path`` values in YAML are joined with ``base_dir``
    when provided, otherwise with :attr:`pathlib.Path.cwd`.

    The environment variable ``SNOW_SPORTS_RAG_KNOWLEDGE_BASE_PATH``, when set,
    overrides the YAML ``knowledge_base_path`` value.

    Parameters
    ----------
    config_path : pathlib.Path or None, optional
        YAML file to read. If ``None``, uses ``configs/default.yaml`` relative
        to the current working directory.
    base_dir : pathlib.Path or None, optional
        Anchor for resolving relative ``knowledge_base_path`` entries. If
        ``None``, :attr:`pathlib.Path.cwd` is used.

    Returns
    -------
    AppConfig
        Parsed, merged, and path-resolved configuration.

    Raises
    ------
    FileNotFoundError
        If ``config_path`` does not exist.
    ValueError
        If the YAML document root is not a ``dict``.
    TypeError
        If ``knowledge_base_path`` in YAML is not a string.
    """
    if config_path is None:
        config_path = Path("configs/default.yaml")

    path = config_path
    if not path.is_file():
        raise FileNotFoundError(f"Config not found: {path.resolve()}")

    with path.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    if not isinstance(raw, dict):
        raise ValueError("Root of YAML config must be a mapping")

    merged = _deep_merge_defaults(raw)

    kb_raw = merged.get("knowledge_base_path", "knowledge-base")
    if not isinstance(kb_raw, str):
        raise TypeError("knowledge_base_path must be a string")

    env_kb = os.environ.get(_ENV_KB_PATH)
    if env_kb is not None:
        kb_raw = env_kb

    root = base_dir if base_dir is not None else Path.cwd()
    kb_path = Path(kb_raw)
    if not kb_path.is_absolute():
        kb_path = (root / kb_path).resolve()

    vs = merged.get("vector_store")
    if isinstance(vs, dict):
        pd_raw = vs.get("persist_directory")
        if pd_raw is not None:
            pd_path = Path(pd_raw) if not isinstance(pd_raw, Path) else pd_raw
            if not pd_path.is_absolute():
                pd_path = (root / pd_path).resolve()
            else:
                pd_path = pd_path.resolve()
            merged["vector_store"] = {**vs, "persist_directory": pd_path}

    return AppConfig(
        knowledge_base_path=kb_path,
        chunking=merged["chunking"],
        embedding=merged["embedding"],
        vector_store=merged["vector_store"],
        retrieval=merged["retrieval"],
        rerank=merged["rerank"],
        llm=merged["llm"],
        query_expansion=merged["query_expansion"],
        logging=merged["logging"],
    )
