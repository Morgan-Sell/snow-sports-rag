"""Configuration loading: YAML defaults, merge rules, and environment overrides."""

from .coalesce import coalesce_openai_timeout_s
from .loader import AppConfig, load_config

__all__ = ["AppConfig", "coalesce_openai_timeout_s", "load_config"]
