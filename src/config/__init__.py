"""Configuration loading: YAML defaults, merge rules, and environment overrides."""

from .loader import AppConfig, load_config

__all__ = ["AppConfig", "load_config"]
