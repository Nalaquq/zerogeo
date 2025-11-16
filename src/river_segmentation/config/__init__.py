"""
Configuration management for annotation pipeline.

This module provides:
- YAML-based configuration
- Type-safe config dataclasses
- Validation and defaults
- Config merging and overrides
"""

from .schema import (
    AnnotationConfig,
    ProjectConfig,
    TilingConfig,
    GroundingDINOConfig,
    SAMConfig,
    ReviewConfig,
    ExportConfig,
)
from .loader import ConfigLoader, load_config, save_config, validate_config

__all__ = [
    "AnnotationConfig",
    "ProjectConfig",
    "TilingConfig",
    "GroundingDINOConfig",
    "SAMConfig",
    "ReviewConfig",
    "ExportConfig",
    "ConfigLoader",
    "load_config",
    "save_config",
    "validate_config",
]
