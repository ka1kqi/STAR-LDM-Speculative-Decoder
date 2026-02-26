"""Configuration loading and merging utilities."""

from __future__ import annotations

import copy
import yaml
from pathlib import Path
from typing import Any


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config file and return a dict."""
    path = Path(path)
    with open(path) as f:
        cfg = yaml.safe_load(f)
    return cfg or {}


def merge_configs(base: dict, override: dict) -> dict:
    """Recursively merge *override* into a deep-copy of *base*."""
    merged = copy.deepcopy(base)
    for k, v in override.items():
        if k in merged and isinstance(merged[k], dict) and isinstance(v, dict):
            merged[k] = merge_configs(merged[k], v)
        else:
            merged[k] = copy.deepcopy(v)
    return merged
