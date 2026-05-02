"""Configuration loading utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import yaml


def project_root() -> Path:
    """Get the root directory of the project."""
    return Path(__file__).resolve().parents[2]


def _load_yaml(path: Path) -> Dict[str, Any]:
    """Load YAML configuration file."""
    if not path.exists():
        return {}

    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}

    if not isinstance(loaded, dict):
        raise ValueError(f"YAML config at {path} must be an object")

    return loaded


def load_model_config(path: Optional[str] = None) -> Dict[str, Any]:
    """Load model configuration from YAML."""
    config_path = Path(path) if path else project_root() / "configs" / "model_config.yaml"
    return _load_yaml(config_path)


def load_risk_rules(path: Optional[str] = None) -> Dict[str, Any]:
    """Load risk rules configuration from YAML."""
    rules_path = Path(path) if path else project_root() / "configs" / "risk_rules.yaml"
    return _load_yaml(rules_path)
