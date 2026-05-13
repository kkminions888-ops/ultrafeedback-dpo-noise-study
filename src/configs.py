from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_yaml_config(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if key == "inherits":
            continue
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_dicts(dict(merged[key]), value)
        else:
            merged[key] = value
    return merged


def resolve_config(config_path: Path) -> dict[str, Any]:
    config = load_yaml_config(config_path)
    inherits = config.get("inherits")
    if not inherits:
        return config
    base_path = config_path.parent / inherits
    base_config = resolve_config(base_path)
    return _merge_dicts(base_config, config)


def list_experiment_configs(plan_path: Path) -> list[Path]:
    plan = load_yaml_config(plan_path)
    base_dir = plan_path.parent
    return [base_dir / Path(item) for item in plan["experiments"]]


def load_experiment_config(config_path: Path) -> dict[str, Any]:
    return resolve_config(config_path)
