from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def save_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")


def deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = value
    return result


def get_default_config_path(env_key: str, algo_key: str) -> Path:
    path = PROJECT_ROOT / "configs" / env_key / f"{algo_key}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Config not found for env={env_key}, algo={algo_key}: {path}")
    return path


def resolve_config(
    config_path: str | None = None,
    env_key: str | None = None,
    algo_key: str | None = None,
    seed: int | None = None,
    train_episodes: int | None = None,
    eval_episodes: int | None = None,
    device: str | None = None,
    extra_overrides: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], Path]:
    if config_path is not None:
        path = Path(config_path).resolve()
    else:
        if env_key is None or algo_key is None:
            raise ValueError("Either config_path or both env_key and algo_key must be provided.")
        path = get_default_config_path(env_key, algo_key).resolve()

    config = load_yaml(path)

    if seed is not None:
        config["seed"] = seed
    if train_episodes is not None:
        config["train_episodes"] = train_episodes
    if eval_episodes is not None:
        config["eval_episodes"] = eval_episodes
    if device is not None:
        config["device"] = device
    if extra_overrides:
        config = deep_update(config, extra_overrides)

    return config, path
