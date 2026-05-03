from __future__ import annotations

from datetime import datetime
from pathlib import Path

from .config import PROJECT_ROOT


def create_run_dir(env_key: str, algo_key: str, seed: int, runs_root: str | None = None) -> Path:
    root = Path(runs_root).resolve() if runs_root else PROJECT_ROOT / "runs"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = root / env_key / algo_key / f"{timestamp}_seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def get_latest_run(env_key: str, algo_key: str, runs_root: str | None = None) -> Path:
    root = Path(runs_root).resolve() if runs_root else PROJECT_ROOT / "runs"
    algo_dir = root / env_key / algo_key
    if not algo_dir.exists():
        raise FileNotFoundError(f"No runs found for env={env_key}, algo={algo_key}.")

    candidates = [path for path in algo_dir.iterdir() if path.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"No run directories found in {algo_dir}.")

    return sorted(candidates)[-1]
