from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from rl_lab.core.config import load_yaml
from rl_lab.core.experiment import get_latest_run
from rl_lab.core.registry import get_algorithm_class, get_env_spec
from rl_lab.utils.logging import save_json


def _find_checkpoint(run_dir: Path) -> Path:
    for name in ("checkpoint.pt", "checkpoint.npz"):
        candidate = run_dir / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No checkpoint found in {run_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a saved Gymnasium Lab run.")
    parser.add_argument("--run", type=str, default=None, help="Path to a run directory.")
    parser.add_argument("--env", type=str, default=None, help="Environment key. Used with --algo to load latest run.")
    parser.add_argument("--algo", type=str, default=None, help="Algorithm key. Used with --env to load latest run.")
    parser.add_argument("--num-episodes", type=int, default=20, help="Number of evaluation episodes.")
    parser.add_argument("--seed", type=int, default=None, help="Evaluation seed override.")
    parser.add_argument("--render", action="store_true", help="Render the environment during evaluation.")
    parser.add_argument("--record-path", type=str, default=None, help="Optional output path for a rollout GIF or video.")
    parser.add_argument("--fps", type=int, default=30, help="Frame rate used when saving a rollout.")
    parser.add_argument("--runs-root", type=str, default=None, help="Custom root directory where runs are stored.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.run is not None:
        run_dir = Path(args.run).resolve()
    else:
        if args.env is None or args.algo is None:
            parser.error("Either --run or both --env and --algo must be provided.")
        run_dir = get_latest_run(args.env, args.algo, runs_root=args.runs_root)

    config = load_yaml(run_dir / "config.yaml")
    env_spec = get_env_spec(config["env"])
    algo_cls = get_algorithm_class(env_spec, config["algo"])
    algorithm = algo_cls(env_spec, config, device=config.get("device", "cpu"))
    result = algorithm.evaluate(
        checkpoint_path=_find_checkpoint(run_dir),
        num_episodes=args.num_episodes,
        seed=args.seed,
        render=args.render,
        record_path=Path(args.record_path).resolve() if args.record_path else None,
        fps=args.fps,
    )
    result["evaluated_at"] = datetime.now().isoformat(timespec="seconds")
    save_json(run_dir / "evaluation_latest.json", result)

    print(f"Run: {run_dir}")
    print(f"Average return: {result['avg_return']:.3f}")
    print(f"Average length: {result['avg_length']:.3f}")
    if args.record_path:
        print(f"Recorded rollout: {Path(args.record_path).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
