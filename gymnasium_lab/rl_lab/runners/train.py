from __future__ import annotations

import argparse
from pathlib import Path

from rl_lab.core.config import resolve_config, save_yaml
from rl_lab.core.experiment import create_run_dir
from rl_lab.core.registry import get_algorithm_class, get_env_spec


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train an RL algorithm in Gymnasium Lab.")
    parser.add_argument("--config", type=str, default=None, help="Path to a YAML config file.")
    parser.add_argument("--env", type=str, default=None, help="Environment key, e.g. frozenlake, cartpole, pendulum.")
    parser.add_argument("--algo", type=str, default=None, help="Algorithm key, e.g. q_learning, dqn, reinforce.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed override.")
    parser.add_argument("--train-episodes", type=int, default=None, help="Training episodes override.")
    parser.add_argument("--eval-episodes", type=int, default=None, help="Evaluation episodes override.")
    parser.add_argument("--device", type=str, default=None, help="Torch device override, e.g. cpu or cuda.")
    parser.add_argument("--runs-root", type=str, default=None, help="Custom root directory for saving runs.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    config, _ = resolve_config(
        config_path=args.config,
        env_key=args.env,
        algo_key=args.algo,
        seed=args.seed,
        train_episodes=args.train_episodes,
        eval_episodes=args.eval_episodes,
        device=args.device,
    )
    env_spec = get_env_spec(config["env"])
    algo_cls = get_algorithm_class(env_spec, config["algo"])
    device = config.get("device", args.device or "cpu")

    run_dir = create_run_dir(env_spec.key, config["algo"], int(config["seed"]), runs_root=args.runs_root)
    save_yaml(run_dir / "config.yaml", config)

    algorithm = algo_cls(env_spec, config, device=device)
    summary = algorithm.train(run_dir)

    print(f"Run saved to: {run_dir}")
    print(f"Final eval return: {summary['final_eval_return']:.3f}")
    if "final_success_rate" in summary:
        print(f"Final success rate: {summary['final_success_rate']:.3f}")
    print(f"Elapsed seconds: {summary['elapsed_seconds']:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
