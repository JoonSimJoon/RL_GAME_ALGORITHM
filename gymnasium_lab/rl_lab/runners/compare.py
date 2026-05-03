from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from rl_lab.core.config import PROJECT_ROOT
from rl_lab.core.experiment import get_latest_run
from rl_lab.utils.logging import read_csv, save_json
from rl_lab.utils.metrics import moving_average, safe_float
from rl_lab.utils.plotting import plot_compare_curves


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare the latest runs of multiple algorithms for one environment.")
    parser.add_argument("--env", type=str, required=True, help="Environment key.")
    parser.add_argument("--algos", nargs="+", required=True, help="Algorithms to compare.")
    parser.add_argument("--runs-root", type=str, default=None, help="Custom root directory for runs.")
    parser.add_argument("--window", type=int, default=20, help="Moving average window for train returns.")
    return parser


def _extract_curve(metrics_path: Path, label: str, window: int) -> dict:
    rows = read_csv(metrics_path)
    eval_episodes: list[int] = []
    eval_returns: list[float] = []

    for row in rows:
        eval_return = safe_float(row.get("eval_return"))
        if eval_return is not None:
            eval_episodes.append(int(row["episode"]))
            eval_returns.append(eval_return)

    if eval_returns:
        return {"label": label, "episodes": eval_episodes, "values": eval_returns}

    train_episodes = [int(row["episode"]) for row in rows]
    train_returns = [float(row["train_return"]) for row in rows]
    return {"label": label, "episodes": train_episodes, "values": moving_average(train_returns, window)}


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    runs_root = Path(args.runs_root).resolve() if args.runs_root else PROJECT_ROOT / "runs"
    compare_root = runs_root / args.env / "comparisons"
    compare_dir = compare_root / datetime.now().strftime("%Y%m%d_%H%M%S")
    compare_dir.mkdir(parents=True, exist_ok=False)

    curves = []
    summary_entries = []

    for algo_key in args.algos:
        run_dir = get_latest_run(args.env, algo_key, runs_root=str(runs_root))
        curve = _extract_curve(run_dir / "train_metrics.csv", algo_key, window=args.window)
        curves.append(curve)

        summary_path = run_dir / "summary.json"
        if summary_path.exists():
            import json

            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        else:
            summary = {}
        summary_entries.append(
            {
                "algo": algo_key,
                "run_dir": str(run_dir),
                "final_eval_return": summary.get("final_eval_return"),
                "best_eval_return": summary.get("best_eval_return"),
                "elapsed_seconds": summary.get("elapsed_seconds"),
            }
        )

    plot_compare_curves(
        curves=curves,
        output_path=compare_dir / "comparison.png",
        title=f"{args.env} algorithm comparison",
    )
    save_json(
        compare_dir / "comparison_summary.json",
        {
            "env": args.env,
            "algorithms": args.algos,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "runs": summary_entries,
        },
    )

    print(f"Comparison saved to: {compare_dir}")
    for item in summary_entries:
        print(f"{item['algo']}: final_eval_return={item['final_eval_return']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
