from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .metrics import moving_average


def plot_training_curve(
    episodes: list[int],
    returns: list[float],
    output_path: Path,
    title: str,
    eval_points: list[tuple[int, float]] | None = None,
    smoothing_window: int = 20,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 5))
    plt.plot(episodes, returns, alpha=0.35, label="Episode Return")
    plt.plot(episodes, moving_average(returns, smoothing_window), linewidth=2, label=f"Moving Avg ({smoothing_window})")

    if eval_points:
        eval_episodes = [point[0] for point in eval_points]
        eval_returns = [point[1] for point in eval_points]
        plt.plot(eval_episodes, eval_returns, marker="o", linewidth=2, label="Evaluation Return")

    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_compare_curves(curves: list[dict], output_path: Path, title: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 5))
    for curve in curves:
        plt.plot(curve["episodes"], curve["values"], linewidth=2, label=curve["label"])

    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
