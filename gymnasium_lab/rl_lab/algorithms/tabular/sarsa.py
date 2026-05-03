from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from rl_lab.core.interfaces import RLAlgorithm
from rl_lab.envs.factory import make_env
from rl_lab.utils.logging import save_json, write_csv
from rl_lab.utils.plotting import plot_training_curve
from rl_lab.utils.seed import set_global_seed
from rl_lab.utils.video import save_frames


class SARSAAlgorithm(RLAlgorithm):
    def __init__(self, env_spec, config: dict[str, Any], device: str = "cpu") -> None:
        super().__init__(env_spec, config, device)
        self.q_table: np.ndarray | None = None

    def save(self, checkpoint_path: Path) -> None:
        if self.q_table is None:
            raise ValueError("Q-table is not initialized.")
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(checkpoint_path, q_table=self.q_table)

    def load(self, checkpoint_path: Path) -> None:
        data = np.load(checkpoint_path)
        self.q_table = data["q_table"]

    def _build_q_table(self) -> np.ndarray:
        env = make_env(self.env_spec, seed=self.config["seed"], env_kwargs=self.config.get("env_kwargs"))
        try:
            num_states = env.observation_space.n
            num_actions = env.action_space.n
        finally:
            env.close()
        return np.zeros((num_states, num_actions), dtype=np.float32)

    def _greedy_action(self, state: int) -> int:
        assert self.q_table is not None
        return int(np.argmax(self.q_table[state]))

    def _epsilon_greedy_action(self, state: int, epsilon: float, action_space_n: int) -> int:
        if np.random.rand() < epsilon:
            return int(np.random.randint(action_space_n))
        return self._greedy_action(state)

    def evaluate(
        self,
        checkpoint_path: Path,
        num_episodes: int = 10,
        seed: int | None = None,
        render: bool = False,
        record_path: Path | None = None,
        fps: int = 30,
    ) -> dict[str, Any]:
        self.load(checkpoint_path)
        eval_seed = self.config["seed"] if seed is None else seed
        env_kwargs = dict(self.config.get("env_kwargs", {}))
        if record_path is not None:
            env_kwargs["render_mode"] = "rgb_array"
        env = make_env(self.env_spec, seed=eval_seed, render=render, env_kwargs=env_kwargs)
        returns: list[float] = []
        lengths: list[int] = []
        successes = 0
        recorded_frames = []

        try:
            for episode in range(num_episodes):
                state, _ = env.reset(seed=eval_seed + episode)
                if record_path is not None and episode == 0:
                    frame = env.render()
                    if frame is not None:
                        recorded_frames.append(frame)
                done = False
                truncated = False
                episode_return = 0.0
                steps = 0

                while not (done or truncated):
                    action = self._greedy_action(int(state))
                    next_state, reward, done, truncated, _ = env.step(action)
                    if record_path is not None and episode == 0:
                        frame = env.render()
                        if frame is not None:
                            recorded_frames.append(frame)
                    state = next_state
                    episode_return += float(reward)
                    steps += 1

                returns.append(episode_return)
                lengths.append(steps)
                if episode_return > 0.0:
                    successes += 1
        finally:
            env.close()

        if record_path is not None:
            save_frames(record_path, recorded_frames, fps=fps)

        return {
            "avg_return": float(np.mean(returns)) if returns else 0.0,
            "avg_length": float(np.mean(lengths)) if lengths else 0.0,
            "success_rate": float(successes / max(1, num_episodes)),
            "num_episodes": num_episodes,
        }

    def train(self, run_dir: Path) -> dict[str, Any]:
        set_global_seed(self.config["seed"])
        params = self.config["algo_params"]
        self.q_table = self._build_q_table()

        env = make_env(self.env_spec, seed=self.config["seed"], env_kwargs=self.config.get("env_kwargs"))
        start_time = perf_counter()
        rows: list[dict[str, Any]] = []
        eval_points: list[tuple[int, float]] = []
        best_eval_return = float("-inf")
        best_success_rate = 0.0
        checkpoint_path = run_dir / "checkpoint.npz"

        try:
            epsilon = float(params["epsilon_start"])
            action_space_n = int(env.action_space.n)
            for episode in range(1, int(self.config["train_episodes"]) + 1):
                state, _ = env.reset(seed=self.config["seed"] + episode)
                action = self._epsilon_greedy_action(int(state), epsilon, action_space_n)
                done = False
                truncated = False
                episode_return = 0.0
                steps = 0

                while not (done or truncated):
                    next_state, reward, done, truncated, _ = env.step(action)
                    next_action = 0 if (done or truncated) else self._epsilon_greedy_action(int(next_state), epsilon, action_space_n)
                    next_q = 0.0 if (done or truncated) else float(self.q_table[int(next_state), int(next_action)])
                    td_target = float(reward) + float(params["gamma"]) * next_q
                    td_error = td_target - float(self.q_table[int(state), int(action)])
                    self.q_table[int(state), int(action)] += float(params["alpha"]) * td_error

                    state = next_state
                    action = next_action
                    episode_return += float(reward)
                    steps += 1

                epsilon = max(float(params["epsilon_end"]), epsilon * float(params["epsilon_decay"]))
                row = {
                    "episode": episode,
                    "train_return": episode_return,
                    "train_length": steps,
                    "epsilon": epsilon,
                    "loss": None,
                    "eval_return": None,
                    "eval_length": None,
                    "eval_success_rate": None,
                }

                if episode % int(self.config["eval_interval"]) == 0 or episode == int(self.config["train_episodes"]):
                    self.save(checkpoint_path)
                    eval_result = self.evaluate(
                        checkpoint_path=checkpoint_path,
                        num_episodes=int(self.config["eval_episodes"]),
                        seed=self.config["seed"] + 10_000 + episode,
                    )
                    row["eval_return"] = eval_result["avg_return"]
                    row["eval_length"] = eval_result["avg_length"]
                    row["eval_success_rate"] = eval_result["success_rate"]
                    eval_points.append((episode, float(eval_result["avg_return"])))
                    best_eval_return = max(best_eval_return, float(eval_result["avg_return"]))
                    best_success_rate = max(best_success_rate, float(eval_result["success_rate"]))

                rows.append(row)
        finally:
            env.close()

        self.save(checkpoint_path)
        write_csv(run_dir / "train_metrics.csv", rows)
        plot_training_curve(
            episodes=[row["episode"] for row in rows],
            returns=[float(row["train_return"]) for row in rows],
            output_path=run_dir / "learning_curve.png",
            title=f"{self.env_spec.key} - SARSA",
            eval_points=eval_points,
            smoothing_window=int(self.config.get("plot_window", 50)),
        )

        final_eval = self.evaluate(
            checkpoint_path=checkpoint_path,
            num_episodes=int(self.config["eval_episodes"]),
            seed=self.config["seed"] + 99_999,
        )
        summary = {
            "env": self.env_spec.key,
            "algo": self.config["algo"],
            "seed": self.config["seed"],
            "train_episodes": int(self.config["train_episodes"]),
            "final_eval_return": final_eval["avg_return"],
            "final_eval_length": final_eval["avg_length"],
            "final_success_rate": final_eval["success_rate"],
            "best_eval_return": best_eval_return,
            "best_success_rate": best_success_rate,
            "elapsed_seconds": perf_counter() - start_time,
        }
        save_json(run_dir / "eval_metrics.json", final_eval)
        save_json(run_dir / "summary.json", summary)
        return summary
