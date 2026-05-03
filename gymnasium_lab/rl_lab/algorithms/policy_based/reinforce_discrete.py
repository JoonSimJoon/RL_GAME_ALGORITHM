from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch

from rl_lab.algorithms.policy_based.common import DiscretePolicyNetwork, discounted_returns
from rl_lab.core.interfaces import RLAlgorithm
from rl_lab.envs.factory import make_env
from rl_lab.envs.wrappers import to_numpy_observation
from rl_lab.utils.logging import save_json, write_csv
from rl_lab.utils.plotting import plot_training_curve
from rl_lab.utils.seed import set_global_seed
from rl_lab.utils.video import save_frames


class ReinforceDiscreteAlgorithm(RLAlgorithm):
    def __init__(self, env_spec, config: dict[str, Any], device: str = "cpu") -> None:
        super().__init__(env_spec, config, device)
        self.policy: DiscretePolicyNetwork | None = None
        self.optimizer: torch.optim.Optimizer | None = None

    def _ensure_initialized(self) -> None:
        if self.policy is not None:
            return

        env = make_env(self.env_spec, seed=self.config["seed"], env_kwargs=self.config.get("env_kwargs"))
        try:
            obs_dim = int(np.prod(env.observation_space.shape))
            action_dim = int(env.action_space.n)
        finally:
            env.close()

        hidden_sizes = self.config["network"]["hidden_sizes"]
        self.policy = DiscretePolicyNetwork(obs_dim, hidden_sizes, action_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=float(self.config["algo_params"]["lr"]))

    def save(self, checkpoint_path: Path) -> None:
        self._ensure_initialized()
        assert self.policy is not None
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"policy": self.policy.state_dict()}, checkpoint_path)

    def load(self, checkpoint_path: Path) -> None:
        self._ensure_initialized()
        assert self.policy is not None
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.policy.load_state_dict(checkpoint["policy"])

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
        assert self.policy is not None
        eval_seed = self.config["seed"] if seed is None else seed
        env_kwargs = dict(self.config.get("env_kwargs", {}))
        if record_path is not None:
            env_kwargs["render_mode"] = "rgb_array"
        env = make_env(self.env_spec, seed=eval_seed, render=render, env_kwargs=env_kwargs)
        returns: list[float] = []
        lengths: list[int] = []
        recorded_frames = []

        try:
            for episode in range(num_episodes):
                observation, _ = env.reset(seed=eval_seed + episode)
                observation = to_numpy_observation(observation)
                if record_path is not None and episode == 0:
                    frame = env.render()
                    if frame is not None:
                        recorded_frames.append(frame)
                terminated = False
                truncated = False
                episode_return = 0.0
                steps = 0

                while not (terminated or truncated):
                    obs_t = torch.as_tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
                    with torch.no_grad():
                        logits = self.policy(obs_t)
                        action = int(torch.argmax(logits, dim=1).item())

                    next_observation, reward, terminated, truncated, _ = env.step(action)
                    if record_path is not None and episode == 0:
                        frame = env.render()
                        if frame is not None:
                            recorded_frames.append(frame)
                    observation = to_numpy_observation(next_observation)
                    episode_return += float(reward)
                    steps += 1

                returns.append(episode_return)
                lengths.append(steps)
        finally:
            env.close()

        if record_path is not None:
            save_frames(record_path, recorded_frames, fps=fps)

        return {
            "avg_return": float(np.mean(returns)) if returns else 0.0,
            "avg_length": float(np.mean(lengths)) if lengths else 0.0,
            "num_episodes": num_episodes,
        }

    def train(self, run_dir: Path) -> dict[str, Any]:
        set_global_seed(self.config["seed"])
        self._ensure_initialized()
        assert self.policy is not None and self.optimizer is not None
        params = self.config["algo_params"]
        env = make_env(self.env_spec, seed=self.config["seed"], env_kwargs=self.config.get("env_kwargs"))

        rows: list[dict[str, Any]] = []
        eval_points: list[tuple[int, float]] = []
        best_eval_return = float("-inf")
        checkpoint_path = run_dir / "checkpoint.pt"
        start_time = perf_counter()

        try:
            for episode in range(1, int(self.config["train_episodes"]) + 1):
                observation, _ = env.reset(seed=self.config["seed"] + episode)
                observation = to_numpy_observation(observation)
                terminated = False
                truncated = False
                rewards: list[float] = []
                log_probs: list[torch.Tensor] = []
                entropies: list[torch.Tensor] = []
                episode_return = 0.0
                steps = 0

                while not (terminated or truncated):
                    obs_t = torch.as_tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
                    dist = self.policy.distribution(obs_t)
                    action_t = dist.sample()
                    log_prob = dist.log_prob(action_t)
                    entropy = dist.entropy()
                    action = int(action_t.item())

                    next_observation, reward, terminated, truncated, _ = env.step(action)
                    observation = to_numpy_observation(next_observation)
                    rewards.append(float(reward))
                    log_probs.append(log_prob.squeeze())
                    entropies.append(entropy.squeeze())
                    episode_return += float(reward)
                    steps += 1

                returns = discounted_returns(rewards, float(params["gamma"])).to(self.device)
                if bool(params.get("normalize_returns", True)) and len(returns) > 1:
                    returns = (returns - returns.mean()) / (returns.std(unbiased=False) + 1e-8)

                policy_loss = -(torch.stack(log_probs) * returns).sum()
                entropy_bonus = torch.stack(entropies).sum()
                loss = policy_loss - float(params.get("entropy_coef", 0.0)) * entropy_bonus

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=10.0)
                self.optimizer.step()

                row = {
                    "episode": episode,
                    "train_return": episode_return,
                    "train_length": steps,
                    "epsilon": None,
                    "loss": float(loss.item()),
                    "eval_return": None,
                    "eval_length": None,
                    "eval_success_rate": None,
                }

                if episode % int(self.config["eval_interval"]) == 0 or episode == int(self.config["train_episodes"]):
                    self.save(checkpoint_path)
                    eval_result = self.evaluate(
                        checkpoint_path=checkpoint_path,
                        num_episodes=int(self.config["eval_episodes"]),
                        seed=self.config["seed"] + 20_000 + episode,
                    )
                    row["eval_return"] = eval_result["avg_return"]
                    row["eval_length"] = eval_result["avg_length"]
                    eval_points.append((episode, float(eval_result["avg_return"])))
                    best_eval_return = max(best_eval_return, float(eval_result["avg_return"]))

                rows.append(row)
        finally:
            env.close()

        self.save(checkpoint_path)
        write_csv(run_dir / "train_metrics.csv", rows)
        plot_training_curve(
            episodes=[row["episode"] for row in rows],
            returns=[float(row["train_return"]) for row in rows],
            output_path=run_dir / "learning_curve.png",
            title=f"{self.env_spec.key} - REINFORCE",
            eval_points=eval_points,
            smoothing_window=int(self.config.get("plot_window", 20)),
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
            "best_eval_return": best_eval_return,
            "elapsed_seconds": perf_counter() - start_time,
        }
        save_json(run_dir / "eval_metrics.json", final_eval)
        save_json(run_dir / "summary.json", summary)
        return summary
