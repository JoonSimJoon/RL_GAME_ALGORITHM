from __future__ import annotations

from collections import deque
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch
from torch import nn

from rl_lab.core.interfaces import RLAlgorithm
from rl_lab.envs.factory import make_env
from rl_lab.envs.wrappers import to_numpy_observation
from rl_lab.utils.logging import save_json, write_csv
from rl_lab.utils.networks import build_mlp
from rl_lab.utils.plotting import plot_training_curve
from rl_lab.utils.seed import set_global_seed
from rl_lab.utils.video import save_frames


class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.buffer: deque[tuple[np.ndarray, int, float, np.ndarray, float]] = deque(maxlen=capacity)

    def append(self, transition: tuple[np.ndarray, int, float, np.ndarray, float]) -> None:
        self.buffer.append(transition)

    def sample(self, batch_size: int):
        indices = np.random.choice(len(self.buffer), size=batch_size, replace=False)
        batch = [self.buffer[idx] for idx in indices]
        obs, actions, rewards, next_obs, dones = zip(*batch)
        return (
            np.stack(obs).astype(np.float32),
            np.asarray(actions, dtype=np.int64),
            np.asarray(rewards, dtype=np.float32),
            np.stack(next_obs).astype(np.float32),
            np.asarray(dones, dtype=np.float32),
        )

    def __len__(self) -> int:
        return len(self.buffer)


class DQNAlgorithm(RLAlgorithm):
    def __init__(self, env_spec, config: dict[str, Any], device: str = "cpu") -> None:
        super().__init__(env_spec, config, device)
        self.q_network: nn.Module | None = None
        self.target_network: nn.Module | None = None
        self.optimizer: torch.optim.Optimizer | None = None
        self.replay_buffer: ReplayBuffer | None = None
        self.obs_dim: int | None = None
        self.action_dim: int | None = None

    def _ensure_initialized(self) -> None:
        if self.q_network is not None:
            return

        env = make_env(self.env_spec, seed=self.config["seed"], env_kwargs=self.config.get("env_kwargs"))
        try:
            self.obs_dim = int(np.prod(env.observation_space.shape))
            self.action_dim = int(env.action_space.n)
        finally:
            env.close()

        hidden_sizes = self.config["network"]["hidden_sizes"]
        self.q_network = build_mlp(self.obs_dim, hidden_sizes, self.action_dim).to(self.device)
        self.target_network = build_mlp(self.obs_dim, hidden_sizes, self.action_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=float(self.config["algo_params"]["lr"]))
        self.replay_buffer = ReplayBuffer(int(self.config["algo_params"]["buffer_size"]))

    def save(self, checkpoint_path: Path) -> None:
        self._ensure_initialized()
        assert self.q_network is not None and self.target_network is not None and self.obs_dim is not None and self.action_dim is not None
        checkpoint = {
            "q_network": self.q_network.state_dict(),
            "target_network": self.target_network.state_dict(),
            "obs_dim": self.obs_dim,
            "action_dim": self.action_dim,
        }
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, checkpoint_path)

    def load(self, checkpoint_path: Path) -> None:
        self._ensure_initialized()
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        assert self.q_network is not None and self.target_network is not None
        self.q_network.load_state_dict(checkpoint["q_network"])
        self.target_network.load_state_dict(checkpoint["target_network"])

    def _select_action(self, observation: np.ndarray, epsilon: float) -> int:
        assert self.q_network is not None and self.action_dim is not None
        if np.random.rand() < epsilon:
            return int(np.random.randint(self.action_dim))

        obs_tensor = torch.as_tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_network(obs_tensor)
        return int(torch.argmax(q_values, dim=1).item())

    def _optimize(self) -> float | None:
        assert self.q_network is not None and self.target_network is not None and self.optimizer is not None and self.replay_buffer is not None
        params = self.config["algo_params"]
        batch_size = int(params["batch_size"])
        if len(self.replay_buffer) < max(batch_size, int(params["min_buffer_size"])):
            return None

        observations, actions, rewards, next_observations, dones = self.replay_buffer.sample(batch_size)
        observations_t = torch.as_tensor(observations, dtype=torch.float32, device=self.device)
        actions_t = torch.as_tensor(actions, dtype=torch.int64, device=self.device)
        rewards_t = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
        next_observations_t = torch.as_tensor(next_observations, dtype=torch.float32, device=self.device)
        dones_t = torch.as_tensor(dones, dtype=torch.float32, device=self.device)

        q_values = self.q_network(observations_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q_values = self.target_network(next_observations_t).max(dim=1).values
            targets = rewards_t + float(params["gamma"]) * next_q_values * (1.0 - dones_t)

        loss = torch.nn.functional.smooth_l1_loss(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=10.0)
        self.optimizer.step()
        return float(loss.item())

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
        assert self.q_network is not None
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
                    action = self._select_action(observation, epsilon=0.0)
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
        assert self.q_network is not None and self.target_network is not None and self.replay_buffer is not None
        params = self.config["algo_params"]

        env = make_env(self.env_spec, seed=self.config["seed"], env_kwargs=self.config.get("env_kwargs"))
        rows: list[dict[str, Any]] = []
        eval_points: list[tuple[int, float]] = []
        best_eval_return = float("-inf")
        checkpoint_path = run_dir / "checkpoint.pt"
        epsilon = float(params["epsilon_start"])
        global_step = 0
        start_time = perf_counter()

        try:
            for episode in range(1, int(self.config["train_episodes"]) + 1):
                observation, _ = env.reset(seed=self.config["seed"] + episode)
                observation = to_numpy_observation(observation)
                terminated = False
                truncated = False
                episode_return = 0.0
                steps = 0
                losses: list[float] = []

                while not (terminated or truncated):
                    action = self._select_action(observation, epsilon=epsilon)
                    next_observation, reward, terminated, truncated, _ = env.step(action)
                    next_observation = to_numpy_observation(next_observation)
                    done_mask = 1.0 if terminated else 0.0
                    self.replay_buffer.append((observation, action, float(reward), next_observation, done_mask))

                    observation = next_observation
                    episode_return += float(reward)
                    steps += 1
                    global_step += 1

                    loss = self._optimize()
                    if loss is not None:
                        losses.append(loss)

                    if global_step % int(params["target_update_interval"]) == 0:
                        self.target_network.load_state_dict(self.q_network.state_dict())

                epsilon = max(float(params["epsilon_end"]), epsilon * float(params["epsilon_decay"]))
                row = {
                    "episode": episode,
                    "train_return": episode_return,
                    "train_length": steps,
                    "epsilon": epsilon,
                    "loss": float(np.mean(losses)) if losses else None,
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
            title=f"{self.env_spec.key} - DQN",
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
