from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch

from rl_lab.algorithms.policy_based.common import DiscretePolicyNetwork, GaussianPolicyNetwork, ValueNetwork
from rl_lab.core.interfaces import RLAlgorithm
from rl_lab.envs.factory import make_env
from rl_lab.envs.wrappers import to_numpy_observation
from rl_lab.utils.logging import save_json, write_csv
from rl_lab.utils.plotting import plot_training_curve
from rl_lab.utils.seed import set_global_seed


class ActorCriticAlgorithm(RLAlgorithm):
    def __init__(self, env_spec, config: dict[str, Any], device: str = "cpu") -> None:
        super().__init__(env_spec, config, device)
        self.actor = None
        self.critic: ValueNetwork | None = None
        self.optimizer: torch.optim.Optimizer | None = None
        self.is_continuous = env_spec.action_type == "continuous"

    def _ensure_initialized(self) -> None:
        if self.actor is not None and self.critic is not None:
            return

        env = make_env(self.env_spec, seed=self.config["seed"], env_kwargs=self.config.get("env_kwargs"))
        try:
            obs_dim = int(np.prod(env.observation_space.shape))
            hidden_sizes = self.config["network"]["hidden_sizes"]
            if self.is_continuous:
                action_dim = int(np.prod(env.action_space.shape))
                self.actor = GaussianPolicyNetwork(
                    obs_dim,
                    hidden_sizes,
                    action_dim,
                    action_low=env.action_space.low,
                    action_high=env.action_space.high,
                    init_log_std=float(self.config["algo_params"].get("init_log_std", -0.5)),
                ).to(self.device)
            else:
                action_dim = int(env.action_space.n)
                self.actor = DiscretePolicyNetwork(obs_dim, hidden_sizes, action_dim).to(self.device)
            self.critic = ValueNetwork(obs_dim, hidden_sizes).to(self.device)
        finally:
            env.close()

        actor_lr = float(self.config["algo_params"].get("actor_lr", self.config["algo_params"]["lr"]))
        critic_lr = float(self.config["algo_params"].get("critic_lr", self.config["algo_params"]["lr"]))
        self.optimizer = torch.optim.Adam(
            [
                {"params": self.actor.parameters(), "lr": actor_lr},
                {"params": self.critic.parameters(), "lr": critic_lr},
            ]
        )

    def save(self, checkpoint_path: Path) -> None:
        self._ensure_initialized()
        assert self.actor is not None and self.critic is not None
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "is_continuous": self.is_continuous,
            },
            checkpoint_path,
        )

    def load(self, checkpoint_path: Path) -> None:
        self._ensure_initialized()
        assert self.actor is not None and self.critic is not None
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])

    def _sample_action(self, observation: np.ndarray):
        assert self.actor is not None
        obs_t = torch.as_tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
        if self.is_continuous:
            dist = self.actor.distribution(obs_t)
            sample = dist.sample()
            action_t = torch.clamp(sample, min=self.actor.action_low, max=self.actor.action_high)
            return action_t.squeeze(0).cpu().numpy(), dist.log_prob(sample).squeeze(), dist.entropy().squeeze(), obs_t

        dist = self.actor.distribution(obs_t)
        action_t = dist.sample()
        return int(action_t.item()), dist.log_prob(action_t).squeeze(), dist.entropy().squeeze(), obs_t

    def _deterministic_action(self, observation: np.ndarray):
        assert self.actor is not None
        obs_t = torch.as_tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            if self.is_continuous:
                mean, _ = self.actor(obs_t)
                return mean.squeeze(0).cpu().numpy()
            logits = self.actor(obs_t)
            return int(torch.argmax(logits, dim=1).item())

    def evaluate(
        self,
        checkpoint_path: Path,
        num_episodes: int = 10,
        seed: int | None = None,
        render: bool = False,
    ) -> dict[str, Any]:
        self.load(checkpoint_path)
        eval_seed = self.config["seed"] if seed is None else seed
        env = make_env(self.env_spec, seed=eval_seed, render=render, env_kwargs=self.config.get("env_kwargs"))
        returns: list[float] = []
        lengths: list[int] = []

        try:
            for episode in range(num_episodes):
                observation, _ = env.reset(seed=eval_seed + episode)
                observation = to_numpy_observation(observation)
                terminated = False
                truncated = False
                episode_return = 0.0
                steps = 0

                while not (terminated or truncated):
                    action = self._deterministic_action(observation)
                    next_observation, reward, terminated, truncated, _ = env.step(action)
                    observation = to_numpy_observation(next_observation)
                    episode_return += float(reward)
                    steps += 1

                returns.append(episode_return)
                lengths.append(steps)
        finally:
            env.close()

        return {
            "avg_return": float(np.mean(returns)) if returns else 0.0,
            "avg_length": float(np.mean(lengths)) if lengths else 0.0,
            "num_episodes": num_episodes,
        }

    def train(self, run_dir: Path) -> dict[str, Any]:
        set_global_seed(self.config["seed"])
        self._ensure_initialized()
        assert self.actor is not None and self.critic is not None and self.optimizer is not None
        params = self.config["algo_params"]
        gamma = float(params["gamma"])
        entropy_coef = float(params.get("entropy_coef", 0.0))
        value_coef = float(params.get("value_coef", 0.5))

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
                episode_return = 0.0
                steps = 0
                losses: list[float] = []

                while not (terminated or truncated):
                    action, log_prob, entropy, obs_t = self._sample_action(observation)
                    value = self.critic(obs_t).squeeze()
                    next_observation, reward, terminated, truncated, _ = env.step(action)
                    next_observation = to_numpy_observation(next_observation)
                    next_obs_t = torch.as_tensor(next_observation, dtype=torch.float32, device=self.device).unsqueeze(0)

                    with torch.no_grad():
                        next_value = 0.0 if terminated else float(self.critic(next_obs_t).item())
                    td_target = float(reward) + gamma * next_value
                    target_t = torch.as_tensor(td_target, dtype=torch.float32, device=self.device)
                    advantage = target_t - value

                    actor_loss = -(log_prob * advantage.detach()) - entropy_coef * entropy
                    critic_loss = value_coef * advantage.pow(2)
                    loss = actor_loss + critic_loss

                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        list(self.actor.parameters()) + list(self.critic.parameters()),
                        max_norm=10.0,
                    )
                    self.optimizer.step()

                    observation = next_observation
                    episode_return += float(reward)
                    steps += 1
                    losses.append(float(loss.item()))

                row = {
                    "episode": episode,
                    "train_return": episode_return,
                    "train_length": steps,
                    "epsilon": None,
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
            title=f"{self.env_spec.key} - Actor-Critic",
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
