from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class EnvSpec:
    key: str
    gym_id: str
    observation_type: str
    action_type: str
    compatible_algorithms: list[str]
    reward_metric: str = "episode_return"
    solved_threshold: float | None = None
    default_env_kwargs: dict[str, Any] = field(default_factory=dict)


ENV_SPECS: dict[str, EnvSpec] = {
    "frozenlake": EnvSpec(
        key="frozenlake",
        gym_id="FrozenLake-v1",
        observation_type="discrete",
        action_type="discrete",
        compatible_algorithms=["q_learning", "sarsa"],
        solved_threshold=0.78,
        default_env_kwargs={"is_slippery": False},
    ),
    "cartpole": EnvSpec(
        key="cartpole",
        gym_id="CartPole-v1",
        observation_type="continuous_vector",
        action_type="discrete",
        compatible_algorithms=["dqn", "reinforce", "a2c"],
        solved_threshold=475.0,
    ),
    "pendulum": EnvSpec(
        key="pendulum",
        gym_id="Pendulum-v1",
        observation_type="continuous_vector",
        action_type="continuous",
        compatible_algorithms=["reinforce", "actor_critic"],
        solved_threshold=-250.0,
    ),
}
