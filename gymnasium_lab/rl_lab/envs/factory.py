from __future__ import annotations

from typing import Any

import gymnasium as gym


def make_env(env_spec, seed: int | None = None, render: bool = False, env_kwargs: dict[str, Any] | None = None):
    kwargs = dict(env_spec.default_env_kwargs)
    if env_kwargs:
        kwargs.update(env_kwargs)

    render_mode = "human" if render else None
    if render_mode is not None:
        kwargs["render_mode"] = render_mode

    env = gym.make(env_spec.gym_id, **kwargs)
    env.reset(seed=seed)
    if hasattr(env.action_space, "seed"):
        env.action_space.seed(seed)
    if hasattr(env.observation_space, "seed"):
        env.observation_space.seed(seed)
    return env
