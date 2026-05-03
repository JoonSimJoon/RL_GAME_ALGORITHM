from __future__ import annotations

from typing import Iterable

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical, Independent, Normal

from rl_lab.utils.networks import build_mlp


def discounted_returns(rewards: list[float], gamma: float) -> torch.Tensor:
    running_return = 0.0
    returns: list[float] = []
    for reward in reversed(rewards):
        running_return = reward + gamma * running_return
        returns.append(running_return)
    returns.reverse()
    return torch.as_tensor(returns, dtype=torch.float32)


class DiscretePolicyNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_sizes: Iterable[int], action_dim: int) -> None:
        super().__init__()
        self.model = build_mlp(input_dim, hidden_sizes, action_dim)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.model(observation)

    def distribution(self, observation: torch.Tensor) -> Categorical:
        return Categorical(logits=self.forward(observation))


class ValueNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_sizes: Iterable[int]) -> None:
        super().__init__()
        self.model = build_mlp(input_dim, hidden_sizes, 1)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.model(observation).squeeze(-1)


class GaussianPolicyNetwork(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_sizes: Iterable[int],
        action_dim: int,
        action_low: np.ndarray,
        action_high: np.ndarray,
        init_log_std: float = -0.5,
    ) -> None:
        super().__init__()
        self.backbone = build_mlp(input_dim, hidden_sizes, hidden_sizes[-1] if hidden_sizes else input_dim)
        last_dim = hidden_sizes[-1] if hidden_sizes else input_dim
        self.mean_head = nn.Linear(last_dim, action_dim)
        self.log_std = nn.Parameter(torch.full((action_dim,), init_log_std))
        action_low_t = torch.as_tensor(action_low, dtype=torch.float32)
        action_high_t = torch.as_tensor(action_high, dtype=torch.float32)
        self.register_buffer("action_scale", (action_high_t - action_low_t) / 2.0)
        self.register_buffer("action_bias", (action_high_t + action_low_t) / 2.0)
        self.register_buffer("action_low", action_low_t)
        self.register_buffer("action_high", action_high_t)

    def forward(self, observation: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(observation)
        mean = torch.tanh(self.mean_head(features)) * self.action_scale + self.action_bias
        log_std = torch.clamp(self.log_std, min=-5.0, max=1.0)
        std = torch.exp(log_std).expand_as(mean)
        return mean, std

    def distribution(self, observation: torch.Tensor) -> Independent:
        mean, std = self.forward(observation)
        return Independent(Normal(mean, std), 1)
