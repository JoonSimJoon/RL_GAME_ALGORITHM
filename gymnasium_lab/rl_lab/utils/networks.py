from __future__ import annotations

from typing import Iterable

import torch
from torch import nn


def build_mlp(input_dim: int, hidden_sizes: Iterable[int], output_dim: int, output_activation: nn.Module | None = None) -> nn.Sequential:
    layers: list[nn.Module] = []
    last_dim = input_dim
    for hidden_size in hidden_sizes:
        layers.append(nn.Linear(last_dim, hidden_size))
        layers.append(nn.ReLU())
        last_dim = hidden_size
    layers.append(nn.Linear(last_dim, output_dim))
    if output_activation is not None:
        layers.append(output_activation)
    return nn.Sequential(*layers)
