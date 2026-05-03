from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class RLAlgorithm(ABC):
    """Common interface for every algorithm in the lab."""

    def __init__(self, env_spec, config: dict[str, Any], device: str = "cpu") -> None:
        self.env_spec = env_spec
        self.config = config
        self.device = device

    @abstractmethod
    def train(self, run_dir: Path) -> dict[str, Any]:
        """Train the algorithm and persist artifacts into ``run_dir``."""

    @abstractmethod
    def evaluate(
        self,
        checkpoint_path: Path,
        num_episodes: int = 10,
        seed: int | None = None,
        render: bool = False,
    ) -> dict[str, Any]:
        """Evaluate a saved checkpoint."""

    @abstractmethod
    def save(self, checkpoint_path: Path) -> None:
        """Save model state."""

    @abstractmethod
    def load(self, checkpoint_path: Path) -> None:
        """Load model state."""
