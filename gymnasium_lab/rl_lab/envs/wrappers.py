from __future__ import annotations

from typing import Any

import numpy as np


def to_numpy_observation(observation: Any) -> np.ndarray:
    """Convert Gymnasium observations to a consistent float32 numpy array."""
    if isinstance(observation, np.ndarray):
        return observation.astype(np.float32, copy=False)
    if np.isscalar(observation):
        return np.asarray([observation], dtype=np.float32)
    return np.asarray(observation, dtype=np.float32)
