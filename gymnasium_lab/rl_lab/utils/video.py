from __future__ import annotations

from pathlib import Path

import imageio.v2 as imageio
import numpy as np


def save_frames(path: Path, frames: list[np.ndarray], fps: int = 30) -> None:
    if not frames:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(path, frames, fps=fps)
