from __future__ import annotations

from collections import deque


def moving_average(values: list[float], window: int) -> list[float]:
    if not values:
        return []
    history = deque(maxlen=max(1, window))
    result: list[float] = []
    for value in values:
        history.append(value)
        result.append(sum(history) / len(history))
    return result


def safe_float(value, default: float | None = None) -> float | None:
    if value in ("", None):
        return default
    return float(value)
