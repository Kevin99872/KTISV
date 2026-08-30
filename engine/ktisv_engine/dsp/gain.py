"""增益平滑與 dB 換算。

音量推桿如果直接套用會產生「拉鍊雜訊」(zipper noise),所以每個增益都
用一段線性斜坡漸進到目標值。
"""

from __future__ import annotations

import math

import numpy as np

MIN_DB = -60.0
"""推桿拉到底時視為靜音的門檻。"""


def db_to_lin(db: float) -> float:
    if db <= MIN_DB:
        return 0.0
    return float(10.0 ** (db / 20.0))


def lin_to_db(lin: float) -> float:
    if lin <= 1e-9:
        return -120.0
    return float(20.0 * math.log10(lin))


class SmoothGain:
    """以固定斜率漸進到目標值的增益。"""

    def __init__(self, value: float = 1.0, ramp_ms: float = 20.0,
                 samplerate: int = 48000) -> None:
        self._current = float(value)
        self._target = float(value)
        self._ramp_frames = max(1.0, ramp_ms * 1e-3 * samplerate)

    @property
    def target(self) -> float:
        return self._target

    @target.setter
    def target(self, value: float) -> None:
        self._target = float(value)

    @property
    def current(self) -> float:
        return self._current

    def snap(self, value: float) -> None:
        self._current = self._target = float(value)

    def is_silent(self) -> bool:
        return self._current == 0.0 and self._target == 0.0

    def envelope(self, frames: int) -> np.ndarray:
        """回傳 (frames, 1) 的增益包絡,並推進內部狀態。"""
        start = self._current
        if start == self._target:
            if start == 1.0:
                return _ONES_CACHE(frames)
            return np.full((frames, 1), start, dtype=np.float32)

        # 每個 block 最多移動 frames / ramp_frames 的比例(以線性增益計)
        max_step = frames / self._ramp_frames
        delta = self._target - start
        if abs(delta) <= max_step:
            end = self._target
        else:
            end = start + math.copysign(max_step, delta)
        self._current = end
        env = np.linspace(start, end, frames, endpoint=False, dtype=np.float32)
        return env.reshape(frames, 1)


_ones_cache: dict[int, np.ndarray] = {}


def _ONES_CACHE(frames: int) -> np.ndarray:
    arr = _ones_cache.get(frames)
    if arr is None:
        arr = np.ones((frames, 1), dtype=np.float32)
        _ones_cache[frames] = arr
    return arr
