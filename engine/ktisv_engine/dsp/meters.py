"""電平表(peak / RMS)與彈道特性。

峰值瞬間跟上、緩慢釋放;RMS 用單極平滑。全部以 dBFS 回報。
"""

from __future__ import annotations

import math

import numpy as np

FLOOR_DB = -72.0


class Meter:
    """單一量測點的電平表。"""

    def __init__(self, samplerate: int = 48000,
                 peak_release_db_per_s: float = 26.0,
                 rms_time_ms: float = 300.0,
                 hold_ms: float = 1200.0) -> None:
        self.samplerate = samplerate
        self._peak_release = peak_release_db_per_s
        self._rms_coef_time = rms_time_ms * 1e-3
        self._hold_s = hold_ms * 1e-3

        self.peak_db = FLOOR_DB
        self.rms_db = FLOOR_DB
        self.hold_db = FLOOR_DB
        self.clipped = False
        self._hold_timer = 0.0
        self._rms_sq = 0.0

    def reset(self) -> None:
        self.peak_db = self.rms_db = self.hold_db = FLOOR_DB
        self.clipped = False
        self._hold_timer = 0.0
        self._rms_sq = 0.0

    def process(self, block: np.ndarray) -> None:
        frames = len(block)
        if frames == 0:
            return
        dt = frames / self.samplerate

        peak_lin = float(np.max(np.abs(block))) if block.size else 0.0
        if peak_lin >= 0.999:
            self.clipped = True
        peak_db = _to_db(peak_lin)

        if peak_db >= self.peak_db:
            self.peak_db = peak_db
        else:
            self.peak_db = max(FLOOR_DB, self.peak_db - self._peak_release * dt)

        mean_sq = float(np.mean(np.square(block, dtype=np.float64))) if block.size else 0.0
        alpha = 1.0 - math.exp(-dt / max(self._rms_coef_time, 1e-4))
        self._rms_sq += alpha * (mean_sq - self._rms_sq)
        self.rms_db = _to_db(math.sqrt(max(self._rms_sq, 0.0)))

        if peak_db >= self.hold_db:
            self.hold_db = peak_db
            self._hold_timer = self._hold_s
        else:
            self._hold_timer -= dt
            if self._hold_timer <= 0.0:
                self.hold_db = max(FLOOR_DB, self.hold_db - self._peak_release * dt)

    def snapshot(self) -> dict:
        clipped = self.clipped
        self.clipped = False
        return {
            "peak": round(self.peak_db, 2),
            "rms": round(self.rms_db, 2),
            "hold": round(self.hold_db, 2),
            "clip": clipped,
        }


def _to_db(lin: float) -> float:
    if lin <= 1e-6:
        return FLOOR_DB
    return max(FLOOR_DB, 20.0 * math.log10(lin))


class MeterBank:
    """一組具名的電平表。"""

    def __init__(self, names, samplerate: int = 48000) -> None:
        self._meters = {name: Meter(samplerate) for name in names}

    def __getitem__(self, name: str) -> Meter:
        return self._meters[name]

    def reset(self) -> None:
        for m in self._meters.values():
            m.reset()

    def snapshot(self) -> dict:
        return {name: m.snapshot() for name, m in self._meters.items()}
