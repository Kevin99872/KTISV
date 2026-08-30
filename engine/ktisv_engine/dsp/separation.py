"""即時中央聲道分離(mid/side 卡拉 OK 演算法)。

原理:多數混音把主唱擺在正中央,也就是完全存在於 mid = (L+R)/2、
在 side = (L-R)/2 幾乎消失。直接輸出 side 就能去掉人聲,但同時會抽掉
所有置中的樂器 —— 尤其是低頻的貝斯與大鼓。

所以這裡把 mid 拆成三段:低於 ``low_cut`` 與高於 ``high_cut`` 的部分視為
「非人聲的置中內容」保留回去,中間那段才是要被移除(或被單獨取出)的
人聲頻段。``strength`` 用來在乾訊號與處理後訊號之間做混合。
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfilt

MODE_OFF = "off"
MODE_REMOVE_VOCALS = "remove_vocals"
MODE_ISOLATE_VOCALS = "isolate_vocals"
MODE_SILENCE = "silence"

MODES = (MODE_OFF, MODE_REMOVE_VOCALS, MODE_ISOLATE_VOCALS, MODE_SILENCE)


class CenterSeparator:
    """零延遲的人聲移除 / 人聲取出。"""

    def __init__(self, samplerate: int = 48000,
                 low_cut: float = 180.0, high_cut: float = 9000.0) -> None:
        self.samplerate = samplerate
        self.mode = MODE_OFF
        self.strength = 1.0
        self._low_cut = low_cut
        self._high_cut = high_cut
        self._build()

    # ── 參數 ────────────────────────────────────────────────────────────
    @property
    def low_cut(self) -> float:
        return self._low_cut

    @low_cut.setter
    def low_cut(self, value: float) -> None:
        value = max(20.0, min(600.0, float(value)))
        if abs(value - self._low_cut) > 1e-3:
            self._low_cut = value
            self._build()

    @property
    def high_cut(self) -> float:
        return self._high_cut

    @high_cut.setter
    def high_cut(self, value: float) -> None:
        value = max(2000.0, min(self.samplerate * 0.45, float(value)))
        if abs(value - self._high_cut) > 1e-3:
            self._high_cut = value
            self._build()

    def _build(self) -> None:
        nyq = self.samplerate * 0.5
        self._sos_low = butter(2, self._low_cut / nyq, btype="low", output="sos")
        self._sos_high = butter(2, self._high_cut / nyq, btype="high", output="sos")
        self._zi_low = np.zeros((self._sos_low.shape[0], 2), dtype=np.float64)
        self._zi_high = np.zeros((self._sos_high.shape[0], 2), dtype=np.float64)

    def reset(self) -> None:
        self._zi_low[:] = 0.0
        self._zi_high[:] = 0.0

    # ── 處理 ────────────────────────────────────────────────────────────
    def process(self, x: np.ndarray) -> np.ndarray:
        """x: (frames, 2) float32 立體聲 → 處理後的 (frames, 2)。"""
        if self.mode == MODE_OFF or self.strength <= 0.0:
            return x
        if self.mode == MODE_SILENCE:
            return np.zeros_like(x)
        if x.shape[1] < 2:
            # 單聲道無法做 mid/side;取出人聲時原樣通過,移除人聲時只能靜音
            return x if self.mode == MODE_ISOLATE_VOCALS else np.zeros_like(x)

        left = x[:, 0].astype(np.float64, copy=False)
        right = x[:, 1].astype(np.float64, copy=False)
        mid = 0.5 * (left + right)
        side = 0.5 * (left - right)

        mid_low, self._zi_low = sosfilt(self._sos_low, mid, zi=self._zi_low)
        mid_high, self._zi_high = sosfilt(self._sos_high, mid, zi=self._zi_high)
        mid_band = mid - mid_low - mid_high

        if self.mode == MODE_REMOVE_VOCALS:
            keep = mid_low + mid_high
            wet_l = side + keep
            wet_r = -side + keep
        else:  # MODE_ISOLATE_VOCALS
            wet_l = mid_band
            wet_r = mid_band

        s = float(self.strength)
        out = np.empty_like(x)
        out[:, 0] = (1.0 - s) * left + s * wet_l
        out[:, 1] = (1.0 - s) * right + s * wet_r
        return out


def mode_from_flags(remove_vocals: bool, remove_instrumental: bool) -> str:
    """把 UI 上的兩個勾選框對應到分離模式。"""
    if remove_vocals and remove_instrumental:
        return MODE_SILENCE
    if remove_vocals:
        return MODE_REMOVE_VOCALS
    if remove_instrumental:
        return MODE_ISOLATE_VOCALS
    return MODE_OFF
