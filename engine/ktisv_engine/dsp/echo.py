"""麥克風回音(卡拉 OK 的「殘響 / ECHO」旋鈕)。

這是**效果器**,不是對齊工具 —— 跟 :mod:`ktisv_engine.dsp.delay` 的用途相反:
那條延遲線是把整路訊號往後推,這裡是把訊號的複本一次次疊回去,製造尾音。

結構是最經典的回授梳狀濾波器,回授路徑上串一顆一階低通:

    y[n] = x[n] + fb · lp(y[n-D])
    out  = x[n] + mix · lp(y[n-D])

低通(``damping``)讓每一次重複都比前一次暗一點。少了它,高頻會一路
反覆疊加,聽起來像金屬罐;有了它才像真實空間的殘響。

為什麼可以整個 block 一起算
--------------------------
回授看起來是逐取樣的遞迴,但只要延遲量 D 不小於一個 block,``y[n-D]``
在這個 block 開始時就已經全部寫好了,不會用到本 block 才要算出來的值。
所以 D 被夾在「至少一個 block」以上,整段用 numpy 一次算完。
"""

from __future__ import annotations

import numpy as np
from scipy.signal import lfilter

MIN_ECHO_MS = 20.0
MAX_ECHO_MS = 1000.0
MAX_FEEDBACK = 0.9
"""回授上限。到 1.0 就是不會衰減的無限迴圈,實際上會一路累積到破音。"""


class Echo:
    """單點回授回音。四個參數都可以即時改,不會爆音。"""

    def __init__(self, samplerate: int, channels: int,
                 max_ms: float = MAX_ECHO_MS) -> None:
        self.samplerate = int(samplerate)
        self.channels = int(channels)
        self.max_samples = int(self.samplerate * max_ms / 1000.0)

        # 跟 DelayLine 同樣的理由:緩衝要比最大延遲再寬裕一個 block,
        # 否則延遲拉到最大時,讀取範圍會跟這次要寫進去的資料重疊。
        self._capacity = self.max_samples + 4096
        self._buf = np.zeros((self._capacity, self.channels), dtype=np.float32)
        self._write = 0

        self.enabled = False
        self._delay_ms = 180.0
        self._feedback = 0.35
        self._mix = 0.25
        self._damping = 0.35

        self._delay = self._samples_for(self._delay_ms)   # 目前生效的延遲
        self._target = self._delay                        # 下一個 block 淡接過去
        self._lp_zi = np.zeros((1, self.channels), dtype=np.float64)

    # ── 參數 ────────────────────────────────────────────────────────────
    def _samples_for(self, ms: float) -> int:
        samples = int(round(float(ms) * self.samplerate / 1000.0))
        return max(1, min(self.max_samples, samples))

    @property
    def delay_ms(self) -> float:
        return self._target / self.samplerate * 1000.0

    @delay_ms.setter
    def delay_ms(self, value: float) -> None:
        value = max(MIN_ECHO_MS, min(MAX_ECHO_MS, float(value)))
        self._delay_ms = value
        self._target = self._samples_for(value)

    @property
    def feedback(self) -> float:
        return self._feedback

    @feedback.setter
    def feedback(self, value: float) -> None:
        self._feedback = max(0.0, min(MAX_FEEDBACK, float(value)))

    @property
    def mix(self) -> float:
        return self._mix

    @mix.setter
    def mix(self, value: float) -> None:
        self._mix = max(0.0, min(1.0, float(value)))

    @property
    def damping(self) -> float:
        return self._damping

    @damping.setter
    def damping(self, value: float) -> None:
        self._damping = max(0.0, min(0.95, float(value)))

    def as_dict(self) -> dict:
        return {
            "enabled": self.enabled,
            "delay_ms": round(self.delay_ms, 1),
            "feedback": round(self._feedback, 3),
            "mix": round(self._mix, 3),
            "damping": round(self._damping, 3),
            "min_delay_ms": MIN_ECHO_MS,
            "max_delay_ms": MAX_ECHO_MS,
        }

    def reset(self) -> None:
        """清掉尾音。關掉效果或重開串流時用,免得下次開啟時漏出舊的回音。"""
        self._buf.fill(0.0)
        self._write = 0
        self._delay = self._target
        self._lp_zi[:] = 0.0

    # ── 處理 ────────────────────────────────────────────────────────────
    def _read(self, delay: int, frames: int) -> np.ndarray:
        """取出「從目前寫入位置往前推 delay」開始的那一段。"""
        start = (self._write - delay) % self._capacity
        end = start + frames
        if end <= self._capacity:
            return self._buf[start:end].copy()
        first = self._capacity - start
        out = np.empty((frames, self.channels), dtype=np.float32)
        out[:first] = self._buf[start:]
        out[first:] = self._buf[:frames - first]
        return out

    def _write_block(self, block: np.ndarray) -> None:
        frames = len(block)
        end = self._write + frames
        if end <= self._capacity:
            self._buf[self._write:end] = block
        else:
            first = self._capacity - self._write
            self._buf[self._write:] = block[:first]
            self._buf[:frames - first] = block[first:]
        self._write = end % self._capacity

    def _damped(self, tail: np.ndarray) -> np.ndarray:
        """回授路徑上的一階低通。damping = 0 時直接旁通。"""
        if self._damping <= 1e-4:
            return tail
        a = self._damping
        y, self._lp_zi = lfilter([1.0 - a], [1.0, -a],
                                 tail.astype(np.float64, copy=False),
                                 axis=0, zi=self._lp_zi)
        return y.astype(np.float32, copy=False)

    def process(self, x: np.ndarray) -> np.ndarray:
        """x: (frames, channels) float32 → 加了回音的新陣列(關閉時原樣回傳)。"""
        frames = len(x)
        if frames == 0:
            return x
        if not self.enabled or self._mix <= 1e-4:
            return x
        if x.shape[1] != self.channels:
            raise ValueError(
                f"回音是 {self.channels} 聲道,收到 {x.shape[1]} 聲道")

        # 延遲量必須大於一個 block,整段才算得起來(見模組說明)。
        # 夾住的是「這次實際使用的值」,使用者設定的 _target 不動 ——
        # block 之後變小時要能回到他原本要的長度。
        target = max(frames, self._target)
        delay = max(frames, self._delay)

        if delay == target:
            tail = self._read(delay, frames)
        else:
            # 延遲剛被改動:舊位置淡出、新位置淡入,免得波形接縫爆音
            ramp = np.linspace(0.0, 1.0, frames, dtype=np.float32)[:, np.newaxis]
            tail = (self._read(delay, frames) * (1.0 - ramp)
                    + self._read(target, frames) * ramp)
        # 記的是「這個 block 實際用到的量」而不是使用者設定值 —— 兩者在
        # block 比延遲還長時會不一樣,記錯下次就會多淡接一次。
        self._delay = target

        tail = self._damped(tail)
        self._write_block(x + tail * self._feedback)
        return x + tail * self._mix
