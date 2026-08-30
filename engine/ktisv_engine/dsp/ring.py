"""執行緒安全的環形緩衝區。

每一路音訊裝置都有自己的時脈,彼此會慢慢飄移。所有跨裝置的資料交換
都經過這裡,並用 slack(預留量)吸收飄移。
"""

from __future__ import annotations

import threading

import numpy as np


class RingBuffer:
    """單生產者 / 單消費者的 float32 環形緩衝。

    寫入溢位時丟棄最舊的資料,讀取不足時補零 —— 兩者都會被計數,
    方便 UI 顯示 xrun。
    """

    def __init__(self, capacity_frames: int, channels: int) -> None:
        self._capacity = int(capacity_frames)
        self._channels = int(channels)
        self._buf = np.zeros((self._capacity, self._channels), dtype=np.float32)
        self._read = 0
        self._write = 0
        self._filled = 0
        self._lock = threading.Lock()
        self.overflows = 0
        self.underflows = 0

    @property
    def channels(self) -> int:
        return self._channels

    @property
    def capacity(self) -> int:
        return self._capacity

    def available(self) -> int:
        with self._lock:
            return self._filled

    def clear(self) -> None:
        with self._lock:
            self._read = self._write = self._filled = 0
            self._buf.fill(0.0)

    def write(self, data: np.ndarray) -> int:
        n = len(data)
        if n == 0:
            return 0
        with self._lock:
            if n > self._capacity:
                data = data[-self._capacity :]
                n = self._capacity
                self.overflows += 1
            free = self._capacity - self._filled
            if n > free:
                drop = n - free
                self._read = (self._read + drop) % self._capacity
                self._filled -= drop
                self.overflows += 1
            first = min(n, self._capacity - self._write)
            self._buf[self._write : self._write + first] = data[:first]
            rest = n - first
            if rest:
                self._buf[:rest] = data[first:]
            self._write = (self._write + n) % self._capacity
            self._filled += n
            return n

    def read(self, frames: int) -> np.ndarray:
        out = np.zeros((frames, self._channels), dtype=np.float32)
        with self._lock:
            n = min(frames, self._filled)
            if n < frames:
                self.underflows += 1
            if n:
                first = min(n, self._capacity - self._read)
                out[:first] = self._buf[self._read : self._read + first]
                rest = n - first
                if rest:
                    out[first:n] = self._buf[:rest]
                self._read = (self._read + n) % self._capacity
                self._filled -= n
        return out

    def drop(self, frames: int) -> int:
        """丟掉最舊的 frames 個取樣(用來修正時脈飄移造成的延遲累積)。"""
        with self._lock:
            n = min(frames, self._filled)
            self._read = (self._read + n) % self._capacity
            self._filled -= n
            return n
