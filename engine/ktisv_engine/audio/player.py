"""音軌播放器。

整首歌以 float32 陣列常駐記憶體(3 分鐘立體聲 48 kHz ≈ 70 MB),
所以 seek 是零成本的,音訊回呼裡只是切片與加權相加。

分軌(stem)以具名字典保存:
  * 即時模式  → {"mix": ...}
  * Demucs 模式 → {"vocals": ..., "instrumental": ...}
每一軌都有自己的平滑增益,勾選框就是把對應音軌的增益推到 0。
"""

from __future__ import annotations

import threading

import numpy as np

from ..dsp.gain import SmoothGain


class StemPlayer:
    def __init__(self, samplerate: int = 48000) -> None:
        self.samplerate = samplerate
        self._lock = threading.RLock()
        self._stems: dict[str, np.ndarray] = {}
        self._gains: dict[str, SmoothGain] = {}
        self._length = 0
        self._position = 0
        self._playing = False       # 使用者意圖:是否要播放
        self._active = False        # 實際狀態:是否仍在輸出(含淡出尾巴)
        self.loop = False
        self._finished = False
        self.title = ""
        # 播放/暫停/跳轉都經過這個 8 ms 的淡入淡出,否則波形被硬切會產生
        # 爆音,還會激發下游 EQ 與分離濾波器的暫態。
        self._transport = SmoothGain(0.0, 8.0, samplerate)

    # ── 載入 ────────────────────────────────────────────────────────────
    def load(self, stems: dict[str, np.ndarray], title: str = "") -> None:
        prepared: dict[str, np.ndarray] = {}
        length = 0
        for name, data in stems.items():
            arr = np.asarray(data, dtype=np.float32)
            if arr.ndim == 1:
                arr = np.column_stack([arr, arr])
            elif arr.shape[1] == 1:
                arr = np.repeat(arr, 2, axis=1)
            elif arr.shape[1] > 2:
                arr = arr[:, :2]
            prepared[name] = np.ascontiguousarray(arr)
            length = max(length, len(arr))

        with self._lock:
            self._stems = prepared
            self._gains = {
                name: self._gains.get(name) or SmoothGain(1.0, 30.0, self.samplerate)
                for name in prepared
            }
            for name in list(self._gains):
                if name not in prepared:
                    del self._gains[name]
            self._length = length
            self._position = 0
            self._finished = False
            self.title = title

    def unload(self) -> None:
        with self._lock:
            self._stems = {}
            self._gains = {}
            self._length = 0
            self._position = 0
            self._playing = False
            self.title = ""

    # ── 傳輸控制 ────────────────────────────────────────────────────────
    @property
    def loaded(self) -> bool:
        return self._length > 0

    @property
    def stem_names(self) -> list[str]:
        return list(self._stems)

    @property
    def playing(self) -> bool:
        return self._playing

    @property
    def duration(self) -> float:
        return self._length / self.samplerate if self.samplerate else 0.0

    @property
    def position(self) -> float:
        return self._position / self.samplerate if self.samplerate else 0.0

    def play(self) -> None:
        with self._lock:
            if not self.loaded:
                return
            if self._finished:
                self._position = 0
                self._finished = False
            self._playing = True
            self._active = True
            self._transport.target = 1.0

    def pause(self) -> None:
        with self._lock:
            self._playing = False
            self._transport.target = 0.0   # _active 會在淡出結束後自行關閉

    def toggle(self) -> None:
        if self._playing:
            self.pause()
        else:
            self.play()

    def stop(self) -> None:
        with self._lock:
            self._playing = False
            self._active = False
            self._transport.snap(0.0)
            self._position = 0
            self._finished = False

    def seek(self, seconds: float) -> None:
        with self._lock:
            frame = int(max(0.0, seconds) * self.samplerate)
            self._position = min(frame, self._length)
            self._finished = False
            # 從新位置淡入,避免跳轉造成的波形不連續
            self._transport.snap(0.0)
            if self._playing:
                self._transport.target = 1.0

    # ── 分軌增益 ────────────────────────────────────────────────────────
    def set_stem_gain(self, name: str, value: float) -> None:
        with self._lock:
            gain = self._gains.get(name)
            if gain is not None:
                gain.target = max(0.0, float(value))

    def set_stem_gains(self, values: dict[str, float]) -> None:
        for name, value in values.items():
            self.set_stem_gain(name, value)

    # ── 音訊回呼 ────────────────────────────────────────────────────────
    def read(self, frames: int) -> np.ndarray:
        """回傳 (frames, 2) 的混合結果;沒有播放時是靜音。"""
        out = np.zeros((frames, 2), dtype=np.float32)
        with self._lock:
            if not self.loaded:
                return out

            if not self._active:
                # 停住的時候增益仍要繼續收斂,否則恢復播放會突然跳一下
                for gain in self._gains.values():
                    gain.envelope(frames)
                self._transport.envelope(frames)
                return out

            start = self._position
            end = min(start + frames, self._length)
            count = end - start

            for name, gain in self._gains.items():
                env = gain.envelope(frames)
                if gain.is_silent():
                    continue
                data = self._stems[name]
                available = max(0, min(count, len(data) - start))
                if available <= 0:
                    continue
                out[:available] += data[start:start + available] * env[:available]

            out *= self._transport.envelope(frames)

            self._position = end
            if end >= self._length:
                if self.loop:
                    self._position = 0
                else:
                    self._playing = False
                    self._finished = True
                    self._transport.target = 0.0

            # 淡出走完才真正停止推進,之後的 block 直接回傳靜音
            if not self._playing and self._transport.current <= 0.0:
                self._active = False
        return out

    def state(self) -> dict:
        return {
            "loaded": self.loaded,
            "playing": self._playing,
            "finished": self._finished,
            "position": round(self.position, 3),
            "duration": round(self.duration, 3),
            "loop": self.loop,
            "title": self.title,
            "stems": self.stem_names,
        }
