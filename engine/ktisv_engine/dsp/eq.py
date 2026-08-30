"""參數式等化器。

頻段數量、每一段的頻率與 Q 值都可以即時改。最低的那一段用 low-shelf、
最高的用 high-shelf、其餘用 peaking,係數採 RBJ Audio EQ Cookbook 的公式,
以 second-order sections 串接後由 ``scipy.signal.sosfilt`` 逐 block 濾波
(保留 zi 狀態以維持連續性)。

為什麼頻段是物件而不是兩條平行的 list
------------------------------------
使用者可以在播放中新增或刪除頻段,而每一段都帶著自己的濾波器狀態(zi)。
用索引對應的話,刪掉中間一段會讓後面所有段的狀態全部錯位,聽起來就是一聲
爆音加上一段亂掉的殘響。改成一段一個物件,重建係數矩陣時就能靠物件本身
認出「這是同一段」,把狀態原封不動搬過去 —— 只有真正被刪掉的那一段會歸零。
"""

from __future__ import annotations

import math

import numpy as np
from scipy.signal import sosfilt

DEFAULT_BANDS: tuple[float, ...] = (
    31.25, 62.5, 125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0, 16000.0,
)
DEFAULT_Q = 1.41
SHELF_Q = 0.7
"""兩端 shelf 的預設 Q。控制的是轉折的陡峭程度,不是頻寬。"""

GAIN_LIMIT_DB = 15.0
MIN_FREQ = 20.0
MIN_Q = 0.1
MAX_Q = 18.0
MAX_BANDS = 24
"""頻段數上限。每一段都是一顆 biquad,毫無節制地加下去只會吃掉音訊回呼的預算。"""


def _peaking(f0: float, gain_db: float, q: float, sr: int) -> np.ndarray:
    a = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * math.pi * f0 / sr
    alpha = math.sin(w0) / (2.0 * q)
    cos_w0 = math.cos(w0)

    b0 = 1.0 + alpha * a
    b1 = -2.0 * cos_w0
    b2 = 1.0 - alpha * a
    a0 = 1.0 + alpha / a
    a1 = -2.0 * cos_w0
    a2 = 1.0 - alpha / a
    return np.array([b0 / a0, b1 / a0, b2 / a0, 1.0, a1 / a0, a2 / a0], dtype=np.float64)


def _shelf(f0: float, gain_db: float, q: float, sr: int, high: bool) -> np.ndarray:
    a = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * math.pi * f0 / sr
    cos_w0 = math.cos(w0)
    sin_w0 = math.sin(w0)

    # Cookbook 的 shelf 用 q 當「斜率 S」。S 一大、增益又深時,根號裡會變成
    # 負的 —— 那不是「更陡」而是無解,算出來是 NaN,整條濾波鏈會一次污染成
    # NaN 然後徹底沒聲音。既然 Q 現在是使用者可以拉到 18 的旋鈕,這裡就得
    # 自己把根號夾在正數,超過物理上限的部分就停在最陡。
    radicand = (a + 1.0 / a) * (1.0 / q - 1.0) + 2.0
    alpha = sin_w0 / 2.0 * math.sqrt(max(radicand, 0.05))
    two_sqrt_a_alpha = 2.0 * math.sqrt(a) * alpha

    if high:
        b0 = a * ((a + 1.0) + (a - 1.0) * cos_w0 + two_sqrt_a_alpha)
        b1 = -2.0 * a * ((a - 1.0) + (a + 1.0) * cos_w0)
        b2 = a * ((a + 1.0) + (a - 1.0) * cos_w0 - two_sqrt_a_alpha)
        a0 = (a + 1.0) - (a - 1.0) * cos_w0 + two_sqrt_a_alpha
        a1 = 2.0 * ((a - 1.0) - (a + 1.0) * cos_w0)
        a2 = (a + 1.0) - (a - 1.0) * cos_w0 - two_sqrt_a_alpha
    else:
        b0 = a * ((a + 1.0) - (a - 1.0) * cos_w0 + two_sqrt_a_alpha)
        b1 = 2.0 * a * ((a - 1.0) - (a + 1.0) * cos_w0)
        b2 = a * ((a + 1.0) - (a - 1.0) * cos_w0 - two_sqrt_a_alpha)
        a0 = (a + 1.0) + (a - 1.0) * cos_w0 + two_sqrt_a_alpha
        a1 = -2.0 * ((a - 1.0) + (a + 1.0) * cos_w0)
        a2 = (a + 1.0) + (a - 1.0) * cos_w0 - two_sqrt_a_alpha

    return np.array([b0 / a0, b1 / a0, b2 / a0, 1.0, a1 / a0, a2 / a0], dtype=np.float64)


class EqBand:
    """一個頻段。``shelf`` 由所在位置決定,不是使用者設定的(見 GraphicEQ)。"""

    __slots__ = ("freq", "gain", "q", "shelf")

    def __init__(self, freq: float, gain: float = 0.0, q: float = DEFAULT_Q) -> None:
        self.freq = float(freq)
        self.gain = float(gain)
        self.q = float(q)
        self.shelf = ""      # "low" / "high" / ""(peaking)

    def to_dict(self) -> dict:
        return {"freq": round(self.freq, 2), "gain": round(self.gain, 2),
                "q": round(self.q, 3),
                "type": self.shelf + "_shelf" if self.shelf else "peaking"}


class GraphicEQ:
    """可增刪頻段的參數式 EQ。所有調整都能在播放中即時生效。"""

    def __init__(self, samplerate: int, channels: int,
                 bands=DEFAULT_BANDS, q: float = DEFAULT_Q) -> None:
        self.samplerate = samplerate
        self.channels = channels
        self.enabled = True
        self.default_q = float(q)

        self._bands: list[EqBand] = []
        self._sos = np.zeros((0, 6), dtype=np.float64)
        self._zi = np.zeros((0, 2, channels), dtype=np.float64)
        # _zi 的第 i 列屬於哪一段。增刪頻段時靠它把狀態搬到正確的新位置。
        self._zi_bands: list[EqBand] = []
        self._bypass = True
        self.set_bands(bands)

    # ── 查詢 ────────────────────────────────────────────────────────────
    @property
    def bands(self) -> tuple[float, ...]:
        """各段的中心頻率。"""
        return tuple(b.freq for b in self._bands)

    @property
    def gains(self) -> list[float]:
        return [b.gain for b in self._bands]

    @property
    def band_count(self) -> int:
        return len(self._bands)

    def band_info(self) -> list[dict]:
        return [b.to_dict() for b in self._bands]

    # ── 值的調整(不改變結構,濾波器狀態原封不動)──────────────────────
    def _clamp_freq(self, freq: float) -> float:
        # 超過 Nyquist 的頻段沒有意義:雙線性轉換會把它折回來,
        # 使用者看到的是一條完全不對應標籤的曲線。
        return max(MIN_FREQ, min(self.samplerate * 0.45, float(freq)))

    @staticmethod
    def _clamp_gain(gain_db: float) -> float:
        return max(-GAIN_LIMIT_DB, min(GAIN_LIMIT_DB, float(gain_db)))

    @staticmethod
    def _clamp_q(q: float) -> float:
        return max(MIN_Q, min(MAX_Q, float(q)))

    def set_gain(self, index: int, gain_db: float) -> None:
        self.set_band(index, gain=gain_db)

    def set_band(self, index: int, gain: float | None = None,
                 freq: float | None = None, q: float | None = None) -> None:
        """改單一頻段的任一項參數。傳 None 表示該項不動。"""
        if not 0 <= index < len(self._bands):
            return
        band = self._bands[index]
        changed = False

        if gain is not None:
            value = self._clamp_gain(gain)
            if abs(band.gain - value) >= 1e-6:
                band.gain = value
                changed = True
        if freq is not None:
            value = self._clamp_freq(freq)
            if abs(band.freq - value) >= 1e-6:
                band.freq = value
                changed = True
        if q is not None:
            value = self._clamp_q(q)
            if abs(band.q - value) >= 1e-6:
                band.q = value
                changed = True

        if changed:
            self._rebuild()

    def set_gains(self, gains_db) -> None:
        changed = False
        for band, g in zip(self._bands, gains_db):
            value = self._clamp_gain(g)
            if abs(band.gain - value) >= 1e-6:
                band.gain = value
                changed = True
        if changed:
            self._rebuild()

    def reset(self) -> None:
        """把所有增益歸零。頻率與 Q 是使用者的配置,不動。"""
        for band in self._bands:
            band.gain = 0.0
        self._rebuild()
        self.clear_state()

    # ── 結構的調整(會影響濾波器狀態的配置)────────────────────────────
    def set_bands(self, specs) -> None:
        """整組換掉。``specs`` 可以是頻率數列,或 {freq, gain, q} 的字典數列。"""
        raw = list(specs)
        bands: list[EqBand] = []
        for i, spec in enumerate(raw):
            if isinstance(spec, EqBand):
                band = spec
            elif isinstance(spec, dict):
                band = EqBand(spec.get("freq", 1000.0),
                              spec.get("gain", 0.0),
                              spec.get("q", self.default_q))
            else:
                # 只給頻率的寫法(例如 DEFAULT_BANDS):兩端會變成 shelf,
                # 而 shelf 的 q 是斜率而非頻寬,用 peaking 的預設值太陡。
                shelf = i == 0 or i == len(raw) - 1
                band = EqBand(spec, 0.0, SHELF_Q if shelf else self.default_q)
            band.freq = self._clamp_freq(band.freq)
            band.gain = self._clamp_gain(band.gain)
            band.q = self._clamp_q(band.q)
            bands.append(band)
            if len(bands) >= MAX_BANDS:
                break

        if not bands:
            raise ValueError("EQ 至少要有一個頻段。")

        bands.sort(key=lambda b: b.freq)
        self._bands = bands
        self._rebuild(structure_changed=True)

    def add_band(self, freq: float, gain: float = 0.0,
                 q: float | None = None) -> int:
        """新增一段,回傳它排序後的索引。"""
        if len(self._bands) >= MAX_BANDS:
            raise ValueError(f"最多只能有 {MAX_BANDS} 個頻段。")

        band = EqBand(self._clamp_freq(freq), self._clamp_gain(gain),
                      self._clamp_q(self.default_q if q is None else q))
        # 依頻率插入。shelf 是照位置給的(頭尾各一),所以維持頻率順序,
        # 兩端才會落在真正最低與最高的那兩段上。
        position = len(self._bands)
        for i, existing in enumerate(self._bands):
            if band.freq < existing.freq:
                position = i
                break
        self._bands.insert(position, band)
        self._rebuild(structure_changed=True)
        return position

    def remove_band(self, index: int) -> None:
        if not 0 <= index < len(self._bands):
            raise ValueError(f"沒有第 {index} 個頻段。")
        if len(self._bands) <= 1:
            raise ValueError("至少要保留一個頻段。")
        self._bands.pop(index)
        self._rebuild(structure_changed=True)

    def clear_state(self) -> None:
        self._zi[:] = 0.0

    # ── 係數 ────────────────────────────────────────────────────────────
    def _rebuild(self, structure_changed: bool = False) -> None:
        count = len(self._bands)

        if structure_changed:
            self._sos = np.zeros((count, 6), dtype=np.float64)
            # 重新配置 zi。整個歸零會在增刪頻段的瞬間爆一聲,所以逐段認人:
            # 留下來的那幾段把自己的狀態帶到新位置,只有新來的從零開始。
            previous = {id(band): row for row, band in enumerate(self._zi_bands)}
            zi = np.zeros((count, 2, self.channels), dtype=np.float64)
            for i, band in enumerate(self._bands):
                row = previous.get(id(band))
                if row is not None:
                    zi[i] = self._zi[row]
            self._zi = zi
            self._zi_bands = list(self._bands)

        last = count - 1
        for i, band in enumerate(self._bands):
            band.shelf = "low" if i == 0 and count > 1 else \
                         "high" if i == last and count > 1 else ""
            if band.shelf:
                self._sos[i] = _shelf(band.freq, band.gain, band.q,
                                      self.samplerate, high=band.shelf == "high")
            else:
                self._sos[i] = _peaking(band.freq, band.gain, band.q,
                                        self.samplerate)
        self._bypass = all(abs(b.gain) < 1e-3 for b in self._bands)

    # ── 處理 ────────────────────────────────────────────────────────────
    def process(self, x: np.ndarray) -> np.ndarray:
        """x: (frames, channels) float32 → 濾波後的新陣列。"""
        if not self.enabled or self._bypass:
            return x
        y, self._zi = sosfilt(self._sos, x.astype(np.float64, copy=False),
                              axis=0, zi=self._zi)
        return y.astype(np.float32, copy=False)

    def response(self, freqs: np.ndarray) -> np.ndarray:
        """回傳指定頻率上的合成響應(dB),給 UI 畫曲線用。"""
        from scipy.signal import sosfreqz

        w = 2.0 * np.pi * np.asarray(freqs, dtype=np.float64) / self.samplerate
        _, h = sosfreqz(self._sos, worN=w)
        return 20.0 * np.log10(np.maximum(np.abs(h), 1e-9))
