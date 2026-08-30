"""頻譜域中央聲道分離。

與時域版本(``separation.py``)的差別
------------------------------------
時域 mid/side 對**整個頻段**做同一個決策:mid 裡中頻的東西全部被移除。
所以一把偏左 30% 的吉他,它的 mid 成分也會被削掉 —— 明明它不是人聲。

這裡改成**逐時頻格**判斷:對每個 (時間, 頻率) 格子單獨評估「這裡的內容
有多置中」,再依此決定要移除多少。偏一邊的樂器因為不置中,自然被保留;
真正置中的人聲才被移除。

置中程度怎麼算
--------------
兩個條件都要滿足才算置中:

  * **相位一致** —— 左右同相。用正規化的互相關 Re(L·conj(R)) / (|L||R|)
  * **振幅平衡** —— 左右一樣大。用 1 - ||L|-|R|| / (|L|+|R|)

兩者相乘。只有同相**且**等大的內容才會被判定為置中,單一條件成立不算。

延遲的取捨
----------
STFT 需要湊滿一個窗才能輸出,所以會引入約 ``n_fft`` 取樣的延遲。這條路徑
只處理**音樂**,不處理麥克風,所以不影響耳返。但唱歌時你是跟著音樂唱的,
音樂延遲會讓你跟著慢 —— 預設 1024(48 kHz 下約 21 ms)是品質與延遲的折衷。
"""

from __future__ import annotations

import numpy as np

MODE_OFF = "off"
MODE_REMOVE_VOCALS = "remove_vocals"
MODE_ISOLATE_VOCALS = "isolate_vocals"
MODE_SILENCE = "silence"

EPS = 1e-12


class SpectralCenterSeparator:
    """逐頻格的置中內容分離器。"""

    def __init__(self, samplerate: int = 48000, n_fft: int = 1024,
                 low_cut: float = 180.0, high_cut: float = 9000.0) -> None:
        self.samplerate = samplerate
        self.n_fft = int(n_fft)
        self.hop = self.n_fft // 4          # 75% 重疊
        self.mode = MODE_OFF
        self.strength = 1.0

        # 判定的銳利度。越大越只移除「非常置中」的內容,對伴奏越安全;
        # 越小則移除得越徹底,但誤傷也越多。
        self.sharpness = 2.0

        self._low_cut = low_cut
        self._high_cut = high_cut

        # sqrt(hann) 同時用於分析與合成,75% 重疊下能完美重建
        window = np.hanning(self.n_fft + 1)[:-1]
        self._window = np.sqrt(window).astype(np.float64)

        # 重疊相加的正規化係數:實測窗平方的疊加總和,比查表可靠
        accumulator = np.zeros(self.n_fft * 4)
        for start in range(0, self.n_fft * 3, self.hop):
            accumulator[start:start + self.n_fft] += self._window ** 2
        self._ola_gain = float(np.median(
            accumulator[self.n_fft:self.n_fft * 2]))

        self._freqs = np.fft.rfftfreq(self.n_fft, 1.0 / samplerate)
        self._build_band_weight()

        # 標準的滑動重疊相加:累加器長度固定為一個窗,每處理一幀就把
        # 最前面 hop 個取樣視為完成、輸出並左移。
        self._in_buffer = np.zeros((0, 2), dtype=np.float64)
        self._accum = np.zeros((self.n_fft, 2), dtype=np.float64)
        self._ready = np.zeros((0, 2), dtype=np.float64)

    # ── 參數 ────────────────────────────────────────────────────────────
    @property
    def low_cut(self) -> float:
        return self._low_cut

    @low_cut.setter
    def low_cut(self, value: float) -> None:
        value = max(20.0, min(600.0, float(value)))
        if abs(value - self._low_cut) > 1e-3:
            self._low_cut = value
            self._build_band_weight()

    @property
    def high_cut(self) -> float:
        return self._high_cut

    @high_cut.setter
    def high_cut(self, value: float) -> None:
        value = max(2000.0, min(self.samplerate * 0.45, float(value)))
        if abs(value - self._high_cut) > 1e-3:
            self._high_cut = value
            self._build_band_weight()

    def _build_band_weight(self) -> None:
        """頻段權重:人聲頻段為 1,兩端平滑降到 0。

        用平滑過渡而非硬切,避免在邊界產生可聽見的假象。
        置中的貝斯與大鼓落在低端、鈸與空氣感落在高端,權重接近 0 就會被保留。
        """
        weight = np.ones_like(self._freqs)

        # 低端:low_cut 的一個八度以下完全保留
        low_start = self._low_cut * 0.5
        low_mask = self._freqs < self._low_cut
        weight[low_mask] = np.clip(
            (self._freqs[low_mask] - low_start) / max(self._low_cut - low_start, 1e-6),
            0.0, 1.0)

        # 高端:high_cut 到其 1.5 倍之間平滑降下
        high_end = self._high_cut * 1.5
        high_mask = self._freqs > self._high_cut
        weight[high_mask] = np.clip(
            (high_end - self._freqs[high_mask]) / max(high_end - self._high_cut, 1e-6),
            0.0, 1.0)

        self._band_weight = weight

    @property
    def latency_samples(self) -> int:
        """演算法延遲(取樣)。

        實測值是 ``n_fft - hop``(即 ¾ 個窗長)—— 一個取樣要等到它所在的
        所有重疊視窗都處理完才算完成。

        注意:若音訊 block 大小不是 ``hop`` 的整數倍,還會多出最多一個 block
        的量化延遲。48 kHz / block=480 / n_fft=1024 時實測總延遲為 960 取樣
        (20 ms),其中 768 是演算法本身、192 是量化。
        """
        return self.n_fft - self.hop

    @property
    def latency_ms(self) -> float:
        return self.latency_samples / self.samplerate * 1000.0

    def reset(self) -> None:
        self._in_buffer = np.zeros((0, 2), dtype=np.float64)
        self._accum = np.zeros((self.n_fft, 2), dtype=np.float64)
        self._ready = np.zeros((0, 2), dtype=np.float64)

    # ── 處理 ────────────────────────────────────────────────────────────
    def process(self, block: np.ndarray) -> np.ndarray:
        """block: (frames, 2) float32 → 同形狀的輸出。"""
        frames = len(block)
        if self.mode == MODE_OFF or self.strength <= 0.0:
            return block
        if self.mode == MODE_SILENCE:
            return np.zeros_like(block)
        if block.shape[1] < 2:
            return block if self.mode == MODE_ISOLATE_VOCALS else np.zeros_like(block)

        self._in_buffer = np.concatenate(
            [self._in_buffer, block.astype(np.float64, copy=False)], axis=0)

        # 每湊滿一個窗就處理一幀:疊加進累加器 → 最前面 hop 個取樣已完成
        # → 輸出並左移。累加器長度恆為 n_fft。
        while len(self._in_buffer) >= self.n_fft:
            processed = self._process_frame(self._in_buffer[:self.n_fft])
            self._accum += processed

            self._ready = np.concatenate(
                [self._ready, self._accum[:self.hop] / self._ola_gain], axis=0)

            self._accum = np.concatenate(
                [self._accum[self.hop:], np.zeros((self.hop, 2))], axis=0)
            self._in_buffer = self._in_buffer[self.hop:]

        if len(self._ready) >= frames:
            out = self._ready[:frames]
            self._ready = self._ready[frames:]
        else:
            # 剛啟動時還湊不滿一個窗,先補靜音
            out = np.zeros((frames, 2), dtype=np.float64)
            if len(self._ready):
                out[:len(self._ready)] = self._ready
            self._ready = np.zeros((0, 2), dtype=np.float64)

        return out.astype(np.float32)

    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        windowed = frame * self._window[:, np.newaxis]
        left = np.fft.rfft(windowed[:, 0])
        right = np.fft.rfft(windowed[:, 1])

        mag_l, mag_r = np.abs(left), np.abs(right)

        # 相位一致度:同相為 1,反相為 -1
        coherence = np.real(left * np.conj(right)) / (mag_l * mag_r + EPS)
        coherence = np.clip(coherence, 0.0, 1.0)

        # 振幅平衡:左右等大為 1
        balance = 1.0 - np.abs(mag_l - mag_r) / (mag_l + mag_r + EPS)

        centeredness = (coherence * balance) ** self.sharpness
        vocal_mask = np.clip(centeredness * self._band_weight, 0.0, 1.0) * self.strength

        mid = 0.5 * (left + right)
        vocal = mid * vocal_mask

        if self.mode == MODE_REMOVE_VOCALS:
            out_l, out_r = left - vocal, right - vocal
        else:                                    # MODE_ISOLATE_VOCALS
            out_l = out_r = vocal

        result = np.empty((self.n_fft, 2), dtype=np.float64)
        result[:, 0] = np.fft.irfft(out_l, n=self.n_fft)
        result[:, 1] = np.fft.irfft(out_r, n=self.n_fft)
        return result * self._window[:, np.newaxis]


def mode_from_flags(remove_vocals: bool, remove_instrumental: bool) -> str:
    if remove_vocals and remove_instrumental:
        return MODE_SILENCE
    if remove_vocals:
        return MODE_REMOVE_VOCALS
    if remove_instrumental:
        return MODE_ISOLATE_VOCALS
    return MODE_OFF
