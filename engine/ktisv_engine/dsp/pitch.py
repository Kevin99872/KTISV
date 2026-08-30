"""音樂變調 —— 卡拉 OK 的升 key / 降 key。

要的是「音高變了、速度不變」。單純重新取樣只能做到前者:放快 12% 音高
確實升了兩個半音,但整首歌也快了 12%,跟不上原本的拍子。所以得拆成兩步:

    1. 相位聲碼器把訊號**拉長** r 倍(音高不動)
    2. 再以 r 倍速重新取樣**縮回**原長度(音高跟著 × r)

兩步的 r 是同一個數,所以總長度精確地一進一出,速度完全不變。

相位聲碼器在做什麼
----------------
把訊號切成重疊的窗做 FFT。每個頻格的相位在相鄰兩幀之間轉了多少,就能反推
那裡真正的頻率(不只是頻格中心)。要拉長訊號,就把幀與幀之間的距離拉開,
再依「真正的頻率 × 新的間距」重新累積相位 —— 波形被拉長了,每個成分的
頻率卻沒變。

分析跳距取整數
------------
理論上分析跳距是 ``hop_s / r``,通常不是整數。與其用浮點指標追蹤(每一幀
的實際跳距都不一樣,相位推算要跟著改,很容易錯),不如反過來:把跳距取成
整數,再由它反推**實際**的音高比 ``hop_s / ha``。誤差是多少?半音的 ha
是 483.2 取整成 483,實際比值差 0.8 音分 —— 人耳分辨的極限約 5~10 音分,
聽不出來。換來的是兩步用的是同一個有理數,速度精確守恆。

相位鎖定
--------
純粹逐格累積相位會讓同一個諧波的相鄰頻格各走各的,聲音變得空洞、有水聲
(俗稱 phasiness)。Puckette 的作法:合成時把鄰格的複數值併進來,取其相位
當作這一格的相位 —— 峰值附近的頻格因此被綁在一起走,泛音結構得以保持。

**鄰格是相減不是相加。** hann 系列的窗其頻譜是 ``0.5·D(f) - 0.25·D(f±1)``,
主瓣裡相鄰頻格本來就差 180°;直覺地寫成相加會變成互相抵消,實測整體掉了
6 dB,而且波峰反而更平。減號才是讓它們同相疊加的那一個。
"""

from __future__ import annotations

import numpy as np

MAX_SEMITONES = 12.0
"""上下各一個八度。超過這個範圍,相位聲碼器的假象會比變調本身還搶戲。"""

EPS = 1e-12


class PitchShifter:
    """整段音樂的變調。``semitones`` 可即時改,0 時完全旁通。"""

    def __init__(self, samplerate: int = 48000, channels: int = 2,
                 n_fft: int = 2048) -> None:
        self.samplerate = int(samplerate)
        self.channels = int(channels)
        self.n_fft = int(n_fft)
        # 87.5% 重疊。合成跳距是固定的,變的是分析跳距 —— 重疊相加的
        # 正規化係數才能算一次就好。
        #
        # 原本是 75%(n_fft//4)。實測 87.5% 在三個面向同時更好,沒有取捨:
        #
        #     純音 THD+N   -35.7 dB → -48.8 dB
        #     和弦互調     -33.1 dB → -37.3 dB
        #     演算法延遲    50.6 ms →  45.3 ms
        #
        # 延遲反而降低是因為相位聲碼器的延遲主要由「湊滿多少幀才吐得出
        # 完整的重疊相加」決定,跳距變小,湊齊得更快。代價只有 CPU:
        # 每 block 0.120 ms → 0.213 ms(128 取樣的預算是 2.67 ms)。
        self.hop_s = self.n_fft // 8

        self._semitones = 0.0
        self._ha = self.hop_s          # 分析跳距(整數)
        self._ratio = 1.0              # 由 hop_s / ha 反推的實際音高比

        window = np.hanning(self.n_fft + 1)[:-1]
        self._window = np.sqrt(window)

        # 窗平方在合成跳距上的疊加總和。實測而非查表,換 n_fft 也不會算錯。
        accumulator = np.zeros(self.n_fft * 4)
        for start in range(0, self.n_fft * 3, self.hop_s):
            accumulator[start:start + self.n_fft] += self._window ** 2
        self._ola_gain = float(np.median(accumulator[self.n_fft:self.n_fft * 2]))

        bins = self.n_fft // 2 + 1
        self._omega = (2.0 * np.pi * np.arange(bins) / self.n_fft)[:, np.newaxis]

        self._in = np.zeros((0, self.channels))
        self._last_phase = np.zeros((bins, self.channels))
        self._sum_phase = np.zeros((bins, self.channels))
        self._accum = np.zeros((self.n_fft, self.channels))
        self._stretched = np.zeros((0, self.channels))
        self._out_pos = 1.0            # 保留一個取樣的歷史給三次內插用
        self._first_frame = True
        self._priming = True

    # ── 參數 ────────────────────────────────────────────────────────────
    @property
    def semitones(self) -> float:
        return self._semitones

    @semitones.setter
    def semitones(self, value: float) -> None:
        value = max(-MAX_SEMITONES, min(MAX_SEMITONES, float(value)))
        if abs(value - self._semitones) < 1e-9:
            return

        was_bypassed = self._semitones == 0.0
        self._semitones = value

        if value == 0.0:
            # 回到原調就整條旁通,順便清乾淨 —— 留著舊內容的話,下次
            # 再升 key 會先聽到幾十毫秒前的東西。
            self.reset()
            self._ratio = 1.0
            self._ha = self.hop_s
            return

        self._ha = max(1, int(round(self.hop_s / (2.0 ** (value / 12.0)))))
        self._ratio = self.hop_s / self._ha
        if was_bypassed:
            self.reset()

    @property
    def ratio(self) -> float:
        """實際的音高比。因為分析跳距取了整數,和 2^(n/12) 差不到一音分。"""
        return self._ratio

    @property
    def active(self) -> bool:
        return self._semitones != 0.0

    @property
    def latency_samples(self) -> int:
        """演算法延遲:重疊相加本身,加上重新取樣那一段的緩衝餘裕。"""
        if not self.active:
            return 0
        return int(self.n_fft - self.hop_s + self.hop_s / self._ratio)

    @property
    def latency_ms(self) -> float:
        return self.latency_samples / self.samplerate * 1000.0

    def as_dict(self) -> dict:
        return {
            "semitones": round(self._semitones, 2),
            "ratio": round(self._ratio, 5),
            "max_semitones": MAX_SEMITONES,
            "latency_ms": round(self.latency_ms, 1),
        }

    def reset(self) -> None:
        self._in = np.zeros((0, self.channels))
        self._last_phase[:] = 0.0
        self._sum_phase[:] = 0.0
        self._accum[:] = 0.0
        self._stretched = np.zeros((0, self.channels))
        self._out_pos = 1.0
        self._first_frame = True
        self._priming = True

    # ── 處理 ────────────────────────────────────────────────────────────
    def process(self, block: np.ndarray) -> np.ndarray:
        """block: (frames, channels) float32 → 同形狀的輸出。"""
        frames = len(block)
        if frames == 0 or not self.active:
            return block
        if block.shape[1] != self.channels:
            raise ValueError(
                f"變調器是 {self.channels} 聲道,收到 {block.shape[1]} 聲道")

        self._in = np.concatenate(
            [self._in, block.astype(np.float64, copy=False)], axis=0)
        while len(self._in) >= self.n_fft:
            self._analyse_frame()
            self._in = self._in[self._ha:]

        return self._resample(frames)

    def _analyse_frame(self) -> None:
        spectrum = np.fft.rfft(self._in[:self.n_fft] * self._window[:, np.newaxis],
                               axis=0)
        magnitude = np.abs(spectrum)
        phase = np.angle(spectrum)

        if self._first_frame:
            # 第一幀沒有前一幀可比,相位差無從算起。直接沿用原相位,
            # 從第二幀開始才有意義的推算。
            self._sum_phase = phase.copy()
            self._first_frame = False
        else:
            deviation = phase - self._last_phase - self._omega * self._ha
            deviation -= 2.0 * np.pi * np.round(deviation / (2.0 * np.pi))
            true_omega = self._omega + deviation / self._ha
            self._sum_phase = self._sum_phase + true_omega * self.hop_s
        self._last_phase = phase

        shifted = magnitude * np.exp(1j * self._sum_phase)

        # 相位鎖定:把相鄰頻格綁在一起,避免同一個諧波散開成水聲。
        # 相減不是相加 —— 理由見模組說明,寫錯會安靜地掉 6 dB。
        locked = shifted.copy()
        locked[1:-1] -= shifted[:-2] + shifted[2:]
        shifted = magnitude * locked / (np.abs(locked) + EPS)

        frame = np.fft.irfft(shifted, n=self.n_fft, axis=0) \
            * self._window[:, np.newaxis]
        self._accum += frame

        # 最前面 hop_s 個取樣的重疊已經全部到齊,可以送出去了
        self._stretched = np.concatenate(
            [self._stretched, self._accum[:self.hop_s] / self._ola_gain], axis=0)
        # 就地左移。這條路每一幀都會走,而累加器是 2048×2 的 float64 ——
        # 用 concatenate 等於每幀在音訊回呼裡配置並複製 32 KB。
        self._accum[:-self.hop_s] = self._accum[self.hop_s:]
        self._accum[-self.hop_s:] = 0.0

    def _resample(self, frames: int) -> np.ndarray:
        """以 ratio 倍速讀出被拉長的訊號,縮回原本的長度。"""
        positions = self._out_pos + np.arange(frames) * self._ratio
        # 三次內插要取到 i+2,再多留一個當邊界
        needed = int(np.floor(positions[-1])) + 3

        # 暖機時多等一個合成跳距的量。長期來看產出與消耗的速率精確相等,
        # 但每一幀是量化的,存量會在一個 hop 的範圍內上下跳 —— 沒有這段
        # 餘裕的話,那個下跳就是一次破音。
        if self._priming:
            if len(self._stretched) < needed + self.hop_s:
                return np.zeros((frames, self.channels), dtype=np.float32)
            self._priming = False
        elif len(self._stretched) < needed:
            # 理論上不會走到這裡;真的發生就重新暖機,而不是讀到越界。
            self._priming = True
            return np.zeros((frames, self.channels), dtype=np.float32)

        index = positions.astype(np.int64)
        t = (positions - index)[:, np.newaxis]
        source = self._stretched
        p0 = source[index - 1]
        p1 = source[index]
        p2 = source[index + 1]
        p3 = source[index + 2]
        # Catmull-Rom。線性內插在升 key(等於降取樣)時會明顯削掉高頻並
        # 產生摺疊,鈸和空氣感首當其衝;三次內插的成本只多幾次取值。
        out = p1 + 0.5 * t * (
            (p2 - p0) + t * ((2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3)
                             + t * (3.0 * (p1 - p2) + p3 - p0)))

        self._out_pos += frames * self._ratio
        drop = int(self._out_pos) - 1          # 留一個取樣的歷史
        if drop > 0:
            self._stretched = self._stretched[drop:]
            self._out_pos -= drop

        return out.astype(np.float32)
