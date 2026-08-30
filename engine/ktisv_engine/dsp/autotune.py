"""即時音高修正(autotune)。

做法
----
1. **偵測** —— YIN 找出你現在唱的基頻。
2. **吸附** —— 把它吸到最近的音階音上,可以只吸一部分(強度)、也可以慢慢
   吸過去(速度),那是「修得自然」與「修成電音」的差別。
3. **移調** —— PSOLA:找出每個基頻週期的位置,把以它為中心的一段加窗取出
   來當顆粒,再依**目標**週期重新排列疊加。

為什麼用 PSOLA 而不是現成的相位聲碼器
--------------------------------------
專案裡已經有一個相位聲碼器(``pitch.py``),但它的延遲是 45 ms,而且顆粒
之間的相位重建會讓人聲帶上「機器味」。PSOLA 直接搬移波形片段,顆粒本身的
頻譜(也就是共振峰)完全沒動 —— 人聲聽起來自然得多,延遲也只要一個基頻
週期。

延遲
----
無法低於「偵測窗 + 一個基頻週期」—— 修正不可能比「知道現在唱的是什麼音」
更早發生。實測偵測窗的地板是 25 ms(男低音 85 Hz 一個週期就 11.8 ms,
至少要看兩個週期):

    窗長 20 ms   男低音誤差 264 音分、21% 完全測不到
    窗長 25 ms   全音域誤差 1 音分以內、0% 失敗

加上顆粒半徑之後的總延遲:男低音 37 ms、女高音 27 ms。這跟商用 autotune
是同一個量級,是修音高本身的代價,不是實作問題。

八度誤判
--------
YIN 會把訊號判成低一個八度(或三分之一)—— 它取「第一個低於門檻的谷」,
而諧波豐富的聲音在 2τ、3τ 也會有夠深的谷。用在 autotune 上那代表音高整個
跳掉一個八度,非常明顯。這裡用兩道防線:候選週期的整數分之一若同樣夠好
就優先取短的,以及跟前一次的偵測結果做連續性比對。
"""

from __future__ import annotations

import numpy as np

# 偵測窗。低於 25 ms 男低音就測不準(實測 20 ms 誤差 264 音分)。
DETECT_MS = 25.0

# 人聲的合理範圍。放太寬只會讓 YIN 有更多機會誤判到不可能的音高。
MIN_HZ = 70.0
MAX_HZ = 800.0

# YIN 的判定門檻。越小越嚴格(寧可測不到也不亂猜)。
YIN_THRESHOLD = 0.12

# 低於這個音量就當作沒在唱 —— 換氣與句子之間不要亂修。
SILENCE_FLOOR = 0.004

SCALES: dict[str, tuple[int, ...]] = {
    "chromatic": tuple(range(12)),
    "major": (0, 2, 4, 5, 7, 9, 11),
    "minor": (0, 2, 3, 5, 7, 8, 10),
    "pentatonic": (0, 2, 4, 7, 9),
}

NOTE_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")


def detect_pitch(frame: np.ndarray, samplerate: int,
                 previous: float = 0.0) -> float:
    """YIN 音高偵測,含八度保護。回傳 Hz;沒把握回 0。"""
    n = len(frame)
    max_tau = min(int(samplerate / MIN_HZ), n // 2)
    min_tau = max(2, int(samplerate / MAX_HZ))
    if max_tau <= min_tau:
        return 0.0

    x = np.asarray(frame, dtype=np.float64)
    x = x - x.mean()
    if float(np.sqrt(np.mean(x ** 2))) < SILENCE_FLOOR:
        return 0.0

    # 差分函數用 FFT 自相關算,比雙迴圈快幾個數量級
    size = 1 << (2 * n - 1).bit_length()
    spec = np.fft.rfft(x, size)
    acf = np.fft.irfft(spec * np.conj(spec), size)[:max_tau + 1]
    power = np.concatenate([[0.0], np.cumsum(x ** 2)])
    taus = np.arange(1, max_tau + 1)
    head = power[n - taus] - power[0]
    tail = power[n] - power[taus]
    diff = np.concatenate([[0.0], head + tail - 2 * acf[1:]])

    cumulative = np.ones_like(diff)
    running = np.cumsum(diff[1:])
    cumulative[1:] = diff[1:] * np.arange(1, len(diff)) / np.maximum(running, 1e-12)

    window = cumulative[min_tau:max_tau]
    if window.size == 0:
        return 0.0

    below = np.flatnonzero(window < YIN_THRESHOLD)
    if below.size:
        tau = min_tau + int(below[0])
        # 走到這個谷的底部
        while tau + 1 < max_tau and cumulative[tau + 1] < cumulative[tau]:
            tau += 1
    else:
        tau = min_tau + int(np.argmin(window))
        if cumulative[tau] > 0.45:
            return 0.0

    # ── 八度保護 ──
    # YIN 取的是「第一個夠深的谷」,而泛音豐富的聲音在 2τ、3τ 也會有谷。
    # 如果 τ 的整數分之一同樣夠好,那才是真正的基頻週期 —— 取短的。
    for divisor in (4, 3, 2):
        candidate = tau // divisor
        if candidate < min_tau:
            continue
        if cumulative[candidate] < YIN_THRESHOLD * 1.6:
            tau = candidate
            break

    # 拋物線內插:取樣解析度不夠時,這一步決定精度
    if 1 <= tau < len(cumulative) - 1:
        a, b, c = cumulative[tau - 1], cumulative[tau], cumulative[tau + 1]
        denom = 2 * (2 * b - a - c)
        if abs(denom) > 1e-12:
            tau = tau + (c - a) / denom
    if tau <= 0:
        return 0.0

    freq = samplerate / tau

    # 第二道防線:跟上一次比。人聲不會在 25 ms 內跳一個八度,所以若
    # 「上一次的兩倍或一半」比這次更接近上一次,那多半是誤判。
    if previous > 0:
        for factor in (0.5, 2.0):
            shifted = freq * factor
            if MIN_HZ <= shifted <= MAX_HZ and \
                    abs(np.log2(shifted / previous)) < abs(np.log2(freq / previous)) - 0.5:
                freq = shifted
                break
    return float(freq)


# A4 = 440 Hz 是頻率基準,但 C 才是音階的習慣起點。root=0 要代表 C 大調,
# 不是 A 大調 —— 否則使用者選「C 大調」卻得到 A 大調,而 C 本身在 A 大調
# 裡正好卡在兩個音級中間,唱偏一點點就會被吸到不同的音。
A4_ABOVE_C = 9


def snap(freq: float, scale: str = "chromatic", root: int = 0,
         strength: float = 1.0) -> float:
    """把頻率吸到音階上。

    ``root`` 是主音,0 = C、2 = D、…(半音為單位)。
    ``strength`` 0 = 不動,1 = 完全吸到位。
    """
    if freq <= 0:
        return 0.0
    degrees = SCALES.get(scale) or SCALES["chromatic"]

    # 以 A4 = 440 Hz 為基準的半音數
    semis = 12.0 * np.log2(freq / 440.0)
    # 換算成「相對於主音的音級」,才知道落在音階的哪裡
    relative = semis + A4_ABOVE_C - root
    octave = np.floor(relative / 12.0)
    within = relative - octave * 12.0

    best = min(degrees, key=lambda d: min(abs(within - d), abs(within - d - 12)))
    if abs(within - best - 12) < abs(within - best):
        best += 12
    target = root - A4_ABOVE_C + octave * 12.0 + best

    corrected = semis + (target - semis) * max(0.0, min(1.0, strength))
    return float(440.0 * (2.0 ** (corrected / 12.0)))


def note_name(freq: float) -> str:
    if freq <= 0:
        return ""
    semis = int(round(12.0 * np.log2(freq / 440.0)))
    return f"{NOTE_NAMES[(semis + 9) % 12]}{4 + (semis + 9) // 12}"


class AutoTune:
    """逐 block 的即時音高修正。

    延遲固定為 ``latency_samples``(偵測窗 + 一個最長週期),與偵測到的
    音高無關 —— 延遲會隨音高變動的話,聲音會忽前忽後,比延遲本身更難聽。
    """

    def __init__(self, samplerate: int, channels: int = 1) -> None:
        self.samplerate = int(samplerate)
        self.channels = int(channels)

        self.enabled = False
        self.scale = "chromatic"
        self.root = 0                 # 0 = C
        self.strength = 1.0           # 吸附強度
        self.retune_ms = 20.0         # 吸過去的速度;越小越「電音」

        self._detect = int(samplerate * DETECT_MS / 1000.0)
        self._max_period = int(samplerate / MIN_HZ)
        # 延遲寫死成最長週期,不隨偵測結果變動
        self._latency = self._detect + self._max_period

        # 輸入歷史:要夠放偵測窗 + 顆粒半徑 + 一個 block 的餘裕
        self._history = self._latency + self._max_period + 4096
        self._in = np.zeros(self._history, dtype=np.float64)
        self._written = 0             # 已寫入的絕對取樣數

        # 輸出累加器:對應 [_out_start, _out_start + len)
        self._out = np.zeros(self._history, dtype=np.float64)
        self._weight = np.zeros(self._history, dtype=np.float64)
        self._out_start = 0
        self._emitted = 0             # 已送出的絕對取樣數

        self._pitch = 0.0             # 偵測到的
        self._current = 0.0           # 平滑後實際使用的目標
        self._next_mark = 0           # 下一個輸出顆粒的絕對位置
        self._voiced = False

    # ── 狀態 ────────────────────────────────────────────────────────────
    @property
    def latency_samples(self) -> int:
        return self._latency if self.enabled else 0

    @property
    def latency_ms(self) -> float:
        return self.latency_samples / self.samplerate * 1000.0

    @property
    def detected_hz(self) -> float:
        return self._pitch

    @property
    def target_hz(self) -> float:
        return self._current

    def as_dict(self) -> dict:
        return {
            "enabled": self.enabled,
            "scale": self.scale,
            "root": self.root,
            "strength": self.strength,
            "retune_ms": self.retune_ms,
            "latency_ms": round(self.latency_ms, 2),
            "detected_hz": round(self._pitch, 1),
            "detected_note": note_name(self._pitch),
            "target_hz": round(self._current, 1),
            "target_note": note_name(self._current),
            "voiced": self._voiced,
        }

    def reset(self) -> None:
        self._in.fill(0.0)
        self._out.fill(0.0)
        self._weight.fill(0.0)
        self._written = 0
        self._out_start = 0
        self._emitted = 0
        self._pitch = 0.0
        self._current = 0.0
        self._next_mark = 0
        self._voiced = False

    # ── 內部 ────────────────────────────────────────────────────────────
    def _read_in(self, start: int, length: int) -> np.ndarray:
        """從輸入歷史取 [start, start+length) 的絕對區間。"""
        index = np.arange(start, start + length) % self._history
        return self._in[index]

    def _find_mark(self, centre: int, period: int) -> int:
        """在 centre 附近找一個能量最大的位置當顆粒中心。

        只在 ±¼ 週期內找,間距才會鎖在週期附近 —— 放寬的話顆粒會以不規則
        的間隔重複,合出來的訊號多出長週期成分,聽起來就是低了一個八度。
        """
        span = max(2, period // 4)
        segment = np.abs(self._read_in(centre - span, 2 * span + 1))
        return centre - span + int(np.argmax(segment))

    # ── 處理 ────────────────────────────────────────────────────────────
    def process(self, block: np.ndarray) -> np.ndarray:
        frames = len(block)
        if frames == 0 or not self.enabled:
            return block

        mono = block[:, 0] if block.ndim > 1 else block

        # 1) 收下輸入
        index = np.arange(self._written, self._written + frames) % self._history
        self._in[index] = mono
        self._written += frames

        if self._written < self._latency + frames:
            # 還沒攢夠,先吐靜音把管線填起來
            return np.zeros_like(block)

        if self._emitted == 0:
            # 建立輸出對輸入的落後量。這件事只做一次 ——
            # 少了它,輸出位置會跟著輸入一起前進(等於零延遲),顆粒還沒
            # 寫到那個位置就被讀走,取出來的全是權重為零的空洞。
            self._emitted = self._written - self._latency - frames

        # 2) 偵測:看最近的一個偵測窗
        frame = self._read_in(self._written - self._detect, self._detect)
        detected = detect_pitch(frame, self.samplerate, self._pitch)
        self._voiced = detected > 0
        if detected > 0:
            self._pitch = detected
            wanted = snap(detected, self.scale, self.root, self.strength)
            # 平滑地滑向目標:瞬間吸到位就是經典的「電音」效果
            span = max(1.0, self.retune_ms) * self.samplerate / 1000.0
            step = min(1.0, frames / span)
            if self._current <= 0:
                self._current = wanted
            else:
                ratio = np.log2(wanted / self._current) * step
                self._current *= 2.0 ** ratio

        # 3) 產生顆粒,直到輸出蓋過我們要送出的區間
        need_until = self._emitted + frames
        if self._voiced and self._pitch > 0 and self._current > 0:
            period_in = self.samplerate / self._pitch
            period_out = self.samplerate / self._current
            radius = int(period_in)

            if self._next_mark < self._emitted:
                self._next_mark = self._emitted

            guard = self._written - self._detect - radius
            while self._next_mark < need_until + radius and self._next_mark < guard:
                source = self._find_mark(self._next_mark, int(period_in))
                grain = self._read_in(source - radius, 2 * radius)
                window = np.hanning(2 * radius)
                slot = np.arange(self._next_mark - radius,
                                 self._next_mark + radius) % self._history
                self._out[slot] += grain * window
                self._weight[slot] += window
                self._next_mark += max(1, int(round(period_out)))
        else:
            # 沒在唱(換氣、句子之間):原樣通過,不要亂修
            slot = np.arange(need_until - frames, need_until) % self._history
            self._out[slot] += self._read_in(self._emitted, frames)
            self._weight[slot] += 1.0

        # 4) 取出這一段輸出,並把用過的位置清乾淨
        slot = np.arange(self._emitted, self._emitted + frames) % self._history
        weight = self._weight[slot]
        result = np.where(weight > 1e-6, self._out[slot] / np.maximum(weight, 1e-6),
                          0.0)
        self._out[slot] = 0.0
        self._weight[slot] = 0.0
        self._emitted += frames

        out = result.astype(np.float32)
        if block.ndim > 1:
            return np.repeat(out[:, None], block.shape[1], axis=1)
        return out
