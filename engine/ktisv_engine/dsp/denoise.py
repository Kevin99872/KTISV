"""麥克風底噪處理 —— 電源哼聲與嘶聲。

「電流聲」通常是兩種東西混在一起,要用不同手段對付:

* **哼聲**:50/60 Hz 的電源頻率與其諧波,來自接地迴路或劣質供電。它是
  窄頻、固定頻率的,用陷波濾波器切掉最乾淨,對人聲幾乎沒有影響。
* **嘶聲**:前級放大器的寬頻白噪。它跟人聲頻譜重疊,切不掉,只能在
  「你沒出聲的時候」把它壓下去 —— 也就是向下擴展(閘門)。

為什麼不用頻譜相減
------------------
頻譜相減(或 Wiener 濾波)能在你講話的同時也壓掉嘶聲,效果更好。但那要
STFT,而 512 點窗在 48 kHz 下就是 8 ms 以上的延遲 —— 這個專案花了很大
力氣才把耳返壓到 10 ms 出頭,為了降噪再吐回去 8 ms 並不划算。這裡的做法
全部是零延遲的:濾波器是 IIR、閘門是逐 block 算增益。

門檻怎麼來
----------
不要求使用者自己找數字:持續追蹤「最近聽過的最小音量」當底噪,門檻設在
它上面幾 dB。這樣換麥克風、換房間都會自動跟上。追蹤用的是慢速最小值
追隨器 —— 掉下去追得快、爬上來走得慢,才不會被一句長音拉高門檻。
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, iirnotch, sosfilt, tf2sos

# 哼聲的諧波要切到第幾階。切太多階會開始吃到人聲基頻(男聲約 85–180 Hz),
# 4 階在 50 Hz 下是 200 Hz —— 已經夠涵蓋實務上聽得到的哼聲。
HUM_HARMONICS = 4

# 陷波的 Q。越高越窄、對人聲影響越小,但對頻率飄移越不寬容;
# 市電頻率其實相當穩,所以可以用得很窄。
HUM_Q = 30.0

# ── 自動找出「異常頻率」 ────────────────────────────────────────────────
#
# 手動選 50/60 Hz 有兩個問題。第一,使用者不知道該選哪個。第二更要命:
# 真正吵的往往不在基頻。實測這台機器的麥克風,60 Hz 比周邊底噪還**低**
# 3 dB,能量全部集中在 120 Hz(高出 15~17 dB)—— 那是全波整流的電源
# 漣波,出現在市電頻率的**兩倍**。照「我在台灣所以選 60」的直覺,
# 切掉的是一個根本沒有能量的地方。
#
# 所以不要猜,直接量:找出頻譜上真的突起的窄峰,陷波就下在那裡。
AUTO_BAND = (40.0, 1000.0)   # 只在這個範圍找。哼聲諧波到 1 kHz 以上
                             # 已經沒有實際能量,再往上只會誤傷人聲
AUTO_FFT = 8192              # 解析度 5.9 Hz @ 48k,足以分開相鄰諧波
AUTO_MAX_PEAKS = 4           # 最多同時切幾個 —— 切太多會開始吃到人聲
AUTO_EXCESS_DB = 8.0         # 高出周邊底噪多少才算「異常」
AUTO_RETUNE_HZ = 3.0         # 偵測結果移動超過這個才重建濾波器
AUTO_MIN_INTERVAL = 2.0      # 兩次重建至少間隔幾秒(重建會有極輕微不連續)

# 自動陷波的 Q。比手動的 HUM_Q(30)寬一點。
#
# 8192 點的 FFT 在 48k 下每格 5.9 Hz,就算做了拋物線內插,峰值頻率的
# 誤差仍有一兩 Hz。Q=30 在 120 Hz 只有 4 Hz 寬 —— 比誤差還窄,常常整個
# 錯過(實測只壓掉 3 dB)。Q=20 是 6 Hz,容得下誤差又不會寬到吃人聲。
AUTO_Q = 20.0

# 只在音量低於這個絕對值時才學。
#
# 不能只靠「低於自適應底噪 + margin」:底噪追蹤器會被持續的長音慢慢
# 拉高,唱久了門檻就漂到人聲之上,於是把歌手的長音當成安靜來學 ——
# 實測唱 123 Hz(男聲 B2)六秒,它真的把 123/246/369 全部當成電流聲。
#
# 電流聲的絕對音量很低(實測這台機器 -77 dBFS),唱歌是 -20 dBFS 量級,
# 中間差了五十幾 dB,拿一個絕對門檻就切得乾淨,而且不會隨時間漂。
AUTO_MAX_LEVEL_DB = -45.0

# 要平均幾個分析窗再找峰。
#
# 不能拿單一個窗的頻譜直接找 —— 噪音本身的隨機起伏會製造一堆假峰,
# 每次的位置和數量都不一樣。實測單窗偵測會給出 [117, 246, 258, 275],
# 其中只有第一個是真的。
#
# 一開始試過「連續兩次偵測到同一組峰才採用」,結果是永遠湊不齊:假峰
# 每次都不同,數量一變就重新計數,於是一個都採用不了。改成先平均頻譜:
# 穩定的哼聲每個窗都在同一格,會被加強;假峰隨機分佈,會被平均掉。
#
# 8 個窗 = 1.37 秒的安靜。使用者不會察覺這段學習時間。
AUTO_AVG_WINDOWS = 8


def _epsilon_db(x: float) -> float:
    return 20.0 * np.log10(max(x, 1e-9))


class MicDenoiser:
    """高通 + 哼聲陷波 + 向下擴展閘門。全部零延遲。"""

    def __init__(self, samplerate: int, channels: int = 1) -> None:
        self.samplerate = int(samplerate)
        self.channels = int(channels)

        self.enabled = False
        self._highpass_hz = 80.0
        self._hum_hz = 0.0            # 0 = 不切哼聲;常用 50 或 60
        self._sos: np.ndarray | None = None
        self._zi: np.ndarray | None = None

        # 閘門
        self.gate_enabled = True
        self._margin_db = 12.0        # 門檻設在底噪之上多少 dB
        self._reduction_db = 18.0     # 最多壓多少
        self._ratio = 3.0
        self.attack_ms = 5.0
        self.release_ms = 120.0

        self._floor_db = -60.0
        self._gain = 1.0
        self._learning = 0            # 還要學幾個 block 的底噪

        # 自動找哼聲。開了之後 hum_hz 就不管用 —— 陷波位置由量測決定。
        self.auto_hum = False
        self._auto_freqs: list[float] = []
        self._auto_buf = np.zeros(AUTO_FFT, dtype=np.float64)
        self._auto_fill = 0
        self._auto_cooldown = 0       # 還要等幾個取樣才允許再重建
        self._auto_spec = None            # 累積中的功率頻譜
        self._auto_windows = 0            # 已累積幾個分析窗

        self._rebuild()

    # ── 參數 ────────────────────────────────────────────────────────────
    @property
    def highpass_hz(self) -> float:
        return self._highpass_hz

    @highpass_hz.setter
    def highpass_hz(self, value: float) -> None:
        value = max(0.0, min(400.0, float(value)))
        if abs(value - self._highpass_hz) > 1e-6:
            self._highpass_hz = value
            self._rebuild()

    @property
    def hum_hz(self) -> float:
        return self._hum_hz

    @hum_hz.setter
    def hum_hz(self, value: float) -> None:
        value = float(value)
        # 只接受 0(關閉)與市電頻率。其他數字多半是誤填,而在人聲頻段
        # 亂放窄陷波會把聲音挖出洞。
        if value not in (0.0, 50.0, 60.0):
            value = 0.0
        if abs(value - self._hum_hz) > 1e-6:
            self._hum_hz = value
            self._rebuild()

    @property
    def margin_db(self) -> float:
        return self._margin_db

    @margin_db.setter
    def margin_db(self, value: float) -> None:
        self._margin_db = max(0.0, min(40.0, float(value)))

    @property
    def reduction_db(self) -> float:
        return self._reduction_db

    @reduction_db.setter
    def reduction_db(self, value: float) -> None:
        self._reduction_db = max(0.0, min(60.0, float(value)))

    @property
    def threshold_db(self) -> float:
        return self._floor_db + self._margin_db

    @property
    def floor_db(self) -> float:
        return self._floor_db

    def as_dict(self) -> dict:
        return {
            "enabled": self.enabled,
            "highpass_hz": self._highpass_hz,
            "hum_hz": self._hum_hz,
            "auto_hum": self.auto_hum,
            "auto_freqs": self.auto_freqs,
            "gate_enabled": self.gate_enabled,
            "margin_db": self._margin_db,
            "reduction_db": self._reduction_db,
            "floor_db": round(self._floor_db, 1),
            "threshold_db": round(self.threshold_db, 1),
        }

    # ── 內部 ────────────────────────────────────────────────────────────
    def _rebuild(self) -> None:
        """把高通與所有陷波串成一組 SOS。

        濾波器狀態要保留 —— 直接丟掉會在改參數的瞬間發出一聲爆音,而使用者
        調降噪的時候正好在聽自己的聲音。
        """
        sections = []
        if self._highpass_hz > 0:
            sections.append(butter(2, self._highpass_hz, "highpass",
                                   fs=self.samplerate, output="sos"))

        # 自動模式優先:量到什麼就切什麼。量不到就不切 —— 沒有偵測到
        # 突起卻硬切,只會平白在頻譜上挖洞。
        if self.auto_hum:
            notch_freqs = list(self._auto_freqs)
        elif self._hum_hz > 0:
            notch_freqs = [self._hum_hz * order
                           for order in range(1, HUM_HARMONICS + 1)]
        else:
            notch_freqs = []

        q = AUTO_Q if self.auto_hum else HUM_Q
        for freq in notch_freqs:
            if freq <= 0 or freq >= self.samplerate * 0.45:
                continue
            b, a = iirnotch(freq, q, fs=self.samplerate)
            sections.append(tf2sos(b, a))

        if not sections:
            self._sos = None
            self._zi = None
            return

        self._sos = np.concatenate(sections, axis=0)
        self._zi = self._fresh_state()

    def _fresh_state(self) -> np.ndarray | None:
        """濾波器的初始狀態,一律從靜止開始。

        不能用 sosfilt_zi():那回傳的是「輸入一直是 1.0」的穩態,拿來當
        音訊串流的起點等於在第一個 block 灌進一個大暫態 —— 實測會讓安靜
        段落反而比處理前大 3.5 dB。音訊從靜音開始,狀態就該是零。
        """
        if self._sos is None:
            return None
        return np.zeros((self._sos.shape[0], 2, self.channels), dtype=np.float64)

    def reset(self) -> None:
        self._gain = 1.0
        self._zi = self._fresh_state()
        # 偵測結果不清掉 —— 換一首歌不代表電源環境變了,重學只是白白
        # 讓電流聲再響幾秒。真的要重學就把 auto_hum 關掉再打開。
        self._auto_fill = 0
        self._auto_cooldown = 0

    def learn_floor(self, seconds: float = 1.0) -> None:
        """重新學底噪:接下來這段時間收到的音量直接當底噪。"""
        self._learning = max(1, int(seconds * self.samplerate / 256))
        self._floor_db = 0.0

    # ── 自動找異常頻率 ──────────────────────────────────────────────────
    def _detect_peaks(self, mean_spec: np.ndarray) -> list[float]:
        """從平均後的功率頻譜找出高出周邊底噪 AUTO_EXCESS_DB 的窄峰。"""
        n = AUTO_FFT
        psd = 10.0 * np.log10(mean_spec + 1e-20)
        freqs = np.fft.rfftfreq(n, 1.0 / self.samplerate)

        lo, hi = AUTO_BAND
        band = np.where((freqs >= lo) & (freqs <= hi))[0]
        if len(band) < 8:
            return []

        # 「周邊底噪」要從**偏開一段距離**的旁瓣估,不能用以自己為中心的
        # 中位數視窗。
        #
        # 實測踩過這個坑:這台機器的電流聲不是一根窄線,而是 90–125 Hz
        # 一整片約 30 Hz 寬的隆起。以中心 ±59 Hz 取中位數的話,視窗整個
        # 落在隆起裡面,基線跟著被抬高,算出來的突起只剩 4.6 dB ——
        # 低於門檻,於是最吵的那一塊完全偵測不到。
        #
        # 改成只取左右各偏開 30–90 Hz 的區間當基線,視窗就落在隆起外面,
        # 同一份訊號量到的突起是 14.9 dB。
        bin_hz = self.samplerate / n
        off = max(2, int(round(30.0 / bin_hz)))
        baseline = np.empty(len(band))
        for i, k in enumerate(band):
            side = np.concatenate([psd[max(0, k - 3 * off):max(0, k - off)],
                                   psd[k + off:k + 3 * off]])
            baseline[i] = np.median(side) if len(side) else psd[k]
        excess = psd[band] - baseline

        found: list[tuple[float, float]] = []
        for i in range(1, len(band) - 1):
            if excess[i] < AUTO_EXCESS_DB:
                continue
            # 只收局部極大值,否則同一個峰的裙擺會被收好幾次
            if psd[band[i]] < psd[band[i - 1]] or psd[band[i]] < psd[band[i + 1]]:
                continue
            # 拋物線內插取次格精度。峰值幾乎不會剛好落在格線上,直接用
            # 格心的話誤差可達半格(2.9 Hz),陷波就會偏掉。
            k = band[i]
            a0, a1, a2 = psd[k - 1], psd[k], psd[k + 1]
            denom = a0 - 2.0 * a1 + a2
            delta = 0.0 if abs(denom) < 1e-12 else 0.5 * (a0 - a2) / denom
            delta = max(-0.5, min(0.5, float(delta)))
            bin_hz = self.samplerate / n
            found.append((float(excess[i]), float(freqs[k] + delta * bin_hz)))

        found.sort(reverse=True)          # 突起最多的優先
        picked: list[float] = []
        for _, f in found:
            # 靠太近的視為同一個峰
            if any(abs(f - p) < 2 * AUTO_RETUNE_HZ for p in picked):
                continue
            picked.append(f)
            if len(picked) >= AUTO_MAX_PEAKS:
                break
        return sorted(picked)

    def _feed_auto(self, mono: np.ndarray) -> None:
        """累積分析用的樣本,滿了就重新偵測。

        **只在安靜時餵。** 這是這整套能安全運作的關鍵:唱歌時的持續音
        在頻譜上跟哼聲長得一模一樣(都是窄峰),分不出來。只在閘門判定
        「沒人在唱」的時候學,就不會把歌手的長音當成電流聲切掉。
        """
        need = AUTO_FFT - self._auto_fill
        take = min(need, len(mono))
        self._auto_buf[self._auto_fill:self._auto_fill + take] = mono[:take]
        self._auto_fill += take
        if self._auto_fill < AUTO_FFT:
            return

        self._auto_fill = 0
        if self._auto_cooldown > 0:
            self._auto_cooldown -= AUTO_FFT
            return

        # 累積功率頻譜。平均而不是逐窗判斷 —— 見 AUTO_AVG_WINDOWS 的說明。
        spec = np.abs(np.fft.rfft(self._auto_buf * np.hanning(AUTO_FFT))) ** 2
        self._auto_spec = spec if self._auto_spec is None else self._auto_spec + spec
        self._auto_windows += 1
        if self._auto_windows < AUTO_AVG_WINDOWS:
            return

        peaks = self._detect_peaks(self._auto_spec / self._auto_windows)
        self._auto_spec = None
        self._auto_windows = 0

        if self._changed(peaks, self._auto_freqs):
            self._auto_freqs = peaks
            self._auto_cooldown = int(AUTO_MIN_INTERVAL * self.samplerate)
            self._rebuild()

    @staticmethod
    def _changed(peaks: list[float], reference: list[float]) -> bool:
        """兩組頻率是否差得夠多(重建濾波器有極輕微不連續,不值得為小變動做)。"""
        if len(peaks) != len(reference):
            return True
        return any(abs(a - b) > AUTO_RETUNE_HZ
                   for a, b in zip(peaks, reference))

    @property
    def auto_freqs(self) -> list[float]:
        """目前自動偵測到、正在被切掉的頻率。"""
        return [round(f, 1) for f in self._auto_freqs]

    # ── 處理 ────────────────────────────────────────────────────────────
    def process(self, block: np.ndarray) -> np.ndarray:
        if not self.enabled or len(block) == 0:
            return block

        frames = len(block)
        out = block

        # 偵測要看**還沒被陷波處理過**的訊號。看處理後的話,峰一被切掉就
        # 偵測不到,下一輪判定「沒有異常頻率」而把陷波拿掉,峰又回來 ——
        # 會在「切掉/不切」之間來回震盪。
        raw = np.asarray(block[:, 0] if block.ndim > 1 else block,
                         dtype=np.float64) if self.auto_hum else None

        if self._sos is not None:
            out, self._zi = sosfilt(self._sos, out, axis=0, zi=self._zi)
            out = out.astype(np.float32, copy=False)

        level = float(np.sqrt(np.mean(np.square(out)))) if frames else 0.0
        level_db = _epsilon_db(level)

        # 底噪追蹤:掉下去追得快、爬上來走得慢。反過來的話,一句長音就會
        # 把底噪拉高,門檻跟著漂,閘門開始咬掉句尾。
        if self._learning > 0:
            self._floor_db = level_db if self._floor_db == 0.0 \
                else min(self._floor_db, level_db)
            self._learning -= 1
        elif level_db < self._floor_db:
            self._floor_db += (level_db - self._floor_db) * 0.25
        else:
            self._floor_db += (level_db - self._floor_db) * 0.0004
        self._floor_db = max(-90.0, min(0.0, self._floor_db))

        # 只在安靜時學。唱歌時的持續音在頻譜上跟哼聲一樣是窄峰,分不出來,
        # 所以一有人聲就把累積到一半的分析緩衝丟掉重來 —— 寧可學得慢,
        # 也不要把歌手的長音誤判成電流聲切掉。
        if self.auto_hum and raw is not None:
            # 兩道關卡都要過:絕對音量夠低(不會隨底噪追蹤器漂),
            # 而且相對於當下底噪也算安靜。
            quiet = (level_db < AUTO_MAX_LEVEL_DB
                     and level_db < self._floor_db + self._margin_db)
            if quiet:
                self._feed_auto(raw)
            else:
                self._auto_fill = 0
                self._auto_spec = None
                self._auto_windows = 0

        if not self.gate_enabled:
            return out

        threshold = self.threshold_db
        if level_db >= threshold:
            target_db = 0.0
        else:
            target_db = max(-self._reduction_db,
                            (level_db - threshold) * (self._ratio - 1.0))
        target = float(10.0 ** (target_db / 20.0))

        # 逐 block 決定增益,block 之內線性內插到位 —— 直接換值會有階梯,
        # 在安靜段落聽起來就是一格一格的雜音。
        block_ms = frames / self.samplerate * 1000.0
        span = self.attack_ms if target < self._gain else self.release_ms
        step = min(1.0, block_ms / max(span, 1e-3))
        new_gain = self._gain + (target - self._gain) * step

        ramp = np.linspace(self._gain, new_gain, frames,
                           dtype=np.float32)[:, np.newaxis]
        self._gain = new_gain
        return (out * ramp).astype(np.float32)
