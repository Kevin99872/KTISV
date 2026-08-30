"""跨時脈的漂移補償重取樣。

問題
----
虛擬音效卡那一路的資料是由**耳機的回呼**產生的,卻由**虛擬卡自己的回呼**
取用。兩個裝置各有各的時脈,標稱都是 48 kHz,實際上差幾十 ppm。

原本的做法是用環形緩衝吸收,存量太多就丟、太少就補零。那只處理得了「累積
起來的差」,處理不了「速率本身就不一樣」這件事 —— 送進虛擬卡的取樣率跟它
自己的時脈對不上,驅動只好自己插補或丟棄來湊,而那些微小的不連續聽起來就是
寬頻的沙沙聲。

實測(注入 440 Hz 純音,從 CABLE Output 錄回,量 ±400 Hz 窗外的能量):

    虛擬卡自己當主時脈      -100.0 dB     (同一個時脈,沒有問題)
    耳機共享當主時脈         -96.5 dB     (同屬 Windows 混音引擎的時脈)
    耳機獨佔當主時脈         -40.5 dB     (獨立硬體時脈,差最多)

做法
----
不要丟樣本,改成**用略微不同的速率把它讀出來**。讀取位置是小數,每個輸出
取樣往前推 ``ratio`` 個輸入取樣,落在取樣之間就內插。``ratio`` 由存量的
誤差回授決定:存量偏高就讀快一點、偏低就讀慢一點。

回授刻意壓得很慢很鈍。實際漂移只有幾十 ppm,ratio 永遠在 1.0 附近幾個
萬分點內 —— 那個程度的音高變化(遠小於一音分)聽不出來。反過來說,如果
讓回授反應太快,ratio 會跟著存量的瞬時抖動亂跑,那才會產生聽得見的抖晃。
"""

from __future__ import annotations

import threading

import numpy as np

# ratio 允許偏離 1.0 多少。實際時脈差是幾十 ppm(1e-5 量級),萬分之五
# 已經是它的幾十倍餘裕;再放寬只會讓失控時的音高偏移變得聽得見。
MAX_RATIO_DEVIATION = 5e-4

# 回授增益。環路的時間常數 = target / (samplerate x GAIN),
# 2e-3 在 target = 4 個 block 時約 5 秒 —— 足以追上開機後的時脈差,
# 又不會快到把存量的瞬時抖動放大成聽得見的音高晃動。
#
# 這裡調過兩次:2e-5 太小(十秒只移動 6.5 ppm,追不上),4e-4 仍然要
# 27 秒才收斂。真正讓高增益可行的是下面那個存量平滑 —— 沒有它的話,
# 存量本身有 ±1 個 block 的抖動,乘上這個增益會變成幾百 ppm 的亂跳。
FEEDBACK_GAIN = 2e-3

# 存量的平滑係數。回授吃的是平滑後的存量,不是瞬時值。
FILL_SMOOTHING = 0.002

# ratio 自己再加一層低通,確保它只會慢慢爬。
RATIO_SMOOTHING = 0.01

# 粗鉗制:存量超過目標的幾倍就直接倒掉多的。
#
# 重取樣環路只補得了「速率的細微差異」(幾十 ppm),補不了「一次多出
# 一大塊」。啟動時耳機串流比虛擬卡早開始產生資料,存量會一口氣衝到上百
# 毫秒 —— 那要用 500 ppm 慢慢排要四分鐘,期間對方聽到的都是舊音訊。
# 所以保留一道粗鉗制:它在穩態下永遠不會觸發(存量穩在目標附近),
# 只在啟動與取用端停擺這類異常時把場面收拾乾淨。倍數不能訂太緊 ——
# 存量本來就會隨著取用端的成串呼叫上下擺盪,鉗得太緊會把正常的
# 高點也砍掉,反而讓下一串呼叫讀到見底。
MAX_FILL_MULTIPLE = 5.0


class DriftCorrector:
    """單生產者 / 單消費者的跨時脈緩衝,用重取樣補速率差。

    ``write()`` 由產生端(耳機回呼)呼叫,``read()`` 由取用端(虛擬卡回呼)
    呼叫,兩者在不同執行緒。
    """

    def __init__(self, samplerate: int, channels: int,
                 target_fill: int, capacity: int | None = None) -> None:
        self.samplerate = int(samplerate)
        self.channels = int(channels)
        self.target_fill = max(1, int(target_fill))

        # 容量要遠大於目標存量:啟動、暫時卡頓都會讓存量衝高,容量不夠就
        # 會覆蓋掉還沒讀的資料。
        self._capacity = int(capacity or max(samplerate // 2,
                                             self.target_fill * 8))
        self._buf = np.zeros((self._capacity, self.channels), dtype=np.float32)
        self._write = 0
        self._read = 0.0            # 讀取位置,小數
        self._filled = 0.0          # 尚未讀出的量(小數)

        self._ratio = 1.0
        self._fill_avg = float(self.target_fill)
        self.underflows = 0
        self.overflows = 0
        self._lock = threading.Lock()

    # ── 狀態 ────────────────────────────────────────────────────────────
    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def ratio(self) -> float:
        """目前的讀取速率倍數。1.0 = 兩邊時脈一致。"""
        return self._ratio

    def available(self) -> int:
        with self._lock:
            return int(self._filled)

    def prime(self) -> None:
        """清空並預先填入一個目標存量的靜音。

        啟動時產生端(耳機回呼)會比取用端(虛擬卡回呼)早跑,不預填的話
        存量會先衝高,而重取樣環路每秒只排得掉二十幾個取樣 —— 實機量到
        對方的延遲因此卡在 130 ms 好幾分鐘。預填之後存量從一開始就在
        目標上,環路只需要處理真正的時脈漂移。
        """
        self.clear()
        with self._lock:
            self._filled = float(self.target_fill)
            self._write = self.target_fill % self._capacity

    def clear(self) -> None:
        with self._lock:
            self._buf.fill(0.0)
            self._write = 0
            self._read = 0.0
            self._filled = 0.0
            self._ratio = 1.0
            self._fill_avg = float(self.target_fill)

    # ── 寫入 ────────────────────────────────────────────────────────────
    def write(self, data: np.ndarray) -> int:
        frames = len(data)
        if frames == 0:
            return 0

        with self._lock:
            if frames >= self._capacity:
                data = data[-self._capacity:]
                frames = len(data)
                self.overflows += 1

            free = self._capacity - self._filled
            if frames > free:
                # 真的滿了才丟。正常運作下不該發生 —— 會走到這裡代表取用端
                # 停了(裝置被拔掉、系統卡住),那時保住最新的資料比較有用。
                drop = frames - free
                self._read = (self._read + drop) % self._capacity
                self._filled -= drop
                self.overflows += 1

            # 粗鉗制:一次多出一大塊(啟動、取用端停擺)就倒掉,只留目標量。
            # 這是不連續,但發生在音訊還沒有意義的時候,而讓它留著的代價是
            # 對方會聽到延遲上百毫秒的舊音訊。
            limit = self.target_fill * MAX_FILL_MULTIPLE
            if self._filled > limit:
                excess = int(self._filled - self.target_fill)
                self._read = (self._read + excess) % self._capacity
                self._filled -= excess
                self._fill_avg = float(self.target_fill)

            end = self._write + frames
            if end <= self._capacity:
                self._buf[self._write:end] = data
            else:
                first = self._capacity - self._write
                self._buf[self._write:] = data[:first]
                self._buf[:frames - first] = data[first:]
            self._write = end % self._capacity
            self._filled += frames
            return frames

    # ── 讀出 ────────────────────────────────────────────────────────────
    def _update_ratio(self) -> None:
        """依存量誤差微調讀取速率。呼叫者必須持有 ``_lock``。

        **在讀取之後**呼叫。寫入端與讀取端在同一個 block 內一寫一讀,
        若在讀之前取存量,量到的永遠多一個 block —— 那會讓設定點固定
        偏移一整個 block,環路乖乖地把它補掉,存量就穩在錯的地方。
        """
        self._fill_avg += (self._filled - self._fill_avg) * FILL_SMOOTHING
        error = self._fill_avg - self.target_fill
        wanted = 1.0 + FEEDBACK_GAIN * (error / self.target_fill)
        wanted = min(1.0 + MAX_RATIO_DEVIATION,
                     max(1.0 - MAX_RATIO_DEVIATION, wanted))
        self._ratio += (wanted - self._ratio) * RATIO_SMOOTHING

    def read(self, frames: int) -> np.ndarray:
        if frames <= 0:
            return np.zeros((0, self.channels), dtype=np.float32)

        with self._lock:
            needed = frames * self._ratio

            if self._filled < needed + 2:
                # 資料不夠。補零並記一筆 —— 這在穩態下不該發生,發生就表示
                # 目標存量給得太小、或產生端整個停住了。
                self.underflows += 1
                out = np.zeros((frames, self.channels), dtype=np.float32)
                take = int(max(0.0, self._filled - 2))
                if take > 0:
                    partial = self._interpolate(take, 1.0)
                    out[:min(take, frames)] = partial[:min(take, frames)]
                    self._filled -= take        # 讀了多少就要扣多少,否則
                self._update_ratio()            # 存量與讀取位置會脫節
                return out

            out = self._interpolate(frames, self._ratio)
            self._filled -= needed
            self._update_ratio()
            return out

    def _interpolate(self, frames: int, ratio: float) -> np.ndarray:
        """從 ``_read`` 開始,以 ratio 的步進取 frames 個取樣(線性內插)。"""
        offsets = self._read + np.arange(frames, dtype=np.float64) * ratio
        lower = np.floor(offsets).astype(np.int64)
        frac = (offsets - lower).astype(np.float32)[:, np.newaxis]

        a = self._buf[lower % self._capacity]
        b = self._buf[(lower + 1) % self._capacity]
        out = (a * (1.0 - frac) + b * frac).astype(np.float32)

        self._read = (self._read + frames * ratio) % self._capacity
        return out
