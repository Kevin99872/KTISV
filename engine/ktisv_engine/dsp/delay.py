"""可即時調整的延遲線。

用途是**對齊**,不是效果器 —— 網路那頭的聲音一定比本地晚,要讓兩邊聽起來
同步,唯一的辦法是把早到的那一路也拖慢。

改變延遲時為什麼用「滑行」而不是「淡接」
----------------------------------------
第一版是淡接:延遲一改,就在一個 block 內從舊的讀取位置淡到新的。問題是
淡接期間輸出等於「訊號 + 自己的時間位移版本」,那是梳狀濾波;而滑桿每
40 ms 送一次新值,一秒鐘就做二十幾次,聽起來是明顯的金屬味電子音。

實測(440 Hz 純音、延遲從 0 掃到 150 ms):淡接法的雜訊只比主音低 7–14 dB,
延遲不動時則是 −129 dB。也就是說電子音全部來自「調整」這個動作本身。

改成滑行之後,讀取位置是連續移動的,任何時刻都只讀一個點(用線性內插取
小數位置),輸出始終是一條連續波形。代價是移動期間音高會輕微晃動 ——
就是磁帶變速的感覺,對「把兩路對齊」這種用途完全可以接受,而且遠比梳狀
濾波自然。

滑行速率刻意有上下限:太慢會讓人以為沒反應,太快則音高晃動太誇張。跳太
遠(例如直接打一個差很多的數字)則不滑行,改用單次淡接一步到位 —— 那種
情況滑行要花好幾秒,反而更難用。
"""

from __future__ import annotations

import threading

import numpy as np

# 預設先配置的量。不是上限 —— 設定超過這個值時緩衝會自己長大,
# 這只是「一開始就配好、之後大多不用再長」的常用範圍。
INITIAL_DELAY_MS = 500.0

# 唯一剩下的天花板,而且它不是功能上的限制,是記憶體的保險絲。
#
# 延遲線必須先配置緩衝才能延遲,所以總得有個數字擋住手誤 —— 少打一個
# 小數點變成 600000 ms 的話,那是 230 GB,行程當場死掉。60 秒對「音樂
# 與歌聲對時」而言已經荒謬到不可能用到(實際用量是幾十到幾百毫秒),
# 代價則是每條線最多 60 s × 48 kHz × 2 ch × 4 B ≈ 23 MB。
CEILING_DELAY_MS = 60_000.0

# 滑行的目標時間:希望多久之內滑到新的延遲量。
GLIDE_SECONDS = 0.4

# 滑行速率的上下限(每輸出一個取樣,讀取位置移動幾個取樣)。
#
# 下限 0.10:再慢就會讓人覺得「調了沒反應」—— 實測 0.03 時,調 120 ms 要
# 花 1.4 秒才到位。
#
# 上限 0.25:代表播放速度暫時變成 0.75 倍。拖滑桿時每次只動一點點,根本
# 碰不到上限;只有「一次套用一個大數字」(例如校準結果)才會用到,那時
# 短暫的磁帶變速感是可以接受的,而且遠比接縫的電子音自然。
MIN_GLIDE_RATE = 0.10
MAX_GLIDE_RATE = 0.25

# 超過這個差距就不滑行了。以上限速率滑 300 ms 要一秒多,再遠下去等待感
# 會壓過音質好處 —— 那時單次淡接的一道接縫反而比長時間的音高怪叫好接受。
LARGE_JUMP_MS = 300.0


class DelayLine:
    """環形延遲。``delay_ms`` 可即時改,不會爆音也不會有梳狀濾波。

    緩衝會隨著設定值長大,所以呼叫端不必事先知道會用到多少延遲。
    """

    def __init__(self, samplerate: int, channels: int,
                 initial_ms: float = INITIAL_DELAY_MS) -> None:
        self.samplerate = int(samplerate)
        self.channels = int(channels)
        self.max_samples = int(self.samplerate * initial_ms / 1000.0)

        # 緩衝要比最大延遲再大一個寬裕的 block,否則延遲拉到最大時,
        # 讀取範圍會跟這次剛寫進去的資料重疊,讀到的是未來的內容。
        self._capacity = self.max_samples + 4096
        self._buf = np.zeros((self._capacity, self.channels), dtype=np.float32)
        self._write = 0
        self._delay = 0.0      # 目前生效的延遲(取樣,可為小數)
        self._target = 0       # 使用者設定的延遲,滑行的目標
        self._large_jump = int(self.samplerate * LARGE_JUMP_MS / 1000.0)

        # 長大時會整個換掉 _buf / _capacity / _write,而音訊回呼同時在讀
        # 這三個。只換其中一個就被讀到的話會索引越界,所以兩邊都要上鎖。
        # 成長本身很罕見(只有使用者輸入更大的值時),平常只是一次
        # 無爭用的 acquire —— 同層的 RingBuffer 也是這樣做的。
        self._lock = threading.Lock()

    # ── 參數 ────────────────────────────────────────────────────────────
    @property
    def delay_ms(self) -> float:
        return self._target / self.samplerate * 1000.0

    @delay_ms.setter
    def delay_ms(self, value: float) -> None:
        value = float(value)
        if value != value:                     # NaN
            value = 0.0
        value = max(0.0, min(CEILING_DELAY_MS, value))
        samples = int(round(value * self.samplerate / 1000.0))
        with self._lock:
            if samples > self.max_samples:
                self._grow(samples)
            self._target = samples

    @property
    def delay_samples(self) -> int:
        return self._target

    @property
    def settled(self) -> bool:
        """已經滑到定位了沒有。測試與診斷用。"""
        return abs(self._target - self._delay) < 1e-6

    @property
    def active(self) -> bool:
        """完全沒有延遲時可以整條略過,省掉一次搬移。"""
        return self._target > 0 or self._delay > 0.0

    def reset(self) -> None:
        with self._lock:
            self._buf.fill(0.0)
            self._write = 0
            self._delay = float(self._target)

    def _grow(self, needed: int) -> None:
        """把緩衝換成裝得下 ``needed`` 的新緩衝,並保住既有的歷史音訊。

        呼叫者必須已經持有 ``_lock``。歷史要照時間順序搬過去,否則長大的
        那一刻正在讀的那段延遲音訊會錯位、發出雜音。
        """
        capacity = needed + 4096
        history = self._linearized()
        buf = np.zeros((capacity, self.channels), dtype=np.float32)
        keep = min(len(history), capacity)
        if keep:
            buf[:keep] = history[-keep:]
        self._buf = buf
        self._capacity = capacity
        self._write = keep % capacity
        self.max_samples = needed

    def _linearized(self) -> np.ndarray:
        """目前的緩衝內容,由最舊到最新排好。"""
        return np.concatenate([self._buf[self._write:], self._buf[:self._write]])

    # ── 處理 ────────────────────────────────────────────────────────────
    #
    # 環形緩衝的存取分成「不跨界」與「跨界」兩條路。不跨界時用切片,
    # numpy 直接記憶體搬移;跨界才拆成兩段。比起無條件用 index 陣列取值,
    # 少掉每個 block 配置一次索引陣列的成本 —— block 只有 64 取樣時,
    # 這種固定成本佔的比例並不小。
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

    def _read(self, delay: int, frames: int) -> np.ndarray:
        """取出「結束於目前寫入位置、往前推 delay」的那一段(整數延遲)。"""
        start = (self._write - frames - delay) % self._capacity
        end = start + frames
        if end <= self._capacity:
            return self._buf[start:end].copy()
        first = self._capacity - start
        out = np.empty((frames, self.channels), dtype=np.float32)
        out[:first] = self._buf[start:]
        out[first:] = self._buf[:frames - first]
        return out

    def _read_gliding(self, start_delay: float, end_delay: float,
                      frames: int) -> np.ndarray:
        """讀一段延遲量從 start 平滑走到 end 的音訊。

        每個輸出取樣各自算自己的讀取位置,落在取樣之間就線性內插。整段
        始終只讀一個點,不會出現「訊號 + 位移版自己」那種梳狀濾波。
        """
        delays = np.linspace(start_delay, end_delay, frames,
                             endpoint=False, dtype=np.float64)
        # self._write 已經跨過這個 block,所以本 block 第一個取樣位在
        # write - frames;第 j 個輸出取樣要讀的就是它往前推 delays[j]。
        positions = (self._write - frames) + np.arange(frames) - delays

        lower = np.floor(positions).astype(np.int64)
        frac = (positions - lower).astype(np.float32)[:, np.newaxis]
        a = self._buf[lower % self._capacity]
        b = self._buf[(lower + 1) % self._capacity]
        return (a * (1.0 - frac) + b * frac).astype(np.float32)

    def process(self, block: np.ndarray) -> np.ndarray:
        frames = len(block)
        if frames == 0:
            return block
        if block.shape[1] != self.channels:
            raise ValueError(
                f"延遲線是 {self.channels} 聲道,收到 {block.shape[1]} 聲道")

        with self._lock:
            # 就算目前沒有延遲也照樣寫入。這樣使用者把延遲從 0 拉起來時,
            # 緩衝裡已經有真正的歷史音訊可以讀 —— 否則會先聽到一段空白,
            # 長度正好等於他剛設定的延遲量。
            self._write_block(block)

            target = float(self._target)
            current = self._delay
            diff = target - current

            if target == 0.0 and current == 0.0:
                return block

            if abs(diff) < 1e-6:
                # 穩態:延遲是整數且不動,走最省的切片路徑
                return self._read(int(round(current)), frames)

            if abs(diff) > self._large_jump:
                # 差太遠,滑行要花好幾秒。改成一次淡接過去 —— 只有一道
                # 接縫,比長時間的音高怪叫好接受。
                old = self._read(int(round(current)), frames)
                new = self._read(self._target, frames)
                ramp = np.linspace(0.0, 1.0, frames,
                                   dtype=np.float32)[:, np.newaxis]
                self._delay = target
                return old * (1.0 - ramp) + new * ramp

            # 平滑滑行:速率依距離調整,但夾在上下限之間
            rate = abs(diff) / (GLIDE_SECONDS * self.samplerate)
            rate = min(MAX_GLIDE_RATE, max(MIN_GLIDE_RATE, rate))
            step = min(abs(diff), rate * frames)
            end = current + (step if diff > 0 else -step)

            out = self._read_gliding(current, end, frames)
            self._delay = end
            return out
