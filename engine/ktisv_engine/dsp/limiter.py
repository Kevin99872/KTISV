"""前瞻峰值限幅器。

為什麼要換掉原本的軟限幅
------------------------
原本用的是 tanh 波形整形:超過 0.7 就把波形壓彎。那有兩個問題:

1. **門檻太低。** 0.7 是 −3 dBFS,而預設的 master 推桿就在 0.8 —— 播一首
   正常響度的歌,訊號一直待在失真區裡。實測 THD+N:−3 dBFS 以下是
   −109.7 dB(等於完全透明),到 −1.9 dBFS 掉到 −55.7 dB,0 dBFS 只剩
   −30.9 dB(約 2.8% 失真,聽得出來)。

2. **它改的是波形,不是音量。** 波形整形會生出諧波,高頻內容的諧波超過
   奈奎斯特之後還會混疊回來變成不諧和的雜音。降增益不會有這個問題 ——
   波形形狀不變,只是變小聲。

這裡怎麼做
----------
每個 block 先看「下一個 block 的峰值」再決定增益,所以增益是在**大聲的
內容送出之前**就降好的,不會有超標漏出去,也不需要在大聲的當下急拉增益
(那會有可聽見的抽動)。代價是一個 block 的延遲:64 取樣時 1.3 ms、
128 取樣時 2.7 ms。

這個前瞻只吃一個 block,是因為引擎本來就是逐 block 處理 —— 呼叫 process()
的時候整個 block 已經在手上了,所以「看未來」不需要額外的緩衝策略,
只要把輸出延後一個 block 就好。
"""

from __future__ import annotations

import numpy as np

# 天花板留 −0.3 dBFS。留一點餘裕是因為裝置端或之後的重取樣可能讓真實峰值
# 略高於取樣點峰值(intersample peak),頂到 0 dBFS 反而會在 DAC 那邊削到。
DEFAULT_CEILING_DB = -0.3

# 放開的速度。太快會有幫浦感(大聲段落之後音量忽然衝回來),
# 太慢則一次瞬間大聲會壓著後面好幾秒。
DEFAULT_RELEASE_MS = 80.0


class Limiter:
    """一個 block 前瞻的峰值限幅器。"""

    def __init__(self, samplerate: int, channels: int,
                 ceiling_db: float = DEFAULT_CEILING_DB,
                 release_ms: float = DEFAULT_RELEASE_MS) -> None:
        self.samplerate = int(samplerate)
        self.channels = int(channels)
        self.ceiling = float(10.0 ** (ceiling_db / 20.0))
        self.release_ms = float(release_ms)

        self._pending: np.ndarray | None = None      # 還沒送出的那一個 block
        self._gain = 1.0

    def reset(self) -> None:
        self._pending = None
        self._gain = 1.0

    @property
    def latency_samples(self) -> int:
        """前瞻造成的延遲。等於上一次處理的 block 長度。

        引擎每次都用同一個 blocksize 呼叫,所以這個值是穩定的;但它確實
        取決於呼叫端,latency_report() 要拿它去加,不能自己假設。
        """
        return 0 if self._pending is None else len(self._pending)

    @property
    def gain(self) -> float:
        """目前的增益(1.0 = 沒在壓)。給介面顯示壓縮量用。"""
        return self._gain

    def process(self, block: np.ndarray) -> np.ndarray:
        frames = len(block)
        if frames == 0:
            return block

        incoming = block
        pending = self._pending
        self._pending = incoming.copy()

        if pending is None or len(pending) != frames:
            # 第一個 block:還沒有東西可以送,輸出等長的靜音把管線填起來。
            # 這一個 block 的靜音就是前瞻的代價。
            return np.zeros((frames, self.channels), dtype=np.float32)

        # 前瞻:看「即將送出的下一個 block」的峰值,增益就能在它出場前
        # 先降好,不必在大聲的當下急拉(那會有可聽見的抽動)。
        peak_next = float(np.max(np.abs(incoming))) if incoming.size else 0.0
        target = 1.0 if peak_next <= self.ceiling else self.ceiling / peak_next

        if target < self._gain:
            new_gain = target          # 要壓就直接到位,斜坡落在較安靜的
        else:                          # 上一個 block 上,聽不出來
            block_ms = frames / self.samplerate * 1000.0
            step = min(1.0, block_ms / max(self.release_ms, 1e-3))
            new_gain = self._gain + (target - self._gain) * step

        # 光看下一個 block 是不夠的:真正要送出去的是 pending,而它的增益
        # 也必須壓得住它自己。串流剛開始、或大聲之後緊接著安靜時,前瞻算出
        # 的增益會是 1.0,若不再夾一次,那個大聲的 block 就原封不動漏出去
        # (實測會漏出峰值 4.0 的訊號)。
        peak_out = float(np.max(np.abs(pending))) if pending.size else 0.0
        allowed = 1.0 if peak_out <= self.ceiling else self.ceiling / peak_out
        start_gain = min(self._gain, allowed)
        end_gain = min(new_gain, allowed)
        self._gain = new_gain

        # 沒在壓的時候要位元透明。乘一個全是 1.0 的斜坡雖然「幾乎」無損,
        # 但那是 float32 的捨入誤差,量得到(−102 dB vs −157 dB)。
        # HiFi 的意思就是不做的事就真的別做。
        if start_gain == 1.0 and end_gain == 1.0:
            return pending

        ramp = np.linspace(start_gain, end_gain, frames,
                           dtype=np.float32)[:, np.newaxis]

        out = pending * ramp
        # 安全網:增益在 block 內是斜坡,理論上壓得住,但浮點與極端瞬變
        # 還是可能擦過天花板。硬夾一次不會有可聽見的代價。
        np.clip(out, -1.0, 1.0, out=out)
        return out.astype(np.float32)
