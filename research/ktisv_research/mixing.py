"""合成訓練資料:把純人聲與純伴奏混成有正確答案的三元組。

為什麼需要這個
--------------
監督式分離訓練需要 (混音, 人聲, 伴奏) 的配對資料。真實歌曲的分軌極其稀少
(MUSDB18 只有 150 首,且以西洋曲風為主),但**純人聲**資料集因為歌聲合成
研究而相對豐富,且涵蓋多種語言。

把純人聲乘上各種伴奏,就能生出大量配對資料 —— 而且人聲的正確答案是精確的,
因為它本來就是單獨錄的。這是取得多語言涵蓋率最實際的路徑。

混音時真正要注意的事
--------------------
天真地把兩段音訊相加會產生模型學不到東西的資料:

1. **響度比例要有變化**。真實歌曲的人聲/伴奏比大約在 -10 ~ +5 dB 之間。
   若訓練資料全是固定比例,模型會學到那個比例而不是學會分離。
2. **不能削波**。相加後超過 ±1.0 會截斷,產生真實音樂裡不存在的失真,
   模型會去學那個假象。
3. **靜音段要處理**。人聲資料集常有大段空白,整段都是靜音的樣本對訓練
   沒有貢獻,還會稀釋 loss。
4. **人聲與伴奏要獨立取樣**。同一段人聲要能配上不同伴奏,否則模型會
   把特定人聲和特定伴奏綁在一起記住。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import SAMPLE_RATE

EPS = 1e-10


def rms(x: np.ndarray) -> float:
    """均方根振幅。"""
    if x.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64))))


def rms_db(x: np.ndarray) -> float:
    value = rms(x)
    return 20.0 * np.log10(value) if value > EPS else -np.inf


def set_rms_db(x: np.ndarray, target_db: float) -> np.ndarray:
    """把訊號縮放到指定的 RMS 響度(dBFS)。"""
    current = rms(x)
    if current < EPS:
        return x.copy()
    gain = (10.0 ** (target_db / 20.0)) / current
    return (x * gain).astype(np.float32)


def is_mostly_silent(x: np.ndarray, threshold_db: float = -50.0,
                     min_active_ratio: float = 0.15) -> bool:
    """判斷片段是否幾乎全靜音。

    以 50 ms 的窗檢查有多少比例超過門檻。人聲資料集常有長段空白,
    整段靜音的樣本對訓練沒有貢獻。
    """
    if x.size == 0:
        return True
    window = max(1, int(0.05 * SAMPLE_RATE))
    mono = x.mean(axis=1) if x.ndim > 1 else x
    frames = mono[:len(mono) // window * window].reshape(-1, window)
    if frames.size == 0:
        return rms_db(mono) < threshold_db
    frame_db = 20.0 * np.log10(np.maximum(
        np.sqrt(np.mean(frames ** 2, axis=1)), EPS))
    return float(np.mean(frame_db > threshold_db)) < min_active_ratio


@dataclass
class MixConfig:
    """合成參數。預設值取自真實流行歌的統計分佈。"""

    # 人聲相對伴奏的響度比(dB)。真實歌曲大多落在這個區間。
    snr_db_range: tuple[float, float] = (-5.0, 10.0)
    # 混音後的目標響度。留 headroom 給後續處理。
    target_db: float = -18.0
    # 超過這個峰值就整體縮小,避免削波
    peak_ceiling: float = 0.99
    # 隨機增益抖動,讓模型不依賴絕對音量
    gain_jitter_db: float = 3.0


def mix_stems(vocals: np.ndarray, accompaniment: np.ndarray,
              config: MixConfig | None = None,
              rng: np.random.Generator | None = None) -> dict[str, np.ndarray]:
    """把人聲與伴奏混成訓練樣本。

    回傳 ``{"mixture", "vocals", "accompaniment"}`` —— 三者的關係嚴格滿足
    ``mixture == vocals + accompaniment``,這是監督式訓練的前提。
    任何對混音的縮放都必須同步套用到兩個分軌上,否則正確答案就錯了。
    """
    config = config or MixConfig()
    rng = rng or np.random.default_rng()

    vocals = _as_stereo(vocals)
    accompaniment = _as_stereo(accompaniment)

    # 對齊長度:短的那個決定樣本長度
    n = min(len(vocals), len(accompaniment))
    vocals, accompaniment = vocals[:n].copy(), accompaniment[:n].copy()

    # 依隨機 SNR 調整兩者比例
    snr = rng.uniform(*config.snr_db_range)
    vocal_rms, acc_rms = rms(vocals), rms(accompaniment)
    if vocal_rms > EPS and acc_rms > EPS:
        # 固定伴奏,調整人聲以達到目標 SNR
        target_vocal_rms = acc_rms * (10.0 ** (snr / 20.0))
        vocals *= target_vocal_rms / vocal_rms

    mixture = vocals + accompaniment

    # 整體調到目標響度,再加一點抖動
    jitter = rng.uniform(-config.gain_jitter_db, config.gain_jitter_db)
    mix_rms = rms(mixture)
    if mix_rms > EPS:
        gain = (10.0 ** ((config.target_db + jitter) / 20.0)) / mix_rms
    else:
        gain = 1.0

    # 防削波:縮放後若仍超過上限,再壓下來。
    # 關鍵 —— 這個增益必須同步套用到分軌,不能只改混音。
    peak = float(np.max(np.abs(mixture))) * gain
    if peak > config.peak_ceiling:
        gain *= config.peak_ceiling / peak

    return {
        "mixture": (mixture * gain).astype(np.float32),
        "vocals": (vocals * gain).astype(np.float32),
        "accompaniment": (accompaniment * gain).astype(np.float32),
    }


def _as_stereo(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.shape[1] == 1:
        arr = np.repeat(arr, 2, axis=1)
    elif arr.shape[1] > 2:
        arr = arr[:, :2]
    return arr


def random_crop(x: np.ndarray, length: int,
                rng: np.random.Generator | None = None) -> np.ndarray:
    """隨機裁切固定長度的片段;不足則補零。"""
    rng = rng or np.random.default_rng()
    if len(x) <= length:
        pad = np.zeros((length - len(x), x.shape[1]), dtype=x.dtype)
        return np.concatenate([x, pad], axis=0)
    start = int(rng.integers(0, len(x) - length + 1))
    return x[start:start + length]


def augment_vocals(vocals: np.ndarray,
                   rng: np.random.Generator | None = None) -> np.ndarray:
    """對人聲做輕度增強,增加多樣性。

    刻意**不做**音高或速度變換 —— 那會改變人聲的諧波結構,而那正是模型
    要學的判別特徵,扭曲它反而有害。這裡只做不改變頻譜結構的操作。
    """
    rng = rng or np.random.default_rng()
    out = vocals.copy()

    # 隨機左右聲道交換(人聲通常置中,但錄音可能有偏移)
    if rng.random() < 0.5 and out.shape[1] == 2:
        out = out[:, ::-1].copy()

    # 隨機極性反轉:對聽感無影響,但能防止模型記住波形的絕對相位
    if rng.random() < 0.5:
        out = -out

    return out
