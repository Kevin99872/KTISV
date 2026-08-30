"""訓練資料管線:把分軌資料集變成 (混音, 人聲, 伴奏) 的張量批次。

資料來源被抽象成同一個東西 —— 一組 :class:`StemPair`(一段人聲 + 一段對應
的伴奏)。不管來自 MIR-1K 的左右聲道、MUSDB18 的分軌資料夾,還是合成音源,
進到訓練迴圈時都長一樣。要加新資料集只要多寫一個 loader。

切分要按「組」而不是按片段
--------------------------
MIR-1K 的 1000 個片段來自 110 首歌、19 位歌手。若隨機把片段丟進 train/val,
同一首歌的不同片段會同時出現在兩邊 —— 驗證分數會漂亮得離譜,因為模型
早就聽過那個歌手、那個伴奏、那個混音風格了。

所以預設**按歌手切**:驗證集裡的人,模型在訓練時完全沒聽過。這比按歌切
更嚴格,而分離模型最該具備的能力正是「換一個沒聽過的人也分得開」。

為什麼人聲與伴奏要能獨立配對
--------------------------
同一段人聲永遠只配同一段伴奏的話,模型可以靠「認出這段伴奏」來反推人聲,
而不是真的學會分離。把人聲與伴奏拆開來隨機配,組合數從 N 變成 N²,而且
逼模型只能靠聲學特徵判斷。代價是配出來的組合不見得音樂上合理(調性、
速度都對不上),所以預設只有一半的樣本這樣配,另一半維持原本的配對。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from . import SAMPLE_RATE
from .mixing import (MixConfig, augment_vocals, is_mostly_silent, mix_stems,
                     random_crop)


@dataclass
class DataConfig:
    samplerate: int = SAMPLE_RATE
    segment_seconds: float = 6.0
    # 有多少比例的樣本用「人聲與伴奏隨機重新配對」(見模組說明)
    independent_prob: float = 0.5
    # 幾乎全靜音的片段重抽,最多試這麼多次就放棄(避免資料本身就很安靜時卡住)
    silence_retries: int = 8
    mix: MixConfig = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.mix is None:
            self.mix = MixConfig()

    @property
    def segment_samples(self) -> int:
        return int(self.segment_seconds * self.samplerate)


@dataclass(slots=True)
class StemPair:
    """一段人聲與它對應的伴奏。兩者等長、同取樣率、(samples, 2)。"""

    name: str
    group: str          # 切分用的分組鍵(歌手 / 專輯 / 歌曲)
    vocals: np.ndarray
    accompaniment: np.ndarray

    @property
    def samples(self) -> int:
        return len(self.vocals)


# ── 讀檔 ────────────────────────────────────────────────────────────────
def _read(path: Path, samplerate: int) -> np.ndarray:
    """讀成 (samples, channels) 的 float32,並重取樣到指定頻率。"""
    from .evaluate import read_audio

    audio, source_rate = read_audio(path)
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 1:
        audio = audio[:, None]
    if source_rate != samplerate:
        audio = _resample(audio, source_rate, samplerate)
    return audio


def _resample(audio: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    from math import gcd

    from scipy.signal import resample_poly

    divisor = gcd(int(source_rate), int(target_rate))
    up, down = int(target_rate) // divisor, int(source_rate) // divisor
    return resample_poly(audio, up, down, axis=0).astype(np.float32)


def _as_stereo(x: np.ndarray) -> np.ndarray:
    if x.ndim == 1:
        x = x[:, None]
    if x.shape[1] == 1:
        return np.repeat(x, 2, axis=1)
    return x[:, :2]


# ── 各資料集的 loader ───────────────────────────────────────────────────
_MIR1K_NAME = re.compile(r"^([^_]+)_(\d+)_(\d+)$")


def load_mir1k(root: Path, config: DataConfig,
               limit: int | None = None) -> list[StemPair]:
    """MIR-1K:單一 wav 檔,左聲道是伴奏、右聲道是人聲。

    檔名格式為 ``歌手_歌曲_片段``,所以歌手可以直接從檔名取出來當分組鍵。
    原始資料是 16 kHz 單聲道音源 —— 重取樣到 44.1 kHz 不會把 8 kHz 以上的
    內容變出來,那個頻段在這批資料裡本來就是空的。這不是錯誤,但要知道:
    只用 MIR-1K 訓練出來的模型,對高頻沒有任何學習訊號。
    """
    root = Path(root)
    files = sorted(p for p in root.rglob("*.wav") if p.is_file())
    if not files:
        raise FileNotFoundError(f"{root} 底下找不到任何 wav —— MIR-1K 解開了嗎?")

    pairs: list[StemPair] = []
    for path in files[:limit]:
        audio, source_rate = _read_raw(path)
        if audio.ndim < 2 or audio.shape[1] < 2:
            raise ValueError(
                f"{path.name} 不是雙聲道。MIR-1K 靠左右聲道分開伴奏與人聲,"
                "單聲道檔案無法作為訓練資料。")

        accompaniment = _as_stereo(audio[:, 0:1])
        vocals = _as_stereo(audio[:, 1:2])
        if source_rate != config.samplerate:
            accompaniment = _resample(accompaniment, source_rate, config.samplerate)
            vocals = _resample(vocals, source_rate, config.samplerate)

        match = _MIR1K_NAME.match(path.stem)
        group = match.group(1) if match else path.stem
        pairs.append(StemPair(path.stem, group, vocals, accompaniment))

    return pairs


def _read_raw(path: Path) -> tuple[np.ndarray, int]:
    from .evaluate import read_audio

    audio, rate = read_audio(path)
    return np.asarray(audio, dtype=np.float32), rate


def load_pair_folders(root: Path, config: DataConfig,
                      limit: int | None = None) -> list[StemPair]:
    """每個子資料夾一首歌,裡面有 vocals.* 與 accompaniment.*(或 instrumental.*)。

    這個版面同時涵蓋 ``make_testset.py`` 產生的合成音源、``evaluate.py`` 的
    測試集,以及大多數分軌資料集解開後的樣子。
    """
    from .evaluate import find_stem

    root = Path(root)
    pairs: list[StemPair] = []
    for folder in sorted(p for p in root.iterdir() if p.is_dir()):
        vocals_path = find_stem(folder, "vocals")
        accompaniment_path = (find_stem(folder, "accompaniment")
                              or find_stem(folder, "instrumental"))
        if vocals_path is None or accompaniment_path is None:
            continue

        vocals = _as_stereo(_read(vocals_path, config.samplerate))
        accompaniment = _as_stereo(_read(accompaniment_path, config.samplerate))
        n = min(len(vocals), len(accompaniment))
        if n <= 0:
            continue
        pairs.append(StemPair(folder.name, folder.name,
                              vocals[:n], accompaniment[:n]))
        if limit and len(pairs) >= limit:
            break

    if not pairs:
        raise FileNotFoundError(
            f"{root} 底下找不到任何 vocals/accompaniment 配對子資料夾。")
    return pairs


LOADERS = {
    "mir1k": load_mir1k,
    "folders": load_pair_folders,
}


def load_dataset(kind: str, root: Path, config: DataConfig,
                 limit: int | None = None) -> list[StemPair]:
    if kind not in LOADERS:
        raise ValueError(f"未知的資料集型別 {kind};可用:{list(LOADERS)}")
    return LOADERS[kind](Path(root), config, limit)


# ── 切分 ────────────────────────────────────────────────────────────────
def split_by_group(pairs: list[StemPair], val_ratio: float = 0.15,
                   seed: int = 0) -> tuple[list[StemPair], list[StemPair]]:
    """按分組鍵切 train/val。同一組絕不會同時出現在兩邊(見模組說明)。"""
    groups = sorted({p.group for p in pairs})
    if len(groups) < 2:
        raise ValueError(
            f"只有 {len(groups)} 個分組,無法做不重疊的切分。"
            "資料太少,或分組鍵抓錯了。")

    rng = np.random.default_rng(seed)
    shuffled = list(groups)
    rng.shuffle(shuffled)
    # 至少留一組給驗證,也至少留一組給訓練
    n_val = min(max(1, round(len(groups) * val_ratio)), len(groups) - 1)
    val_groups = set(shuffled[:n_val])

    train = [p for p in pairs if p.group not in val_groups]
    val = [p for p in pairs if p.group in val_groups]
    return train, val


# ── Dataset ─────────────────────────────────────────────────────────────
class MixtureDataset(Dataset):
    """隨機裁切 → 隨機配對 → 合成混音。

    ``length`` 是「一個 epoch 要生幾個樣本」,與底層片段數無關 —— 這類訓練
    是無限取樣的,epoch 只是記錄與排程的單位。

    ``deterministic=True`` 時每個索引固定產生同一個樣本,給驗證集用:
    驗證分數若每次都在不同的隨機混音上算,就分不出是模型變好還是題目變簡單。
    """

    def __init__(self, pairs: list[StemPair], config: DataConfig,
                 length: int, seed: int = 0, deterministic: bool = False) -> None:
        if not pairs:
            raise ValueError("沒有任何資料。")
        self.pairs = pairs
        self.config = config
        self.length = int(length)
        self.seed = int(seed)
        self.deterministic = deterministic

    def __len__(self) -> int:
        return self.length

    def _rng(self, index: int) -> np.random.Generator:
        if self.deterministic:
            return np.random.default_rng([self.seed, index])
        # 非決定性:每次取樣都不同,但仍然由 seed 起頭以便整體重現
        return np.random.default_rng(
            [self.seed, index, int(torch.randint(0, 2 ** 31 - 1, (1,)).item())])

    def _crop(self, pair: StemPair, which: str,
              rng: np.random.Generator) -> np.ndarray:
        source = pair.vocals if which == "vocals" else pair.accompaniment
        return random_crop(source, self.config.segment_samples, rng)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        rng = self._rng(index)
        config = self.config

        vocals = accompaniment = None
        for _ in range(config.silence_retries):
            pair = self.pairs[int(rng.integers(len(self.pairs)))]
            vocals = self._crop(pair, "vocals", rng)

            if rng.random() < config.independent_prob and len(self.pairs) > 1:
                other = self.pairs[int(rng.integers(len(self.pairs)))]
            else:
                other = pair
            accompaniment = self._crop(other, "accompaniment", rng)

            # 整段沒有人聲的樣本學不到「哪裡是人聲」,只會稀釋損失。
            # 伴奏可以安靜(那是合理的獨唱段落),人聲不行。
            if not is_mostly_silent(vocals):
                break

        vocals = augment_vocals(vocals, rng)
        stems = mix_stems(vocals, accompaniment, config.mix, rng)

        return {
            # 模型吃 (channels, samples),而音訊慣例是 (samples, channels)
            "mixture": torch.from_numpy(stems["mixture"].T.copy()),
            "vocals": torch.from_numpy(stems["vocals"].T.copy()),
            "accompaniment": torch.from_numpy(stems["accompaniment"].T.copy()),
        }


def describe(pairs: list[StemPair], samplerate: int) -> str:
    """一行摘要,訓練開始前印出來確認資料真的是預期的樣子。"""
    total = sum(p.samples for p in pairs) / samplerate
    groups = len({p.group for p in pairs})
    return (f"{len(pairs)} 段 / {groups} 組 / 共 {total / 60:.1f} 分鐘"
            f" @ {samplerate} Hz")
