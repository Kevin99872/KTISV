"""訓練資料管線:分軌資料夾 → ``tf.data`` 的 (混音, 人聲) 批次。

合成規則不在這裡
----------------
響度比例隨機化、避免削波、靜音段剔除、人聲與伴奏獨立配對 —— 這四件事
``ktisv_research.mixing`` 已經處理好了,而且是純 numpy 的,兩個環境都能匯入。
這個檔案只負責「怎麼把資料餵進 TensorFlow」,不重寫混音規則:同一件事有
兩份實作,遲早會有一份悄悄走鐘,而聲音變差是不會報錯的。

不把資料讀進記憶體
------------------
121 分鐘的立體聲 44.1 kHz、人聲與伴奏各一份,攤成 float32 大約 5 GB ——
比這台機器的 VRAM 還大。所以走**隨機存取**:每次只從 wav 檔裡讀出需要的
那 3 秒。標註輸出是 16-bit PCM,``soundfile`` 可以直接 seek,成本是一次
磁碟尋道,比解碼整首歌便宜幾個數量級。

代價是每個樣本都要碰硬碟。實測下來仍然是 GPU 先吃飽 —— 一個樣本只讀
2 × 3 秒 × 2 聲道 × 2 bytes ≈ 1 MB。

切分按「歌」而不是按片段
------------------------
同一首歌的不同片段若同時出現在 train 與 val,驗證分數會漂亮得離譜 ——
模型早就聽過那個歌手、那段伴奏、那個混音風格了。所以驗證集是**整首**
被抽走的歌,模型在訓練時一秒都沒聽過。
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import soundfile as sf
import tensorflow as tf

# ktisv_research 是純 numpy 的部分(mixing / metrics),TF 環境也能用。
# 從 research/ 目錄執行時 sys.path 已經含當前目錄;直接跑檔案時補上。
if __package__ in (None, ""):  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ktisv_research.mixing import (MixConfig, augment_vocals, is_mostly_silent,
                                   mix_stems)

from . import SAMPLE_RATE


@dataclass
class DataConfig:
    samplerate: int = SAMPLE_RATE
    segment_samples: int = 130560          # 由 model.aligned_length 決定
    # 有多少比例的樣本用「人聲與伴奏隨機重新配對」。
    # 同一段人聲永遠只配同一段伴奏的話,模型可以靠「認出這段伴奏」反推人聲,
    # 而不是真的學會分離。拆開隨機配,組合數從 N 變成 N²。
    independent_prob: float = 0.5
    # 幾乎全靜音的片段重抽,最多試這麼多次(避免資料本身很安靜時卡住)
    silence_retries: int = 8
    mix: MixConfig = field(default_factory=MixConfig)


# ── 資料來源 ────────────────────────────────────────────────────────────
class Track:
    """一首歌的人聲與伴奏。只記路徑與長度,讀取時才碰硬碟。"""

    def __init__(self, name: str, vocals: Path, accompaniment: Path) -> None:
        self.name = name
        self.group = name
        self.vocals_path = vocals
        self.accompaniment_path = accompaniment

        info_v = sf.info(str(vocals))
        info_a = sf.info(str(accompaniment))
        if info_v.samplerate != info_a.samplerate:
            raise ValueError(f"{name}:兩軌取樣率不同 "
                             f"({info_v.samplerate} vs {info_a.samplerate})")
        self.samplerate = int(info_v.samplerate)
        self.frames = int(min(info_v.frames, info_a.frames))

    @property
    def seconds(self) -> float:
        return self.frames / self.samplerate

    def read(self, which: str, start: int, length: int) -> np.ndarray:
        """讀出 (length, 2) 的 float32。超出尾端就補零。"""
        path = self.vocals_path if which == "vocals" else self.accompaniment_path
        with sf.SoundFile(str(path)) as handle:
            handle.seek(min(start, max(0, self.frames - 1)))
            block = handle.read(length, dtype="float32", always_2d=True)
        return _fit(block, length)


def read_audio(path: Path, samplerate: int,
               max_seconds: float | None = None) -> np.ndarray:
    """讀任意音檔成 (samples, 2) 的 float32 @ samplerate。

    ``ktisv_research.data`` 有一份功能相同的 ``_read``,但那個模組在 module
    層 ``import torch``(它有一個 ``torch.utils.data.Dataset`` 子類別),而
    ``.venv-tf`` 裡的 torch 是 ``--no-deps`` 裝的、只是一包 CUDA DLL,
    import 會直接炸掉。所以這條 I/O 路徑在這裡自己走一遍。

    共用的界線是刻意畫在這裡的:**合成混音的規則與評分指標只該有一份
    實作**(那兩個是純 numpy 的,直接 import),但「怎麼把位元組變成陣列」
    這種管路重複一次無妨 —— 它壞掉會直接報錯,不會安靜地讓聲音變差。
    """
    path = Path(path)
    try:
        audio, rate = sf.read(str(path), dtype="float32", always_2d=True)
    except Exception:
        audio, rate = _decode_via_ffmpeg(path, samplerate, max_seconds)

    if rate != samplerate:
        audio = _resample(audio, rate, samplerate)
    if max_seconds:
        audio = audio[:int(max_seconds * samplerate)]
    return _fit(audio, len(audio))


def _decode_via_ffmpeg(path: Path, samplerate: int,
                       max_seconds: float | None) -> tuple[np.ndarray, int]:
    """soundfile 讀不了的格式(webm / m4a / opus)改用 ffmpeg。

    ``-t`` 放在 ``-i`` 前面是刻意的:那樣 ffmpeg 只解碼開頭那一段就收工,
    而不是整檔解完再切。快取裡有幾百 MB 的長檔,差別是幾秒還是幾分鐘。
    """
    import subprocess

    try:
        import imageio_ffmpeg
        exe = imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError as exc:
        raise RuntimeError(
            f"無法讀取 {path.name}:soundfile 不支援此格式,且找不到 "
            "imageio-ffmpeg。") from exc

    limit = ["-t", f"{max_seconds:.3f}"] if max_seconds else []
    proc = subprocess.run(
        [exe, "-hide_banner", "-loglevel", "error", *limit, "-i", str(path),
         "-vn", "-f", "f32le", "-acodec", "pcm_f32le",
         "-ac", "2", "-ar", str(samplerate), "-"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
        creationflags=0x08000000 if sys.platform == "win32" else 0)

    if proc.returncode != 0 or not proc.stdout:
        detail = proc.stderr.decode("utf-8", "replace").strip()
        raise RuntimeError(f"ffmpeg 解碼 {path.name} 失敗:{detail[:200]}")

    data = np.frombuffer(proc.stdout, dtype="<f4")
    usable = (data.size // 2) * 2
    return (np.ascontiguousarray(data[:usable].reshape(-1, 2),
                                 dtype=np.float32), samplerate)


def _resample(audio: np.ndarray, source_rate: int,
              target_rate: int) -> np.ndarray:
    from math import gcd

    from scipy.signal import resample_poly

    divisor = gcd(int(source_rate), int(target_rate))
    return resample_poly(audio, target_rate // divisor,
                         source_rate // divisor, axis=0).astype(np.float32)


def _fit(block: np.ndarray, length: int) -> np.ndarray:
    """統一成 (length, 2)。單聲道複製、多聲道取前兩軌、不足補零。"""
    if block.shape[1] == 1:
        block = np.repeat(block, 2, axis=1)
    elif block.shape[1] > 2:
        block = block[:, :2]
    if len(block) < length:
        pad = np.zeros((length - len(block), block.shape[1]), dtype=np.float32)
        block = np.concatenate([block, pad], axis=0)
    return np.ascontiguousarray(block[:length], dtype=np.float32)


def load_tracks(root: Path, limit: int | None = None) -> list[Track]:
    """掃描 ``<root>/<歌>/{vocals,accompaniment}.wav``。

    版面與 ``ktisv_research.data.load_pair_folders`` 相同 —— 不管資料是
    ``label_cache`` 產生的、``make_testset`` 合成的,還是解開的公開資料集,
    到這裡都長一樣。
    """
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"找不到資料目錄 {root}")

    tracks: list[Track] = []
    for folder in sorted(p for p in root.iterdir() if p.is_dir()):
        vocals = _find(folder, "vocals")
        accompaniment = (_find(folder, "accompaniment")
                         or _find(folder, "instrumental")
                         or _find(folder, "no_vocals"))
        if vocals is None or accompaniment is None:
            continue
        track = Track(folder.name, vocals, accompaniment)
        if track.frames <= 0:
            continue
        tracks.append(track)
        if limit and len(tracks) >= limit:
            break

    if not tracks:
        raise FileNotFoundError(
            f"{root} 底下找不到任何 vocals/accompaniment 配對資料夾。"
            "先跑 python -m ktisv_research.label_cache 產生訓練資料。")
    return tracks


def _find(folder: Path, name: str) -> Path | None:
    for suffix in (".wav", ".flac", ".ogg", ".mp3"):
        candidate = folder / f"{name}{suffix}"
        if candidate.exists():
            return candidate
    return None


def split_by_track(tracks: list[Track], val_ratio: float = 0.15,
                   seed: int = 0) -> tuple[list[Track], list[Track]]:
    """整首整首地切,驗證集的歌訓練時完全沒出現過。"""
    if len(tracks) < 2:
        raise ValueError(f"只有 {len(tracks)} 首,無法切出驗證集。")
    rng = np.random.default_rng(seed)
    order = list(range(len(tracks)))
    rng.shuffle(order)
    n_val = min(max(1, round(len(tracks) * val_ratio)), len(tracks) - 1)
    val_index = set(order[:n_val])
    train = [t for i, t in enumerate(tracks) if i not in val_index]
    val = [t for i, t in enumerate(tracks) if i in val_index]
    return train, val


def describe(tracks: list[Track]) -> str:
    total = sum(t.seconds for t in tracks)
    return f"{len(tracks)} 首 / {total / 60:.1f} 分鐘 @ {tracks[0].samplerate} Hz"


# ── 取樣 ────────────────────────────────────────────────────────────────
def sample_once(tracks: list[Track], config: DataConfig,
                rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """抽一個訓練樣本。回傳 (混音, 人聲),都是 (channels, samples)。"""
    length = config.segment_samples
    vocals = accompaniment = None

    for _ in range(config.silence_retries):
        track = tracks[int(rng.integers(len(tracks)))]
        vocals = track.read("vocals", _start(track, length, rng), length)

        if rng.random() < config.independent_prob and len(tracks) > 1:
            other = tracks[int(rng.integers(len(tracks)))]
        else:
            other = track
        accompaniment = other.read("accompaniment",
                                   _start(other, length, rng), length)

        # 整段沒有人聲的樣本學不到「哪裡是人聲」,只會稀釋損失。
        # 伴奏可以安靜(那是合理的獨唱段落),人聲不行。
        if not is_mostly_silent(vocals):
            break

    stems = mix_stems(augment_vocals(vocals, rng), accompaniment,
                      config.mix, rng)
    # 模型吃 (channels, samples),音訊慣例是 (samples, channels)
    return stems["mixture"].T.copy(), stems["vocals"].T.copy()


def _start(track: Track, length: int, rng: np.random.Generator) -> int:
    if track.frames <= length:
        return 0
    return int(rng.integers(0, track.frames - length + 1))


def make_dataset(tracks: list[Track], config: DataConfig, batch_size: int,
                 seed: int = 0, deterministic: bool = False,
                 length: int | None = None) -> tf.data.Dataset:
    """建 ``tf.data`` 管線。

    ``deterministic=True`` 給驗證集用:每次驗證都在**同一批**混音上算分,
    否則分不出是模型變好、還是這次剛好抽到比較簡單的題目。

    ``length`` 是「一個 epoch 產幾個樣本」。這類訓練是無限取樣的,
    epoch 只是記錄與排程的單位,和底層有幾首歌無關。
    """
    samples = config.segment_samples
    channels = 2

    def generator():
        index = 0
        while True:
            # 決定性模式下,第 i 個樣本永遠由同一顆種子產生。每次重新迭代
            # 這個 dataset 都會重跑 generator,index 從 0 開始 —— 所以每一輪
            # 驗證看到的是同一批混音。
            rng = (np.random.default_rng([seed, index]) if deterministic
                   else np.random.default_rng([seed, index, _entropy()]))
            yield sample_once(tracks, config, rng)
            index = index + 1 if length is None else (index + 1) % length

    spec = (tf.TensorSpec(shape=(channels, samples), dtype=tf.float32),
            tf.TensorSpec(shape=(channels, samples), dtype=tf.float32))
    dataset = tf.data.Dataset.from_generator(generator, output_signature=spec)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    return dataset.prefetch(tf.data.AUTOTUNE)


_counter = np.random.default_rng()


def _entropy() -> int:
    return int(_counter.integers(0, 2 ** 31 - 1))


__all__ = ["DataConfig", "Track", "load_tracks", "split_by_track", "describe",
           "sample_once", "make_dataset"]
