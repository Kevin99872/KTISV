"""建立有正確答案的測試集。

為什麼需要這個
--------------
從 YouTube 抓下來的歌只有混音,沒有正確答案,所以只能主觀試聽。要得到
客觀分數,必須知道人聲與伴奏各自長什麼樣。

在拿到正式資料集(需要授權流程)之前,可以用**已知成分的合成音**先把
評估流程跑通,並且測出模型在特定情境下的行為 —— 例如把置中人聲換成
偏一邊、或把伴奏換成音域與人聲重疊的樂器。

這些合成樣本**不能代表真實音樂的表現**,但可以用來:
  * 驗證整條評估管線正確
  * 做受控實驗(單獨改變一個變因,看分數怎麼動)
  * 當作訓練資料增強的原型

用法::

    python -m ktisv_research.make_testset data/testset
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

from . import SAMPLE_RATE
from .mixing import mix_stems

rng = np.random.default_rng(20260726)


def _envelope(t: np.ndarray, rate: float = 0.7) -> np.ndarray:
    """樂句包絡 —— 真實歌聲不會整首持續發聲。"""
    return 0.5 + 0.5 * np.sin(2 * np.pi * rate * t - np.pi / 2)


def voice(t: np.ndarray, f0: float, harmonics: int = 8,
          vibrato_hz: float = 5.5, vibrato_depth: float = 0.02,
          breathiness: float = 0.0) -> np.ndarray:
    """合成帶顫音與諧波衰減的歌聲。

    ``breathiness`` 加入氣音成分 —— 那是真實人聲有而純正弦沒有的特徵,
    也是分離模型判別人聲的重要線索之一。
    """
    vibrato = 1.0 + vibrato_depth * np.sin(2 * np.pi * vibrato_hz * t)
    signal = sum(
        (0.6 ** k) * np.sin(2 * np.pi * f0 * (k + 1) * t * vibrato)
        for k in range(harmonics)
    )
    if breathiness > 0:
        noise = rng.standard_normal(len(t))
        # 讓氣音集中在高頻
        noise = np.convolve(noise, [1.0, -0.85], mode="same")
        signal = signal + breathiness * noise
    return signal * _envelope(t)


def stereo(mono: np.ndarray, pan: float = 0.0) -> np.ndarray:
    """單聲道加上定位。pan: -1 全左,0 置中,+1 全右。"""
    left = mono * np.sqrt((1.0 - pan) / 2.0)
    right = mono * np.sqrt((1.0 + pan) / 2.0)
    return np.column_stack([left, right]).astype(np.float32)


def band(t: np.ndarray, root: float, kind: str) -> np.ndarray:
    """合成伴奏。``kind`` 決定樂器的頻率分佈。"""
    if kind == "wide":
        # 低音 + 和弦 + 打擊:與人聲的頻譜重疊少
        bass = 0.45 * np.sin(2 * np.pi * root * t)
        chord = sum(0.10 * np.sin(2 * np.pi * root * r * t)
                    for r in (4.0, 5.04, 6.0))
        hats = 0.12 * rng.standard_normal(len(t)) * (
            np.sin(2 * np.pi * 4 * t) > 0.93)
        return bass + chord + hats

    if kind == "overlap":
        # 音域與人聲高度重疊的樂器(弦樂、二胡這類)—— 最難分離的情境
        lead = sum((0.5 ** k) * np.sin(2 * np.pi * root * 3.0 * (k + 1) * t *
                                       (1 + 0.015 * np.sin(2 * np.pi * 4.2 * t)))
                   for k in range(6))
        bass = 0.3 * np.sin(2 * np.pi * root * t)
        return 0.45 * lead + bass

    if kind == "sparse":
        # 稀疏編制:只有低音與零星和弦,人聲最容易被分出來
        bass = 0.5 * np.sin(2 * np.pi * root * t)
        pluck = np.zeros_like(t)
        for onset in np.arange(0, t[-1], 1.0):
            index = int(onset * SAMPLE_RATE)
            length = min(int(0.4 * SAMPLE_RATE), len(t) - index)
            if length <= 0:
                continue
            decay = np.exp(-np.arange(length) / (0.12 * SAMPLE_RATE))
            pluck[index:index + length] += 0.3 * decay * np.sin(
                2 * np.pi * root * 4 * np.arange(length) / SAMPLE_RATE)
        return bass + pluck

    raise ValueError(f"未知的伴奏類型:{kind}")


# 受控實驗:每個案例只改變一個變因,才能歸因分數的變化
CASES = [
    # (名稱, 人聲基頻, 人聲定位, 氣音, 伴奏類型, 說明)
    ("centred_wide", 233.0, 0.0, 0.15, "wide",
     "置中人聲 + 寬編制伴奏(最典型的流行歌配置)"),
    ("centred_overlap", 233.0, 0.0, 0.15, "overlap",
     "置中人聲 + 音域重疊的樂器(二胡/弦樂這類的難題)"),
    ("centred_sparse", 233.0, 0.0, 0.15, "sparse",
     "置中人聲 + 稀疏伴奏(理論上最容易)"),
    ("panned_wide", 233.0, -0.4, 0.15, "wide",
     "人聲偏左 + 寬編制(測試對非置中人聲的依賴)"),
    ("low_voice", 130.0, 0.0, 0.15, "wide",
     "低音域人聲(男低音),與貝斯頻段接近"),
    ("high_voice", 392.0, 0.0, 0.15, "wide",
     "高音域人聲(女高音)"),
    ("breathy", 233.0, 0.0, 0.45, "wide",
     "氣音很重的唱腔"),
    ("clean_tone", 233.0, 0.0, 0.0, "wide",
     "完全沒有氣音的純諧波人聲"),
]


def build(output: Path, seconds: float = 20.0) -> list[dict]:
    t = np.arange(int(seconds * SAMPLE_RATE)) / SAMPLE_RATE
    manifest: list[dict] = []

    for name, f0, pan, breath, kind, description in CASES:
        vocal = stereo(voice(t, f0, breathiness=breath), pan)
        accompaniment = stereo(band(t, 82.0, kind), 0.0)
        # 伴奏加一點立體聲寬度(真實混音的伴奏不會完全置中)
        accompaniment[:, 1] = np.roll(accompaniment[:, 1], 13)

        # 注意:不能用 Python 內建的 hash() —— 字串雜湊每個行程都會隨機化,
        # 測試集會變得不可重現,那樣就無法比較微調前後的分數。
        seed = int(hashlib.sha256(name.encode("utf-8")).hexdigest()[:8], 16)
        mixed = mix_stems(vocal, accompaniment,
                          rng=np.random.default_rng(seed))

        folder = output / name
        folder.mkdir(parents=True, exist_ok=True)
        for stem, data in mixed.items():
            sf.write(str(folder / f"{stem}.wav"), data, SAMPLE_RATE)

        residual = float(np.max(np.abs(
            mixed["mixture"] - mixed["vocals"] - mixed["accompaniment"])))
        manifest.append({"name": name, "description": description,
                         "f0": f0, "pan": pan, "breathiness": breath,
                         "accompaniment": kind, "residual": residual})
        print(f"  {name:<18} {description}")

    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="ktisv_research.make_testset",
        description="建立有正確答案的合成測試集")
    parser.add_argument("output", type=Path, nargs="?",
                        default=Path("data/testset"))
    parser.add_argument("--seconds", type=float, default=20.0)
    args = parser.parse_args(argv)

    print(f"建立測試集到 {args.output}(每段 {args.seconds:.0f} 秒)\n")
    manifest = build(args.output, args.seconds)

    worst = max(m["residual"] for m in manifest)
    print(f"\n可加性檢查:最大殘差 {worst:.2e}")
    if worst > 1e-5:
        print("  警告:殘差過大,正確答案可能不精確!")
        return 1

    print(f"\n完成:{len(manifest)} 個案例")
    print("\n注意:這是合成音,不能代表真實音樂的表現。")
    print("      用途是驗證評估流程,以及做單一變因的受控實驗。")
    print(f"\n接著執行:")
    print(f"  python -m ktisv_research.evaluate {args.output} --full")
    return 0


if __name__ == "__main__":
    sys.exit(main())
