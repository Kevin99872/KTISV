"""零和檢查:不需要正確答案的客觀測試。

原理
----
分離出來的人聲與伴奏加回去,理論上應該等於原曲::

    vocals + accompaniment == mixture

兩者的差(殘差)就是模型**弄丟或憑空造出**的東西。這是唯一不需要正確答案
就能做的客觀量測 —— 對只有原曲的真實歌曲特別有用。

殘差怎麼解讀
------------
殘差本身**不等於分離品質**。一個把所有東西都丟進伴奏軌、人聲軌全靜音的
爛模型,殘差會是完美的零。所以這個指標只能用來抓特定的失敗:

  * **殘差大** → 模型在重建時丟失了能量,通常伴隨可聽見的破損
  * **殘差集中在某頻段** → 那個頻段被錯誤處理
  * **殘差接近零** → 重建完整(但不保證分得對)

換句話說:殘差大一定有問題,殘差小不代表沒問題。

用法::

    python -m ktisv_research.nulltest data/results/real --original <原曲資料夾>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from .evaluate import AUDIO_SUFFIXES, read_audio
from .metrics import usdr


def find_original(name: str, search_dirs: list[Path]) -> Path | None:
    """依分離結果的資料夾名回頭找原曲。"""
    for folder in search_dirs:
        if not folder.is_dir():
            continue
        for suffix in AUDIO_SUFFIXES:
            candidate = folder / f"{name}{suffix}"
            if candidate.exists():
                return candidate
    return None


def band_residual(residual: np.ndarray, samplerate: int) -> dict[str, float]:
    """把殘差能量拆到各頻段,看問題出在哪裡。"""
    mono = residual.mean(axis=1) if residual.ndim > 1 else residual
    spectrum = np.abs(np.fft.rfft(mono))
    freqs = np.fft.rfftfreq(len(mono), 1.0 / samplerate)

    bands = {"低頻 <250Hz": (0, 250), "中低 250-1k": (250, 1000),
             "中高 1k-4k": (1000, 4000), "高頻 >4k": (4000, samplerate / 2)}
    total = float(np.sum(spectrum ** 2)) + 1e-12
    return {name: float(np.sum(spectrum[(freqs >= lo) & (freqs < hi)] ** 2)
                        / total * 100.0)
            for name, (lo, hi) in bands.items()}


def check_one(folder: Path, original: Path) -> dict | None:
    vocals_path = folder / "vocals.wav"
    acc_path = folder / "accompaniment.wav"
    if not (vocals_path.exists() and acc_path.exists()):
        return None

    vocals, rate = read_audio(vocals_path)
    accompaniment, _ = read_audio(acc_path)
    mixture, mix_rate = read_audio(original)

    if mix_rate != rate:
        from .separate import _resample
        mixture = _resample(mixture, mix_rate, rate)

    n = min(len(vocals), len(accompaniment), len(mixture))
    vocals, accompaniment, mixture = vocals[:n], accompaniment[:n], mixture[:n]

    reconstructed = vocals + accompaniment
    residual = mixture - reconstructed

    # 用 SDR 衡量重建品質:越高代表殘差越小
    reconstruction_sdr = usdr(mixture, reconstructed)

    # 兩軌的能量佔比 —— 若人聲軌幾乎沒能量,代表模型把一切都丟給伴奏了
    vocal_energy = float(np.sum(vocals ** 2))
    acc_energy = float(np.sum(accompaniment ** 2))
    total_energy = vocal_energy + acc_energy + 1e-12

    return {
        "name": folder.name,
        "reconstruction_sdr": reconstruction_sdr,
        "residual_db": 20.0 * np.log10(
            max(float(np.sqrt(np.mean(residual ** 2))), 1e-12)),
        "vocal_share": vocal_energy / total_energy * 100.0,
        "bands": band_residual(residual, rate),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="ktisv_research.nulltest",
        description="零和檢查:驗證 vocals + accompaniment == mixture")
    parser.add_argument("results", type=Path,
                        help="分離結果的資料夾(每個子資料夾含 vocals/accompaniment)")
    parser.add_argument("--original", type=Path, action="append", default=[],
                        help="原曲所在資料夾(可指定多次)")
    args = parser.parse_args(argv)

    if not args.results.is_dir():
        print(f"找不到:{args.results}")
        return 1

    search_dirs = args.original or []
    if not search_dirs:
        print("請用 --original 指定原曲所在的資料夾。")
        return 1

    folders = [f for f in sorted(args.results.iterdir()) if f.is_dir()]
    if not folders:
        print(f"{args.results} 底下沒有分離結果。")
        return 1

    print(f"{'曲目':<24}{'重建SDR':>10}{'殘差':>10}{'人聲佔比':>10}   殘差頻段分佈")
    print("─" * 96)

    records = []
    for folder in folders:
        original = find_original(folder.name, search_dirs)
        if original is None:
            print(f"{folder.name[:23]:<24}{'找不到原曲':>10}")
            continue
        try:
            record = check_one(folder, original)
        except Exception as exc:
            print(f"{folder.name[:23]:<24}  失敗:{str(exc)[:50]}")
            continue
        if record is None:
            continue

        worst_band = max(record["bands"].items(), key=lambda kv: kv[1])
        sdr_text = ("∞" if np.isinf(record["reconstruction_sdr"])
                    else f"{record['reconstruction_sdr']:.1f}")
        print(f"{record['name'][:23]:<24}{sdr_text:>10}"
              f"{record['residual_db']:>10.1f}"
              f"{record['vocal_share']:>9.1f}%   "
              f"{worst_band[0]} 佔 {worst_band[1]:.0f}%")
        records.append(record)

    if not records:
        return 1

    print()
    sdrs = [r["reconstruction_sdr"] for r in records
            if np.isfinite(r["reconstruction_sdr"])]
    if sdrs:
        median = float(np.median(sdrs))
        print(f"重建 SDR 中位數:{median:.1f} dB")
        if median > 40:
            print("  → 重建幾乎無損,模型沒有丟失能量。")
        elif median > 20:
            print("  → 有輕微損失,通常聽不出來。")
        else:
            print("  → 損失明顯,分離結果可能有可聽見的破損。")

    shares = [r["vocal_share"] for r in records]
    print(f"人聲軌能量佔比中位數:{np.median(shares):.1f}%")
    print("  (典型流行歌約 10~30%。過低代表人聲被吃掉,"
          "過高代表伴奏漏進人聲軌。)")

    print("\n注意:殘差小不代表分得對 —— 把所有東西丟進伴奏軌的爛模型,")
    print("      殘差一樣會是零。這個檢查只能抓「重建有破損」這類問題。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
