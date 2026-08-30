"""分離品質評估 CLI。

用途:在花時間訓練之前,先量出現成模型在**你關心的曲風**上到底表現如何。

兩種輸入
--------
**有正確答案**(能算分數):資料夾裡放
``mixture.*`` + ``vocals.*``(``accompaniment.*`` 可省略,會用 mixture - vocals 推出來)

**沒有正確答案**(只能聽):直接放音訊檔。程式會輸出分離結果讓你試聽,
但不會有分數 —— 沒有正確答案就無法計算誤差,這點沒有捷徑。

用法::

    python -m ktisv_research.evaluate data/testset --out results
    python -m ktisv_research.evaluate mysong.mp3 --out results
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from .metrics import evaluate_track, summarize
from .separate import DEFAULT_MODEL, available_device, model_info, separate

AUDIO_SUFFIXES = {".wav", ".flac", ".mp3", ".ogg", ".m4a", ".aiff", ".aif",
                  ".opus", ".webm", ".aac", ".wma", ".mp4", ".mkv"}


def read_audio(path: Path) -> tuple[np.ndarray, int]:
    """讀入音訊。soundfile 讀不了的格式(m4a / webm / opus)改用 ffmpeg。"""
    try:
        audio, rate = sf.read(str(path), dtype="float32", always_2d=True)
        return audio, int(rate)
    except Exception:
        return _read_via_ffmpeg(path)


def _read_via_ffmpeg(path: Path, samplerate: int = 44100,
                     channels: int = 2) -> tuple[np.ndarray, int]:
    """用 ffmpeg 解碼成 float32 PCM。"""
    import subprocess

    try:
        import imageio_ffmpeg
        exe = imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError as exc:
        raise RuntimeError(
            f"無法讀取 {path.name}:soundfile 不支援此格式,"
            "且找不到 imageio-ffmpeg。請執行 uv sync。") from exc

    proc = subprocess.run(
        [exe, "-hide_banner", "-loglevel", "error", "-i", str(path),
         "-vn", "-f", "f32le", "-acodec", "pcm_f32le",
         "-ac", str(channels), "-ar", str(samplerate), "-"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
        creationflags=0x08000000 if sys.platform == "win32" else 0)

    if proc.returncode != 0:
        detail = proc.stderr.decode("utf-8", "replace").strip()
        raise RuntimeError(f"ffmpeg 解碼 {path.name} 失敗:{detail[:200]}")

    data = np.frombuffer(proc.stdout, dtype="<f4")
    if data.size == 0:
        raise RuntimeError(f"{path.name} 解碼結果是空的")
    usable = (data.size // channels) * channels
    return np.ascontiguousarray(
        data[:usable].reshape(-1, channels), dtype=np.float32), samplerate


def find_stem(folder: Path, name: str) -> Path | None:
    for suffix in AUDIO_SUFFIXES:
        candidate = folder / f"{name}{suffix}"
        if candidate.exists():
            return candidate
    return None


class TestCase:
    """一個待評估的項目。"""

    def __init__(self, name: str, mixture: Path,
                 vocals: Path | None = None,
                 accompaniment: Path | None = None) -> None:
        self.name = name
        self.mixture = mixture
        self.vocals = vocals
        self.accompaniment = accompaniment

    @property
    def has_reference(self) -> bool:
        return self.vocals is not None


def discover(root: Path) -> list[TestCase]:
    """找出要評估的項目。

    支援三種結構:
      1. 單一音訊檔                → 只分離,無分數
      2. 資料夾含 mixture + vocals → 完整評估
      3. 上層資料夾含多個 (2)      → 批次評估
    """
    if root.is_file():
        return [TestCase(root.stem, root)]

    if not root.is_dir():
        raise FileNotFoundError(f"找不到:{root}")

    # 這個資料夾本身就是一個測試項目?
    mixture = find_stem(root, "mixture") or find_stem(root, "mix")
    if mixture:
        return [TestCase(root.name, mixture,
                         find_stem(root, "vocals"),
                         find_stem(root, "accompaniment")
                         or find_stem(root, "no_vocals"))]

    cases: list[TestCase] = []
    for child in sorted(root.iterdir()):
        if child.is_dir():
            cases.extend(discover(child))
        elif child.suffix.lower() in AUDIO_SUFFIXES:
            cases.append(TestCase(child.stem, child))
    return cases


def load_references(case: TestCase, samplerate: int,
                    length: int) -> dict[str, np.ndarray] | None:
    """讀入正確答案。缺伴奏軌時用 mixture - vocals 推導。"""
    if not case.has_reference:
        return None

    vocals, rate = read_audio(case.vocals)          # type: ignore[arg-type]
    if rate != samplerate:
        from .separate import _resample
        vocals = _resample(vocals, rate, samplerate)

    if case.accompaniment:
        accompaniment, acc_rate = read_audio(case.accompaniment)
        if acc_rate != samplerate:
            from .separate import _resample
            accompaniment = _resample(accompaniment, acc_rate, samplerate)
    else:
        mixture, mix_rate = read_audio(case.mixture)
        if mix_rate != samplerate:
            from .separate import _resample
            mixture = _resample(mixture, mix_rate, samplerate)
        n = min(len(mixture), len(vocals))
        accompaniment = mixture[:n] - vocals[:n]

    n = min(length, len(vocals), len(accompaniment))
    return {"vocals": vocals[:n], "accompaniment": accompaniment[:n]}


def evaluate_one(case: TestCase, args) -> dict:
    audio, rate = read_audio(case.mixture)
    duration = len(audio) / rate

    started = time.perf_counter()
    stems = separate(audio, rate, model_name=args.model, device=args.device,
                     two_stems=True, segment=args.segment,
                     progress=args.progress)
    elapsed = time.perf_counter() - started

    target_rate = model_info(args.model)["samplerate"]
    record: dict = {
        "name": case.name,
        "duration_s": round(duration, 2),
        "process_s": round(elapsed, 2),
        "realtime_factor": round(duration / elapsed, 2) if elapsed > 0 else None,
        "has_reference": case.has_reference,
    }

    if args.out:
        out_dir = Path(args.out) / case.name
        out_dir.mkdir(parents=True, exist_ok=True)
        for name, data in stems.items():
            sf.write(str(out_dir / f"{name}.wav"), data, target_rate)
        record["output"] = str(out_dir)

    references = load_references(case, target_rate,
                                 len(stems["vocals"]))
    if references:
        n = len(references["vocals"])
        estimates = {k: v[:n] for k, v in stems.items()}
        scores = evaluate_track(references, estimates, target_rate,
                                full=args.full)
        record["scores"] = scores

    return record


def format_table(records: list[dict]) -> str:
    """把結果排成好讀的表格。"""
    scored = [r for r in records if "scores" in r]
    if not scored:
        return ""

    metrics = ["si_sdr", "usdr", "csdr"]
    if any("sir" in r["scores"].get("vocals", {}) for r in scored):
        metrics += ["sir", "sar"]

    header = f"{'曲目':<28}" + "".join(f"{m:>9}" for m in metrics)
    lines = [header, "─" * len(header)]

    for record in scored:
        vocals = record["scores"].get("vocals", {})
        row = f"{record['name'][:27]:<28}"
        for metric in metrics:
            value = vocals.get(metric)
            row += f"{value:>9.2f}" if isinstance(value, (int, float)) \
                and np.isfinite(value) else f"{'—':>9}"
        lines.append(row)

    summary = summarize([r["scores"]["vocals"] for r in scored
                         if "vocals" in r["scores"]])
    if summary:
        lines.append("─" * len(header))
        median_row = f"{'中位數':<28}"
        for metric in metrics:
            stats = summary.get(metric)
            median_row += f"{stats['median']:>9.2f}" if stats else f"{'—':>9}"
        lines.append(median_row)

    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="ktisv_research.evaluate",
        description="評估人聲分離品質")
    parser.add_argument("input", type=Path,
                        help="音訊檔,或含 mixture/vocals 的資料夾")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default=None,
                        help="cuda / cpu(預設自動選)")
    parser.add_argument("--out", type=Path, default=None,
                        help="輸出分離結果的資料夾")
    parser.add_argument("--segment", type=float, default=None,
                        help="每段秒數;VRAM 不足時調小(6 GB 建議 5~7)")
    parser.add_argument("--full", action="store_true",
                        help="加算 SIR/SAR(較慢,但能區分失敗原因)")
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--json", type=Path, default=None,
                        help="把完整結果寫成 JSON")
    args = parser.parse_args(argv)

    try:
        cases = discover(args.input)
    except FileNotFoundError as exc:
        print(f"錯誤:{exc}")
        return 1

    if not cases:
        print(f"在 {args.input} 找不到任何音訊檔。")
        return 1

    device = args.device or available_device()
    info = model_info(args.model)
    print(f"模型 {args.model}  ·  裝置 {device}  ·  "
          f"{info['samplerate']} Hz  ·  音軌 {'/'.join(info['sources'])}")
    if device == "cpu":
        print("(CPU 模式 —— 大約需要曲長的數倍時間。要用 GPU 需改裝 CUDA 版 torch)")
    print(f"待評估:{len(cases)} 項\n")

    records: list[dict] = []
    for index, case in enumerate(cases, 1):
        print(f"[{index}/{len(cases)}] {case.name}"
              f"{'' if case.has_reference else '  (無正確答案,只輸出分離結果)'}")
        try:
            record = evaluate_one(case, args)
        except Exception as exc:
            print(f"    失敗:{exc}")
            records.append({"name": case.name, "error": str(exc)})
            continue

        detail = f"    {record['duration_s']:.0f} 秒音訊,耗時 " \
                 f"{record['process_s']:.1f} 秒"
        if record.get("realtime_factor"):
            detail += f"({record['realtime_factor']:.2f}× 即時)"
        print(detail)

        if "scores" in record:
            vocals = record["scores"].get("vocals", {})
            print("    人聲 " + "  ".join(
                f"{k}={v:.2f}" for k, v in vocals.items()
                if isinstance(v, (int, float)) and np.isfinite(v)))
        records.append(record)

    print()
    table = format_table(records)
    if table:
        print(table)
        print()
        print("SDR 越高越好。同樣的 SDR 下:")
        print("  SIR 低 → 沒濾乾淨(人聲軌裡還有伴奏)")
        print("  SAR 低 → 產生了原本不存在的假訊號")
    else:
        print("沒有可計算的分數 —— 所有項目都缺少正確答案(vocals 音軌)。")
        print("請試聽輸出的分離結果做主觀判斷。")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n完整結果:{args.json}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
