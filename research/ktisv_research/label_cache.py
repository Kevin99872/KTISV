"""把 KTISV 的執行期快取變成訓練資料 —— 對 htdemucs 做知識蒸餾。

    python -m ktisv_research.label_cache

為什麼要這一步
--------------
``DATASETS.md`` 說「從影片網站抓的音檔無法當訓練資料」—— 那句話的前提是
**沒有正確答案**。但 KTISV 自己就內建 htdemucs,把它的輸出當標籤,那批
音檔就變成配對資料了。這是知識蒸餾:htdemucs 當老師,我們訓練一個小得多、
能即時跑的學生。

代價要講清楚,不然會對結果有錯誤期待:

1. **品質上限就是 htdemucs**。學生模型不可能超過老師,只能逼近它,
   而且是用少量資料逼近 —— 目標是「接近的品質、可即時的成本」,
   不是「更好的分離」。
2. **標籤帶著老師的假象**。htdemucs 漏掉的和聲、殘留的鼓聲,學生會照學。
3. **權重不可散布**。標籤衍生自影片網站的版權音源,訓練出來的權重同樣是
   衍生物。自用與驗證管線沒問題,要跟 KTISV 一起發佈就得換成
   ``DATASETS.md`` 裡授權明確的資料重訓。

輸出版面刻意配合 ``data.load_pair_folders``:每首歌一個資料夾,裡面是
``vocals.wav`` 與 ``accompaniment.wav``。所以標完之後訓練端不必知道
資料是從哪來的。
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from . import SAMPLE_RATE

DEFAULT_MODEL = "htdemucs"

AUDIO_SUFFIXES = {".webm", ".m4a", ".mp4", ".opus", ".mp3", ".wav", ".flac"}


def cache_root() -> Path:
    """KTISV 執行期快取的位置(與 README「設定與快取」那一節一致)。"""
    base = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
    return Path(base) / "KTISV" / "cache"


# ── 來源掃描 ────────────────────────────────────────────────────────────
def find_downloads(root: Path) -> list[Path]:
    """下載快取裡的音檔。副檔名不設限死 —— yt-dlp 給什麼就是什麼。"""
    folder = root / "downloads"
    if not folder.is_dir():
        return []
    return sorted(p for p in folder.iterdir()
                  if p.is_file() and p.suffix.lower() in AUDIO_SUFFIXES)


def find_existing_stems(root: Path) -> list[tuple[str, Path, Path]]:
    """已經跑過 Demucs 的結果,直接拿來用,不必再算一次。

    版面是 ``stems/<key>/<model>/{vocals,no_vocals}.wav``。
    """
    folder = root / "stems"
    if not folder.is_dir():
        return []

    found: list[tuple[str, Path, Path]] = []
    for key in sorted(p for p in folder.iterdir() if p.is_dir()):
        for model in sorted(p for p in key.iterdir() if p.is_dir()):
            vocals = model / "vocals.wav"
            other = model / "no_vocals.wav"
            if vocals.exists() and other.exists():
                found.append((f"stems-{key.name}", vocals, other))
                break
    return found


# ── 讀寫 ────────────────────────────────────────────────────────────────
def read_stereo(path: Path, samplerate: int,
                max_seconds: float | None) -> np.ndarray:
    """讀成 (samples, 2) 的 float32 @ samplerate,最多 ``max_seconds`` 秒。

    長度上限交給 ffmpeg 的 ``-t`` 而不是讀完再切:快取裡有一個 400 MB 的
    長檔,整個解碼要好幾分鐘,而我們只要開頭那幾分鐘。
    """
    from .data import _as_stereo, _read

    if max_seconds:
        trimmed = _decode_head(path, samplerate, max_seconds)
        if trimmed is not None:
            return _as_stereo(trimmed)

    audio = _as_stereo(_read(path, samplerate))
    if max_seconds:
        limit = int(max_seconds * samplerate)
        if len(audio) > limit:
            audio = audio[:limit]
    return audio


def _decode_head(path: Path, samplerate: int,
                 seconds: float) -> np.ndarray | None:
    """用 ffmpeg 只解碼開頭 ``seconds`` 秒。失敗時回傳 None 讓呼叫端退回一般路徑。"""
    import subprocess
    import sys

    try:
        import imageio_ffmpeg
        exe = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None

    proc = subprocess.run(
        [exe, "-hide_banner", "-loglevel", "error", "-t", f"{seconds:.3f}",
         "-i", str(path), "-vn", "-f", "f32le", "-acodec", "pcm_f32le",
         "-ac", "2", "-ar", str(samplerate), "-"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
        creationflags=0x08000000 if sys.platform == "win32" else 0)

    if proc.returncode != 0 or not proc.stdout:
        return None
    data = np.frombuffer(proc.stdout, dtype="<f4")
    usable = (data.size // 2) * 2
    if usable == 0:
        return None
    return np.ascontiguousarray(data[:usable].reshape(-1, 2), dtype=np.float32)


def write_pair(folder: Path, vocals: np.ndarray, accompaniment: np.ndarray,
               samplerate: int) -> float:
    """寫成 load_pair_folders 讀得懂的版面。回傳實際寫入的秒數。

    存 16-bit PCM 而不是 float32:磁碟省一半,而量化噪訊在 -96 dBFS,
    比 htdemucs 自己的分離假象低了幾十 dB —— 瓶頸不在這裡。
    """
    folder.mkdir(parents=True, exist_ok=True)
    n = min(len(vocals), len(accompaniment))
    sf.write(folder / "vocals.wav", vocals[:n], samplerate, subtype="PCM_16")
    sf.write(folder / "accompaniment.wav", accompaniment[:n], samplerate,
             subtype="PCM_16")
    return n / samplerate


# ── Demucs ──────────────────────────────────────────────────────────────
def build_separator(model: str, device: str | None, segment: float | None):
    """建立 demucs.api.Separator。模型權重第一次用會自己下載。"""
    import torch
    from demucs.api import Separator

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    kwargs: dict = {"model": model, "device": device, "progress": False}
    if segment:
        kwargs["segment"] = segment
    return Separator(**kwargs), device


def separate_two_stems(separator, audio: np.ndarray,
                       samplerate: int) -> tuple[np.ndarray, np.ndarray]:
    """回傳 (人聲, 伴奏),都是 (samples, 2)。

    ``demucs.api`` 沒有 CLI 的 ``--two-stems``,所以自己把 vocals 以外的
    分軌加回去 —— 這正是 CLI 在做的事,結果等價。
    """
    import torch

    tensor = torch.from_numpy(audio.T.copy())          # (channels, samples)
    _, stems = separator.separate_tensor(tensor, samplerate)

    vocals = stems["vocals"]
    others = [t for name, t in stems.items() if name != "vocals"]
    accompaniment = (sum(others[1:], others[0]) if others
                     else torch.zeros_like(vocals))

    return (vocals.cpu().numpy().T.astype(np.float32),
            accompaniment.cpu().numpy().T.astype(np.float32))


# ── 主流程 ──────────────────────────────────────────────────────────────
def run(args: argparse.Namespace) -> int:
    root = Path(args.cache) if args.cache else cache_root()
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    if not root.is_dir():
        print(f"找不到快取目錄 {root} —— KTISV 還沒下載過任何東西?")
        return 1

    manifest_path = out_root / "manifest.json"
    manifest: dict[str, dict] = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text("utf-8"))

    def record(name: str, seconds: float, source: str) -> None:
        manifest[name] = {"seconds": round(seconds, 2), "source": source}
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), "utf-8")

    max_seconds = args.max_minutes * 60 if args.max_minutes else None

    # 1. 已經分離過的直接搬過來 —— 不必再算一次
    existing = find_existing_stems(root)
    for name, vocals_path, other_path in existing:
        folder = out_root / name
        if folder.exists() and not args.force:
            continue
        vocals = read_stereo(vocals_path, SAMPLE_RATE, max_seconds)
        accompaniment = read_stereo(other_path, SAMPLE_RATE, max_seconds)
        seconds = write_pair(folder, vocals, accompaniment, SAMPLE_RATE)
        record(name, seconds, "cache/stems")
        print(f"  搬移 {name}  {seconds:.0f}s")

    # 2. 沒有標籤的下載檔,跑 demucs
    downloads = find_downloads(root)
    todo = [p for p in downloads
            if args.force or not (out_root / f"dl-{p.stem}").exists()]

    print(f"\n快取:{len(downloads)} 個下載檔、{len(existing)} 組現成分軌")
    print(f"待標註:{len(todo)} 個\n")

    if not todo:
        print("沒有需要標註的檔案。")
        return summarise(manifest, out_root)

    separator, device = build_separator(args.model, args.device, args.segment)
    print(f"demucs {args.model} @ {device}(每首上限 {args.max_minutes} 分鐘)\n")

    for index, path in enumerate(todo, 1):
        name = f"dl-{path.stem}"
        started = time.time()
        try:
            audio = read_stereo(path, SAMPLE_RATE, max_seconds)
        except Exception as exc:
            print(f"[{index}/{len(todo)}] {path.name} 解碼失敗,跳過:{exc}")
            continue

        minutes = len(audio) / SAMPLE_RATE / 60
        print(f"[{index}/{len(todo)}] {path.name}  {minutes:.1f} 分鐘 ...",
              end="", flush=True)
        try:
            vocals, accompaniment = separate_two_stems(
                separator, audio, SAMPLE_RATE)
        except Exception as exc:
            print(f" 失敗:{type(exc).__name__}: {exc}")
            continue

        seconds = write_pair(out_root / name, vocals, accompaniment, SAMPLE_RATE)
        record(name, seconds, str(path))
        print(f" 完成({time.time() - started:.0f}s)")

    return summarise(manifest, out_root)


def summarise(manifest: dict[str, dict], out_root: Path) -> int:
    total = sum(entry["seconds"] for entry in manifest.values())
    print(f"\n共 {len(manifest)} 首 / {total / 60:.1f} 分鐘 → {out_root}")
    if total < 600:
        print("⚠ 不到 10 分鐘 —— 這個量只夠驗證管線,訓不出能用的模型。")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="用 htdemucs 把 KTISV 快取標成訓練資料")
    parser.add_argument("--cache", default=None,
                        help="KTISV 快取目錄(預設 LOCALAPPDATA\\KTISV\\cache)")
    parser.add_argument("--out", default="data/ktisv-cache",
                        help="輸出目錄(預設 data/ktisv-cache)")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default=None, help="cuda / cpu(預設自動)")
    parser.add_argument("--segment", type=float, default=None,
                        help="demucs 分段秒數;VRAM 不夠時調小")
    parser.add_argument("--max-minutes", type=float, default=12.0,
                        help="每首最多取幾分鐘,避免單一長檔佔滿資料集")
    parser.add_argument("--force", action="store_true", help="重做已完成的項目")
    return run(parser.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
