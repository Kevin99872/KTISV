"""把公開資料集整理成訓練用的分軌資料夾。

    python -m ktisv_tf.prepare musdb    data/raw/musdb18hq.zip   --out data/open
    python -m ktisv_tf.prepare gtsinger data/raw/gtsinger        --out data/open

輸出的版面和 ``label_cache`` 一樣是 ``<out>/<歌>/vocals.wav (+ accompaniment.wav)``,
``ktisv_tf.data.load_tracks`` 直接讀得懂 —— 訓練端不必知道資料從哪來。

和 htdemucs 標籤的根本差別
--------------------------
``label_cache`` 的答案是 htdemucs 猜的,學生的上限被老師鎖死。這裡的答案是
**錄音時就分開的分軌**,是真的正確答案。

MUSDB18 / MUSDB18-HQ
--------------------
直接從 zip 裡讀,不必先解開。伴奏 = drums + bass + other。兩種版本都吃:
HQ 版是每軌一個 wav;壓縮版(4.7 GB,Zenodo 對單一 IP 限速時下載快得多)
是每首一個 ``.stem.mp4``,五條 AAC 音軌,用 ffmpeg 解。Open-Unmix 的預設
訓練資料就是壓縮版,256 kbps AAC 對分離訓練夠用。
官方切分照搬:``train`` 裡的 14 首驗證曲另外標成 ``musdb-valid-*``,
``test`` 的 50 首標成 ``musdb-test-*``,留給最終評估、訓練時要 ``--exclude``。

GTSinger
--------
只有人聲(乾錄音),沒有伴奏。同一位歌手、同一首歌、同一組(technique /
control)的片段依序接成一首,中間留 0.3 秒空白,寫成單聲道 44.1 kHz,
並放一個 ``solo.txt`` 標記 —— 訓練時一律配上別首歌的伴奏(見
``data.sample_once``)。說話錄音(paired speech)跳過:那不是歌聲。

授權
----
MUSDB18 僅限教育/非商業;GTSinger 是 CC BY-NC-SA 4.0 外加使用條款。
用這些資料訓練的權重**只能自用或研究**,不能隨 KTISV 散布。
"""

from __future__ import annotations

import argparse
import hashlib
import io
import re
import sys
import zipfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf

SAMPLE_RATE = 44100

# sigsep/musdb 的官方驗證曲(mus.yaml 的 validation_tracks)
MUSDB_VALID = {
    "Actions - One Minute Smile", "Clara Berry And Wooldog - Waltz For My Victims",
    "Johnny Lokke - Promises & Lies", "Patrick Talbot - A Reason To Leave",
    "Triviul - Angelsaint", "Alexander Ross - Goodbye Bolero",
    "Fergessen - Nos Palpitants", "Leaf - Summerghost",
    "Skelpolu - Human Mistakes", "Young Griffo - Pennies", "ANiMAL - Rockshow",
    "James May - On The Line", "Meaxic - Take A Step",
    "Traffic Experiment - Sirens",
}
# 壓縮版 .stem.mp4 裡的音軌順序
STEM_ORDER = ("mixture", "drums", "bass", "other", "vocals")


def slug(text: str) -> str:
    """資料夾名:ASCII、小寫、可當 --val-tracks 樣式用。"""
    ascii_part = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    if ascii_part:
        return ascii_part[:60]
    # 全中文的名字:用雜湊,避免 Windows 主控台與 tf.data 的編碼問題
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]


def write_pair(folder: Path, vocals: np.ndarray,
               accompaniment: np.ndarray | None) -> None:
    """寫成 16-bit PCM。超過 0 dBFS 時兩軌同步縮小,保持可加性。"""
    stems = [vocals] if accompaniment is None else [vocals, accompaniment]
    peak = max(float(np.max(np.abs(s))) for s in stems)
    gain = 0.999 / peak if peak > 0.999 else 1.0
    folder.mkdir(parents=True, exist_ok=True)
    sf.write(folder / "vocals.wav", vocals * gain, SAMPLE_RATE, subtype="PCM_16")
    if accompaniment is None:
        (folder / "solo.txt").write_text("vocals only\n", "utf-8")
    else:
        sf.write(folder / "accompaniment.wav", accompaniment * gain,
                 SAMPLE_RATE, subtype="PCM_16")


def resample(audio: np.ndarray, source: int) -> np.ndarray:
    if source == SAMPLE_RATE:
        return audio
    from scipy.signal import resample_poly

    d = gcd(source, SAMPLE_RATE)
    return resample_poly(audio, SAMPLE_RATE // d, source // d,
                         axis=0).astype(np.float32)


# ── MUSDB18-HQ ──────────────────────────────────────────────────────────
def prepare_musdb(source: Path, out: Path, force: bool) -> int:
    archive = zipfile.ZipFile(source) if source.suffix == ".zip" else None
    names = archive.namelist() if archive else \
        [p.relative_to(source).as_posix() for p in source.rglob("*.wav")]

    if not archive:
        names += [p.relative_to(source).as_posix()
                  for p in source.rglob("*.stem.mp4")]

    songs: dict[tuple[str, str], dict[str, str]] = defaultdict(dict)
    for name in names:
        parts = name.split("/")
        # HQ 版:…/train/<歌>/vocals.wav(容許 zip 多包一層資料夾)
        if len(parts) >= 3 and parts[-3] in ("train", "test") and parts[-1].endswith(".wav"):
            songs[(parts[-3], parts[-2])][parts[-1][:-4]] = name
        # 壓縮版:…/train/<歌>.stem.mp4,五條音軌包在同一個檔裡
        elif len(parts) >= 2 and parts[-2] in ("train", "test") \
                and parts[-1].endswith(".stem.mp4"):
            song = parts[-1][:-len(".stem.mp4")]
            songs[(parts[-2], song)] = {k: f"{name}#{i}" for i, k in
                                        enumerate(STEM_ORDER) if i}

    def read(name: str) -> np.ndarray:
        if "#" in name:
            return read_stem(name)
        if archive:
            data, rate = sf.read(io.BytesIO(archive.read(name)), dtype="float32",
                                 always_2d=True)
        else:
            data, rate = sf.read(source / name, dtype="float32", always_2d=True)
        return resample(data, rate)

    def read_stem(name: str) -> np.ndarray:
        """NI STEMS(mp4 裡的第 N 條 AAC 音軌)交給 ffmpeg 解碼。"""
        import subprocess
        import tempfile

        import imageio_ffmpeg

        member, index = name.rsplit("#", 1)
        with tempfile.TemporaryDirectory() as tmp:
            path = source / member
            if archive:
                path = Path(tmp) / "song.mp4"
                path.write_bytes(archive.read(member))
            proc = subprocess.run(
                [imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-loglevel",
                 "error", "-i", str(path), "-map", f"0:a:{index}",
                 "-f", "f32le", "-acodec", "pcm_f32le", "-ac", "2",
                 "-ar", str(SAMPLE_RATE), "-"],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        if proc.returncode != 0 or not proc.stdout:
            raise RuntimeError(f"ffmpeg 解不開 {name}:"
                               f"{proc.stderr.decode('utf-8', 'replace')[:200]}")
        return np.frombuffer(proc.stdout, "<f4").reshape(-1, 2).copy()

    count = 0
    for (split, song), stems in sorted(songs.items()):
        needed = {"vocals", "drums", "bass", "other"}
        if not needed <= stems.keys():
            print(f"  略過 {song}:缺 {needed - stems.keys()}")
            continue
        kind = "valid" if split == "train" and song in MUSDB_VALID else split
        folder = out / f"musdb-{kind}-{slug(song)}"
        if (folder / "accompaniment.wav").exists() and not force:
            count += 1
            continue
        vocals = read(stems["vocals"])
        backing = sum(read(stems[k]) for k in ("drums", "bass", "other"))
        n = min(len(vocals), len(backing))
        write_pair(folder, vocals[:n], backing[:n])
        (folder / "source.txt").write_text(f"MUSDB18 {split}/{song}\n", "utf-8")
        count += 1
        print(f"  [{count:>3}] {folder.name}", flush=True)
    missing = MUSDB_VALID - {s for (_, s) in songs}
    if missing:
        print(f"⚠ 找不到這些官方驗證曲:{sorted(missing)}")
    return count


# ── GTSinger ────────────────────────────────────────────────────────────
def prepare_gtsinger(source: Path, out: Path, force: bool,
                     gap: float = 0.3) -> int:
    languages = {"Chinese": "zh", "English": "en"}
    groups: dict[tuple, list[Path]] = defaultdict(list)
    for wav in source.rglob("*.wav"):
        rel = wav.relative_to(source).parts
        if rel[0] not in languages or any("speech" in p.lower() for p in rel):
            continue
        # <語言>/<歌手>/<技巧>/<歌名>/<組>/0000.wav
        groups[tuple(rel[:-1])].append(wav)

    def build(key: tuple, files: list[Path]) -> str | None:
        language, singer = languages[key[0]], key[1]
        folder = out / f"gts-{language}-{slug(singer)}-{slug('/'.join(key[2:]))}"
        if (folder / "vocals.wav").exists() and not force:
            return None
        pieces = []
        for wav in sorted(files):
            data, rate = sf.read(wav, dtype="float32", always_2d=True)
            pieces.append(resample(data.mean(axis=1, keepdims=True), rate))
            pieces.append(np.zeros((int(gap * SAMPLE_RATE), 1), np.float32))
        write_pair(folder, np.concatenate(pieces), None)
        (folder / "source.txt").write_text(
            "GTSinger " + "/".join(key) + "\n", "utf-8")
        return folder.name

    with ThreadPoolExecutor(max_workers=8) as pool:
        for index, name in enumerate(pool.map(lambda kv: build(*kv),
                                              sorted(groups.items())), 1):
            if name and index % 25 == 0:
                print(f"  [{index:>4}/{len(groups)}] {name}", flush=True)
    return len(groups)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="整理公開資料集成分軌資料夾")
    parser.add_argument("kind", choices=["musdb", "gtsinger"])
    parser.add_argument("source", help="musdb18.zip / musdb18hq.zip(或解開的目錄)"
                                       "/ GTSinger 根目錄")
    parser.add_argument("--out", default="data/open")
    parser.add_argument("--force", action="store_true", help="重做已存在的項目")
    args = parser.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.kind == "musdb":
        count = prepare_musdb(Path(args.source), out, args.force)
    else:
        count = prepare_gtsinger(Path(args.source), out, args.force)
    print(f"完成:{count} 首 → {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
