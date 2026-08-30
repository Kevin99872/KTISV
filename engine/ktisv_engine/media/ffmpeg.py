"""ffmpeg 定位與音訊解碼。

不要求使用者自行安裝 ffmpeg:先找 PATH,再找 ``imageio-ffmpeg`` 內附的
靜態執行檔,最後才看環境變數指定的路徑。
"""

from __future__ import annotations

import functools
import glob
import os
import shutil
import subprocess
import sys

import numpy as np

_CREATE_NO_WINDOW = 0x08000000 if os.name == "nt" else 0


class FfmpegNotFound(RuntimeError):
    pass


def _bundled_ffmpeg() -> str | None:
    """打包版:直接在 PyInstaller 解壓出來的目錄裡找 ffmpeg。

    不先問 imageio_ffmpeg 是有原因的 —— 它靠
    ``importlib.resources.files("imageio_ffmpeg.binaries")`` 定位執行檔,
    凍結之後那條路很容易斷(少收一個 __init__.py 就找不到),而且斷掉時
    它不會拋錯,而是安靜地退回「系統 ffmpeg」再失敗。exe 就在我們自己
    打包進去的位置,自己找比較可靠。
    """
    base = getattr(sys, "_MEIPASS", None)
    if not base:
        return None
    pattern = os.path.join(base, "imageio_ffmpeg", "binaries", "ffmpeg*")
    for candidate in sorted(glob.glob(pattern)):
        if os.path.isfile(candidate) and not candidate.lower().endswith(".md"):
            return candidate
    return None


@functools.lru_cache(maxsize=1)
def ffmpeg_path() -> str:
    explicit = os.environ.get("KTISV_FFMPEG")
    if explicit and os.path.isfile(explicit):
        return explicit

    found = shutil.which("ffmpeg")
    if found:
        return found

    bundled = _bundled_ffmpeg()
    if bundled:
        return bundled

    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        pass

    raise FfmpegNotFound(
        "找不到 ffmpeg。請執行 `pip install imageio-ffmpeg`,"
        "或安裝 ffmpeg 後加入 PATH,或設定環境變數 KTISV_FFMPEG 指向 ffmpeg.exe。"
    )


def have_ffmpeg() -> bool:
    try:
        ffmpeg_path()
        return True
    except FfmpegNotFound:
        return False


def decode_to_array(path: str, samplerate: int = 48000,
                    channels: int = 2) -> np.ndarray:
    """把任意音訊/影片檔解碼成 (frames, channels) 的 float32 陣列。"""
    cmd = [
        ffmpeg_path(),
        "-hide_banner", "-loglevel", "error",
        "-i", path,
        "-vn",
        "-f", "f32le",
        "-acodec", "pcm_f32le",
        "-ac", str(channels),
        "-ar", str(samplerate),
        "-",
    ]
    proc = subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        creationflags=_CREATE_NO_WINDOW, check=False,
    )
    if proc.returncode != 0:
        err = proc.stderr.decode("utf-8", "replace").strip()
        raise RuntimeError(f"ffmpeg 解碼失敗: {err or proc.returncode}")

    audio = np.frombuffer(proc.stdout, dtype="<f4")
    if audio.size == 0:
        raise RuntimeError("解碼結果是空的音訊。")
    usable = (audio.size // channels) * channels
    return np.ascontiguousarray(audio[:usable].reshape(-1, channels), dtype=np.float32)


def encode_wav(path: str, audio: np.ndarray, samplerate: int) -> None:
    """把 float32 陣列寫成 16-bit WAV(給 Demucs 當輸入用)。"""
    import wave

    clipped = np.clip(audio, -1.0, 1.0)
    pcm = (clipped * 32767.0).astype("<i2")
    with wave.open(path, "wb") as wf:
        wf.setnchannels(audio.shape[1])
        wf.setsampwidth(2)
        wf.setframerate(samplerate)
        wf.writeframes(pcm.tobytes())
