"""離線人聲分離(Demucs)。

兩條執行路徑,依序嘗試:

1. **行程內**(現在的主要路徑)—— 打包版內附 CPU 版 torch 與 demucs,
   直接呼叫 ``demucs.separate.main()``,不開子行程、不需要外部環境。
2. **外部直譯器**(退路)—— 由 ``KTISV_DEMUCS_PYTHON`` 或自動搜尋決定。
   留著它有兩個實際理由:使用者想用自己的 CUDA 環境換取速度
   (內建的是 CPU 版,一首歌要幾分鐘),或是內建的那份出了問題。

分離結果會依「檔案內容雜湊 + 模型名稱 + 參數」快取,所以慢只慢第一次。
"""

from __future__ import annotations

import hashlib
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from typing import Callable

import numpy as np

from ..paths import cache_dir
from . import ffmpeg

_CREATE_NO_WINDOW = 0x08000000 if os.name == "nt" else 0
_PROGRESS_RE = re.compile(r"(\d+)%")

DEFAULT_MODEL = "htdemucs"


class DemucsError(RuntimeError):
    pass


@dataclass
class Stems:
    """分離結果。取樣率一律轉成引擎的取樣率。"""

    vocals: np.ndarray
    instrumental: np.ndarray
    samplerate: int
    source: str = ""


def _is_frozen() -> bool:
    """是否為 PyInstaller 打包後的執行檔。"""
    return getattr(sys, "frozen", False)


_BUILTIN: tuple[bool, str] | None = None


def builtin_ready() -> tuple[bool, str]:
    """本行程自己能不能跑 demucs。回傳 (可用, 不可用的原因)。

    **只查有沒有,不真的 import。**

    這裡踩過一次坑:原本的實作直接 ``import torch`` 來判斷,而
    ``resolve_python()`` 會呼叫它,``session.status()`` 又會呼叫
    ``resolve_python()`` —— 於是前端每次輪詢狀態都可能觸發一次 torch 匯入。
    實測第一次要 **4.35 秒**(凍結版更久),那段時間整個 IPC 執行緒卡住,
    前端逾時,症狀是「載入音樂失敗」—— 跟 Demucs 看起來毫無關係。

    ``find_spec`` 只查模組在不在,不會執行 torch 的 __init__,所以幾乎不花時間。
    真正的匯入延到 ``_run_demucs_inproc()`` 要用的時候才做。
    """
    global _BUILTIN
    if _BUILTIN is not None:
        return _BUILTIN

    import importlib.util
    try:
        for mod in ("torch", "demucs"):
            if importlib.util.find_spec(mod) is None:
                _BUILTIN = (False, f"找不到 {mod}")
                return _BUILTIN
    except Exception as exc:                    # pragma: no cover - 視環境而定
        _BUILTIN = (False, f"{type(exc).__name__}: {exc}")
        return _BUILTIN

    _BUILTIN = (True, "")
    return _BUILTIN


def resolve_python() -> str:
    """決定要用哪個外部 Python 執行 demucs。空字串 = 走行程內。

    優先序:
      1. KTISV_DEMUCS_PYTHON —— 使用者明確指定(通常是為了 GPU 加速)
      2. 內建的 torch —— 回傳 "" 讓呼叫端走行程內,這是一般情況
      3. 搜尋已知的外部環境 —— 內建的壞掉時的退路

    打包後 sys.executable 是引擎 exe 本身,不能拿來跑 `-c`,所以 frozen 狀態下
    絕不退回 sys.executable —— 找不到專用直譯器就回傳空字串。
    """
    explicit = os.environ.get("KTISV_DEMUCS_PYTHON")
    if explicit and os.path.isfile(explicit):
        return explicit

    # 內建的能用就用內建的。開子行程只是白白多一份 torch 的載入時間。
    if builtin_ready()[0]:
        return ""

    if _is_frozen():
        # 打包版把 exe 放在 dist/ktisv-engine/,往上找使用者放的 demucs 環境
        base = os.path.dirname(sys.executable)
        candidates = [
            os.path.join(base, ".venv-demucs", "Scripts", "python.exe"),
            os.path.join(base, "..", ".venv-demucs", "Scripts", "python.exe"),
        ]
    else:
        here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        root = os.path.dirname(here)          # engine/ 的上一層
        candidates = [
            # 專用的 demucs 環境優先
            os.path.join(here, ".venv-demucs", "Scripts", "python.exe"),
            os.path.join(here, ".venv-demucs", "bin", "python"),
            # research/ 的環境已經裝了 torch + demucs,直接借用
            os.path.join(root, "research", ".venv", "Scripts", "python.exe"),
            os.path.join(root, "research", ".venv", "bin", "python"),
            # 引擎自己的環境(通常沒有 torch,但還是試一下)
            os.path.join(here, ".venv", "Scripts", "python.exe"),
            os.path.join(here, ".venv", "bin", "python"),
        ]

    for candidate in candidates:
        if os.path.isfile(candidate):
            return os.path.abspath(candidate)

    return "" if _is_frozen() else sys.executable


def set_python(path: str) -> None:
    if path:
        os.environ["KTISV_DEMUCS_PYTHON"] = path
    else:
        os.environ.pop("KTISV_DEMUCS_PYTHON", None)


def probe(python_exe: str | None = None) -> dict:
    """檢查指定直譯器能不能跑 demucs。"""
    exe = python_exe or resolve_python()
    if not exe:
        ok, why = builtin_ready()
        if ok:
            import torch
            import demucs
            return {"available": True, "python": "(內建)",
                    "torch": torch.__version__,
                    "demucs": getattr(demucs, "__version__", "?"),
                    "cuda": bool(torch.cuda.is_available()),
                    "builtin": True}
        return {"available": False, "python": "",
                "reason": f"內建的 Demucs 無法載入({why})。可另外建一個裝了 "
                          "demucs 的環境,在設定填入其 python.exe 路徑,"
                          "或設定 KTISV_DEMUCS_PYTHON。"}
    if not os.path.isfile(exe):
        return {"available": False, "python": exe, "reason": "找不到直譯器"}
    try:
        proc = subprocess.run(
            [exe, "-c",
             "import torch, demucs;"
             "print(torch.__version__);"
             "print(getattr(demucs,'__version__','?'));"
             "print(torch.cuda.is_available())"],
            capture_output=True, text=True, timeout=90,
            creationflags=_CREATE_NO_WINDOW, check=False,
        )
    except Exception as exc:
        return {"available": False, "python": exe, "reason": str(exc)}

    if proc.returncode != 0:
        reason = (proc.stderr or "").strip().splitlines()
        return {
            "available": False,
            "python": exe,
            "reason": reason[-1] if reason else "無法匯入 torch / demucs",
        }

    lines = proc.stdout.strip().splitlines()
    return {
        "available": True,
        "python": exe,
        "torch": lines[0] if lines else "?",
        "demucs": lines[1] if len(lines) > 1 else "?",
        "cuda": (lines[2].strip().lower() == "true") if len(lines) > 2 else False,
    }


def _file_key(path: str, model: str) -> str:
    h = hashlib.sha1()
    h.update(model.encode("utf-8"))
    stat = os.stat(path)
    h.update(str(stat.st_size).encode("utf-8"))
    with open(path, "rb") as fh:
        h.update(fh.read(1 << 20))
        if stat.st_size > (1 << 21):
            fh.seek(-(1 << 20), os.SEEK_END)
            h.update(fh.read(1 << 20))
    return h.hexdigest()[:20]


def separate(path: str,
             samplerate: int = 48000,
             model: str = DEFAULT_MODEL,
             python_exe: str | None = None,
             device: str = "auto",
             shifts: int = 0,
             overlap: float = 0.25,
             progress: Callable[[str, float], None] | None = None,
             cancel: Callable[[], bool] | None = None) -> Stems:
    """把音訊檔分成人聲與伴奏兩軌。

    ``shifts``
        隨機位移平均(UVR 稱之為 shifts)。把輸入做 N 次不同的時間位移各跑一次
        再平均,能抵銷模型對特定時間對齊的偏好,通常換得零點幾 dB 的 SDR。
        **代價是時間線性增加** —— shifts=2 就是兩倍時間。0 表示關閉。

    ``overlap``
        分塊推論時相鄰塊的重疊比例。太低會在塊邊界產生接縫,太高則變慢。
        Demucs 預設 0.25,UVR 常用 0.5 換取更平滑的接縫。
    """
    exe = python_exe or resolve_python()
    # 參數會影響輸出,所以要進快取鍵 —— 否則改了參數還會拿到舊結果
    key = _file_key(path, f"{model}|s{shifts}|o{overlap:.2f}")
    out_root = cache_dir("stems", key)

    vocals_path, other_path = _locate_stems(out_root, model, path)

    if not (vocals_path and other_path):
        _run_demucs(path, out_root, model, exe, device, shifts, overlap,
                    progress, cancel)
        vocals_path, other_path = _locate_stems(out_root, model, path)

    if not (vocals_path and other_path):
        raise DemucsError(
            f"Demucs 執行完畢但找不到輸出音軌。已搜尋 {out_root} 底下的所有位置。")

    if progress:
        progress("載入分離結果", 0.95)

    vocals = ffmpeg.decode_to_array(vocals_path, samplerate, 2)
    instrumental = ffmpeg.decode_to_array(other_path, samplerate, 2)
    n = min(len(vocals), len(instrumental))

    if progress:
        progress("完成", 1.0)

    return Stems(vocals=vocals[:n], instrumental=instrumental[:n],
                 samplerate=samplerate, source=path)


def _demucs_args(out_root: str, model: str, device: str,
                 shifts: int, overlap: float) -> list[str]:
    """組 demucs 的命令列參數。行程內與子行程兩條路共用同一組,
    免得兩邊悄悄長出不同行為。"""
    args = ["-n", model,
            "--two-stems", "vocals",
            "-o", out_root,
            "--filename", "{stem}.{ext}",
            "--overlap", f"{max(0.0, min(0.9, float(overlap))):.2f}"]
    if shifts > 0:
        args += ["--shifts", str(int(shifts))]
    if device in ("cpu", "cuda"):
        args += ["-d", device]
    return args


class _ProgressTap:
    """接住 demucs 寫到 stderr 的進度條,轉成 progress 回呼。

    demucs 用 tqdm,進度只存在於它印出來的字串裡,沒有程式化的介面。
    行程內執行時 stderr 是我們自己的,所以直接換掉它來讀。
    """

    def __init__(self, progress, cancel) -> None:
        self._progress = progress
        self._cancel = cancel
        self.tail: list[str] = []
        # 防重入。stdout 也被導到這裡,而回呼常常會 print ——
        # 沒有這道鎖就是 write -> progress -> print -> write 的無限遞迴
        # (實測直接撞 RecursionError)。
        self._busy = False

    def write(self, text: str) -> int:
        if self._busy:
            return len(text)
        if text.strip():
            self.tail.append(text.strip())
            del self.tail[:-15]

        self._busy = True
        try:
            match = _PROGRESS_RE.search(text)
            if match and self._progress:
                self._progress("分離中",
                               min(int(match.group(1)) / 100.0 * 0.9, 0.9))
            # 取消只能在這裡動手 —— demucs 沒有取消鉤子,而它會持續寫進度,
            # 所以從寫入端丟例外是唯一能中斷推論迴圈的地方。
            if self._cancel and self._cancel():
                raise DemucsError("已取消分離。")
        finally:
            self._busy = False
        return len(text)

    def flush(self) -> None:
        pass

    def isatty(self) -> bool:
        return False


def _run_demucs_inproc(path: str, args: list[str], progress, cancel) -> None:
    """在本行程直接呼叫 demucs。"""
    import contextlib
    from demucs.separate import main as demucs_main

    if progress:
        progress("啟動 Demucs(內建)", 0.0)

    tap = _ProgressTap(progress, cancel)
    try:
        with contextlib.redirect_stderr(tap), contextlib.redirect_stdout(tap):
            demucs_main([*args, path])
    except DemucsError:
        raise
    except SystemExit as exc:
        # demucs 的 CLI 進入點在參數有問題時會呼叫 sys.exit()
        if exc.code not in (0, None):
            detail = "\n".join(tap.tail[-6:]) or f"結束碼 {exc.code}"
            raise DemucsError(f"Demucs 執行失敗:\n{detail}") from exc
    except Exception as exc:
        detail = "\n".join(tap.tail[-6:])
        raise DemucsError(
            f"Demucs 執行失敗:{type(exc).__name__}: {exc}"
            + (f"\n{detail}" if detail else "")) from exc


def _run_demucs(path: str, out_root: str, model: str, exe: str, device: str,
                shifts: int, overlap: float,
                progress: Callable[[str, float], None] | None,
                cancel: Callable[[], bool] | None) -> None:
    args = _demucs_args(out_root, model, device, shifts, overlap)

    # 沒有指定外部直譯器,而且內建的能用 -> 走行程內,不開子行程。
    if not exe and builtin_ready()[0]:
        _run_demucs_inproc(path, args, progress, cancel)
        return
    if not exe:
        ok, why = builtin_ready()
        raise DemucsError(f"沒有可用的 Demucs 環境。內建的無法載入:{why}")

    cmd = [exe, "-m", "demucs", *args, path]

    env = dict(os.environ)
    env.setdefault("PYTHONIOENCODING", "utf-8")

    if progress:
        progress("啟動 Demucs", 0.0)

    proc = subprocess.Popen(
        cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
        text=True, encoding="utf-8", errors="replace",
        env=env, creationflags=_CREATE_NO_WINDOW,
    )

    tail: list[str] = []
    assert proc.stderr is not None
    for raw in proc.stderr:
        line = raw.rstrip()
        if line:
            tail.append(line)
            del tail[:-15]
        match = _PROGRESS_RE.search(line)
        if match and progress:
            progress("分離中", min(int(match.group(1)) / 100.0 * 0.9, 0.9))
        if cancel and cancel():
            proc.terminate()
            raise DemucsError("已取消分離。")

    code = proc.wait()
    if code != 0:
        detail = "\n".join(tail[-6:]) or f"結束碼 {code}"
        raise DemucsError(f"Demucs 執行失敗:\n{detail}")


def _locate_stems(out_root: str, model: str, source: str) -> tuple[str | None, str | None]:
    """找出人聲與伴奏檔。

    Demucs 的輸出結構會隨參數改變:給了 ``--filename "{stem}.{ext}"`` 時
    檔案直接放在 ``<out>/<model>/``,沒給時則多一層曲名資料夾。不同版本
    也可能不同。與其假設某一種結構,不如把可能的位置都找過。
    """
    track = os.path.splitext(os.path.basename(source))[0]
    candidates = [
        os.path.join(out_root, model),              # --filename 時的位置
        os.path.join(out_root, model, track),       # 預設結構
        out_root,
    ]

    # 再加上 model 底下實際存在的子資料夾(涵蓋曲名被改寫的情況)
    model_dir = os.path.join(out_root, model)
    if os.path.isdir(model_dir):
        candidates.extend(
            os.path.join(model_dir, name) for name in os.listdir(model_dir)
            if os.path.isdir(os.path.join(model_dir, name)))

    for folder in candidates:
        vocals = _find_stem(folder, "vocals")
        other = _find_stem(folder, "no_vocals")
        if vocals and other:
            return vocals, other
    return None, None


def _find_stem(stem_dir: str, name: str) -> str | None:
    if not os.path.isdir(stem_dir):
        return None
    for ext in (".wav", ".flac", ".mp3", ".m4a"):
        candidate = os.path.join(stem_dir, name + ext)
        if os.path.isfile(candidate):
            return candidate
    return None
