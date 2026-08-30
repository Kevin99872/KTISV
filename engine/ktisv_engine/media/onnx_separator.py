"""自訓模型的離線分離(ONNX Runtime)。

跟 Demucs 那條路平行,差別在模型是你自己訓的。研究端的
``research/ktisv_research/export.py`` 負責把檢查點匯出成這裡吃得下的 ONNX。

為什麼要切塊
------------
``export.py`` 刻意把片段長度**固定**(不設動態軸),理由寫在它的 docstring:
這個 U-Net 的尺寸對齊邏輯沒辦法安全地匯成動態圖。所以整首歌得由這一端
切塊、重疊相加。片段長度不寫死在這裡 —— 直接從 ONNX 的輸入形狀讀出來,
換了模型也不用改程式。

為什麼要重疊
------------
分塊推論的邊界會有接縫:每一塊的頭尾缺少上下文,模型在那裡的判斷最差。
直接首尾相接會聽到規律的「咔、咔」。做法是讓相鄰塊重疊一半,各自乘上
升降斜坡再相加 —— 接縫被斜坡抹平,而且兩塊的權重恆為 1,不會有音量起伏。

模型放哪
--------
``%LOCALAPPDATA%\\KTISV\\models\\*.onnx``。訓練完把檔案丟進去就會出現在
選單裡,不需要重新打包。
"""

from __future__ import annotations

import os
from typing import Callable

import numpy as np

from ..paths import app_data_dir
from . import ffmpeg
from .separator import Stems


class OnnxSeparatorError(RuntimeError):
    pass


def models_dir() -> str:
    path = os.path.join(app_data_dir(), "models")
    os.makedirs(path, exist_ok=True)
    return path


def list_models() -> list[dict]:
    """列出可用的自訓模型。"""
    out: list[dict] = []
    try:
        names = sorted(os.listdir(models_dir()))
    except OSError:
        return out
    for name in names:
        if not name.lower().endswith(".onnx"):
            continue
        full = os.path.join(models_dir(), name)
        try:
            size = os.path.getsize(full)
        except OSError:
            continue
        out.append({"name": os.path.splitext(name)[0], "path": full, "bytes": size})
    return out


def available() -> tuple[bool, str]:
    """能不能用。回傳 (可用, 不可用的原因)。

    跟 separator.builtin_ready() 一樣只查模組在不在,不真的 import ——
    這個函式會被 status() 呼叫,不能在那條路上付匯入的代價。
    """
    import importlib.util
    if importlib.util.find_spec("onnxruntime") is None:
        return False, "找不到 onnxruntime"
    if not list_models():
        return False, f"{models_dir()} 裡沒有 .onnx 模型"
    return True, ""


def _make_session(model_path: str):
    import onnxruntime as ort

    opts = ort.SessionOptions()
    # 這是離線工作,但引擎同時還在跑音訊回呼。放任 ORT 吃滿所有核心會
    # 讓音訊執行緒搶不到 CPU,聽起來就是分離時破音。留一半給音訊。
    threads = max(1, (os.cpu_count() or 2) // 2)
    opts.intra_op_num_threads = threads
    opts.inter_op_num_threads = 1
    return ort.InferenceSession(model_path, opts,
                                providers=["CPUExecutionProvider"])


def _segment_length(session) -> int:
    """從 ONNX 的輸入形狀讀出片段取樣數。"""
    shape = session.get_inputs()[0].shape
    length = shape[-1]
    if not isinstance(length, int) or length <= 0:
        raise OnnxSeparatorError(
            f"這個模型的輸入長度是動態的({shape}),但 export.py 應該匯出固定長度。")
    return length


def separate(path: str,
             model_path: str,
             samplerate: int = 48000,
             progress: Callable[[str, float], None] | None = None,
             cancel: Callable[[], bool] | None = None) -> Stems:
    """用自訓的 ONNX 模型分離。回傳跟 Demucs 那條路相同的 Stems。"""
    if not os.path.isfile(model_path):
        raise OnnxSeparatorError(f"找不到模型:{model_path}")

    if progress:
        progress("載入模型", 0.02)
    try:
        session = _make_session(model_path)
    except Exception as exc:
        raise OnnxSeparatorError(f"無法載入模型:{exc}") from exc

    seg = _segment_length(session)
    in_name = session.get_inputs()[0].name
    channels = session.get_inputs()[0].shape[1]
    channels = channels if isinstance(channels, int) else 2

    if progress:
        progress("解碼音訊", 0.05)
    # 模型是用某個取樣率訓練的,而 export.py 預設 44100。先照模型的取樣率
    # 推論,最後再轉成引擎要的取樣率 —— 不能拿 48k 的資料餵給 44.1k 訓的模型。
    model_sr = 44100
    audio = ffmpeg.decode_to_array(path, model_sr, channels)   # (n, ch)
    total = len(audio)
    if total == 0:
        raise OnnxSeparatorError("解碼後沒有音訊。")

    hop = seg // 2                      # 重疊一半
    # 升降斜坡。相鄰兩塊的斜坡相加恆為 1,所以重疊區不會有音量起伏。
    ramp = np.linspace(0.0, 1.0, hop, endpoint=False, dtype=np.float32)
    window = np.concatenate([ramp, 1.0 - ramp])[:, None]

    padded = total + seg
    acc_v = np.zeros((padded, channels), dtype=np.float32)
    acc_i = np.zeros((padded, channels), dtype=np.float32)
    weight = np.zeros((padded, 1), dtype=np.float32)

    starts = list(range(0, total, hop))
    last = len(starts) - 1
    for n, start in enumerate(starts):
        if cancel and cancel():
            raise OnnxSeparatorError("已取消分離。")

        # 頭尾不做淡入淡出。
        #
        # 中間的每個取樣都被兩塊蓋到,斜坡相加為 1,除回去剛好還原。但最前面
        # 的 hop 個取樣只有第一塊蓋到,權重就是那條從 0 開始的斜坡 —— 除以
        # 一個趨近 0 的數,float32 的誤差被放大。實測直通測試在開頭 100 個
        # 取樣的誤差達 0.132(接縫處反而是 0)。
        # 讓第一塊的前半段與最後一塊的後半段維持 1.0,那些位置的權重就不會
        # 趨近 0,誤差回到 1e-7 量級。
        w = window
        if n == 0 or n == last:
            w = window.copy()
            if n == 0:
                w[:hop] = 1.0
            if n == last:
                w[hop:] = 1.0
        chunk = np.zeros((seg, channels), dtype=np.float32)
        take = min(seg, total - start)
        chunk[:take] = audio[start:start + take]

        # ONNX 要 (batch, channel, sample)
        x = np.ascontiguousarray(chunk.T[None, ...], dtype=np.float32)
        try:
            voc, acc = session.run(None, {in_name: x})[:2]
        except Exception as exc:
            raise OnnxSeparatorError(f"推論失敗:{exc}") from exc

        v = np.ascontiguousarray(voc[0].T)      # (seg, ch)
        a = np.ascontiguousarray(acc[0].T)
        acc_v[start:start + seg] += v * w
        acc_i[start:start + seg] += a * w
        weight[start:start + seg] += w

        if progress:
            progress("分離中", 0.05 + (n + 1) / len(starts) * 0.85)

    # 頭尾各只有一塊覆蓋,權重不是 1,要除回來
    np.maximum(weight, 1e-6, out=weight)
    vocals = (acc_v[:total] / weight[:total]).astype(np.float32)
    instrumental = (acc_i[:total] / weight[:total]).astype(np.float32)

    if samplerate != model_sr:
        if progress:
            progress("重取樣", 0.93)
        vocals = _resample(vocals, model_sr, samplerate)
        instrumental = _resample(instrumental, model_sr, samplerate)

    if progress:
        progress("完成", 1.0)
    n = min(len(vocals), len(instrumental))
    return Stems(vocals=vocals[:n], instrumental=instrumental[:n],
                 samplerate=samplerate, source=path)


def _resample(x: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    """線性內插重取樣。

    這裡用線性內插是可以接受的:44.1k -> 48k 的比例接近 1,而且結果馬上
    要進播放器與後續的 DSP,不是最終母帶。要更好就得引入 polyphase,
    那對這一步的收益不成比例。
    """
    n_out = int(round(len(x) * dst_sr / src_sr))
    if n_out <= 0:
        return x[:0]
    pos = np.linspace(0, len(x) - 1, n_out, dtype=np.float64)
    lo = np.floor(pos).astype(np.int64)
    hi = np.minimum(lo + 1, len(x) - 1)
    frac = (pos - lo).astype(np.float32)[:, None]
    return (x[lo] * (1.0 - frac) + x[hi] * frac).astype(np.float32)
