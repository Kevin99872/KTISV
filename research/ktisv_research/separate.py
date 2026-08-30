"""Demucs 推論包裝。

把 Demucs 的模型載入與分塊推論包成單純的「陣列進、分軌出」介面,
讓評估與訓練程式碼不必重複處理裝置、分塊、重疊等細節。
"""

from __future__ import annotations

import functools

import numpy as np

DEFAULT_MODEL = "htdemucs"


def available_device(prefer_gpu: bool = True) -> str:
    """挑一個可用的裝置。"""
    if not prefer_gpu:
        return "cpu"
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"


@functools.lru_cache(maxsize=4)
def load_model(name: str = DEFAULT_MODEL, device: str | None = None):
    """載入預訓練模型(有快取,重複呼叫不會重新載入)。"""
    try:
        from demucs.pretrained import get_model
    except ImportError as exc:
        raise ImportError(
            "需要 demucs。請執行:cd research && uv sync --extra torch"
        ) from exc

    device = device or available_device()
    model = get_model(name)
    model.to(device)
    model.eval()
    return model


def model_info(name: str = DEFAULT_MODEL) -> dict:
    model = load_model(name)
    return {
        "name": name,
        "sources": list(model.sources),
        "samplerate": int(model.samplerate),
        "channels": int(model.audio_channels),
    }


def separate(audio: np.ndarray, samplerate: int,
             model_name: str = DEFAULT_MODEL,
             device: str | None = None,
             two_stems: bool = True,
             overlap: float = 0.25,
             segment: float | None = None,
             progress: bool = False) -> dict[str, np.ndarray]:
    """分離一段音訊。

    ``audio``:``(samples, channels)`` 的 float 陣列。
    回傳 ``{"vocals", "accompaniment"}``(``two_stems=True``)
    或模型的全部音軌(drums / bass / other / vocals)。

    ``segment`` 控制每次送進模型的秒數 —— VRAM 不足時調小它。
    6 GB 的顯示卡通常要降到 5~7 秒才不會 OOM。
    """
    import torch
    from demucs.apply import apply_model

    model = load_model(model_name, device)
    device = device or available_device()

    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 1:
        audio = audio.reshape(-1, 1)

    # 模型有固定的取樣率與聲道數,先對齊
    if samplerate != model.samplerate:
        audio = _resample(audio, samplerate, model.samplerate)
    if audio.shape[1] == 1 and model.audio_channels == 2:
        audio = np.repeat(audio, 2, axis=1)
    elif audio.shape[1] > model.audio_channels:
        audio = audio[:, :model.audio_channels]

    # Demucs 期望 (channels, samples);正規化能讓不同音量的輸入表現一致
    wav = torch.from_numpy(audio.T.copy())
    ref = wav.mean(0)
    mean, std = ref.mean(), ref.std()
    wav = (wav - mean) / (std + 1e-8)

    kwargs = {"device": device, "split": True, "overlap": overlap,
              "progress": progress}
    if segment is not None:
        kwargs["segment"] = segment

    with torch.no_grad():
        stems = apply_model(model, wav[None], **kwargs)[0]

    stems = stems * std + mean
    result = {name: stems[i].cpu().numpy().T.astype(np.float32)
              for i, name in enumerate(model.sources)}

    if not two_stems:
        return result

    vocals = result.get("vocals")
    if vocals is None:
        raise RuntimeError(f"模型 {model_name} 沒有 vocals 音軌: {list(result)}")
    accompaniment = sum(v for k, v in result.items() if k != "vocals")
    return {"vocals": vocals, "accompaniment": accompaniment}


def _resample(audio: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    """多相重取樣(scipy 的 resample_poly,比 FFT 版本快且不會有邊界振鈴)。"""
    if source_rate == target_rate:
        return audio
    from math import gcd

    from scipy.signal import resample_poly

    divisor = gcd(int(source_rate), int(target_rate))
    up = int(target_rate) // divisor
    down = int(source_rate) // divisor
    return resample_poly(audio, up, down, axis=0).astype(np.float32)
