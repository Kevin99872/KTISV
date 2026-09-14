"""用訓練好的模型分離一個真實音檔 —— 產生可以直接聽的 wav。

    .venv-tf\\Scripts\\python -m ktisv_tf.separate 歌.webm --checkpoint data/runs/tf/best
    .venv-tf\\Scripts\\python -m ktisv_tf.separate 歌.webm --onnx data/models/vocals-tf.onnx

驗證分數會告訴你模型有沒有在學,但**分數好聽起來仍然可能很糟** ——
SI-SDR 對某些人耳很敏感的假象(音樂噪聲、高頻抖動)幾乎沒有反應。
所以每次訓練完都該真的聽一次。

``--onnx`` 走的是 C# 端會用的那個檔案與那套執行引擎,是最接近實際部署的
一次演練。兩種模式輸出應該幾乎一樣;差很多就代表匯出有問題。

分塊與重疊相加
--------------
模型的片段長度是固定的,整首歌要自己切。直接切了再接起來會在接縫留下
喀噠聲 —— 每一塊的邊界受 STFT 補零影響,而且相鄰兩塊的遮罩不連續。

所以用半塊的間距滑動、每塊乘上一個 Hann 窗再疊加,最後除以窗的疊加總和。
相鄰塊在接縫處平滑地交叉淡入淡出,而窗和為常數保證中段的振幅不變。
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

if __package__ in (None, ""):  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from . import SAMPLE_RATE
from .data import read_audio


# ── 推論後端 ────────────────────────────────────────────────────────────
class KerasBackend:
    """直接用 TensorFlow 跑。訓練完馬上試聽用這個,不必先匯出。"""

    def __init__(self, checkpoint: Path, seconds: float) -> None:
        import tensorflow as tf

        tf.config.set_visible_devices([], "GPU")   # 整首歌的長度在 CPU 上比較穩
        from .export import build_export_model

        self.model, self.config, self.segment, self.meta = build_export_model(
            checkpoint, seconds)
        self.channels = self.config.channels

    def __call__(self, block: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        vocals, accompaniment = self.model(block[None], training=False)
        return vocals.numpy()[0], accompaniment.numpy()[0]

    def describe(self) -> str:
        return (f"TensorFlow / {self.meta.get('preset')} / "
                f"step {self.meta.get('step')}")


class OnnxBackend:
    """走 ONNX Runtime —— 和 C# 端跑的是同一個檔案、同一套引擎。"""

    def __init__(self, path: Path) -> None:
        import onnx
        import onnxruntime as ort

        meta = {p.key: p.value for p in onnx.load(str(path)).metadata_props}
        self.segment = int(meta["segment_samples"])
        self.channels = int(meta.get("channels", 2))
        self.samplerate = int(meta.get("samplerate", SAMPLE_RATE))
        self.framework = meta.get("framework", "?")
        self.session = ort.InferenceSession(
            str(path), providers=["CPUExecutionProvider"])
        self.path = path

    def __call__(self, block: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        vocals, accompaniment = self.session.run(
            ["vocals", "accompaniment"], {"mixture": block[None]})
        return vocals[0], accompaniment[0]

    def describe(self) -> str:
        return f"ONNX Runtime / {self.path.name} / {self.framework}"


# ── 分塊推論 ────────────────────────────────────────────────────────────
def separate(backend, audio: np.ndarray, progress: bool = True
             ) -> tuple[np.ndarray, np.ndarray]:
    """(samples, 2) → (人聲, 伴奏),都是 (samples, 2)。"""
    segment = backend.segment
    hop = segment // 2
    total = len(audio)

    # 至少要有一整塊;不足的補零,最後再切回原長度
    padded_length = max(segment, ((total + hop - 1) // hop) * hop + hop)
    padded = np.zeros((padded_length, 2), dtype=np.float32)
    padded[:total] = audio[:, :2]

    window = np.hanning(segment + 1)[:-1].astype(np.float32)[:, None]
    vocals = np.zeros_like(padded)
    envelope = np.zeros((padded_length, 1), dtype=np.float32)

    starts = list(range(0, padded_length - segment + 1, hop))
    started = time.time()
    for index, start in enumerate(starts, 1):
        block = np.ascontiguousarray(padded[start:start + segment].T)
        estimate, _ = backend(block)
        vocals[start:start + segment] += estimate.T * window
        envelope[start:start + segment] += window
        if progress:
            print(f"\r  {index}/{len(starts)} 塊"
                  f"  {time.time() - started:.0f}s", end="", flush=True)
    if progress:
        print()

    vocals /= np.maximum(envelope, 1e-8)
    vocals = vocals[:total]
    # 伴奏由減法得到,而不是各自重疊相加 —— 這樣兩軌相加必定還原成原曲,
    # 不會因為兩邊的重疊誤差不同而漏掉或多出東西。
    return vocals, audio[:total, :2] - vocals


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="用訓練好的模型分離一個音檔")
    parser.add_argument("input", type=Path)
    parser.add_argument("--out", type=Path, default=Path("data/separated"))
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="不含副檔名的前綴,例如 data/runs/tf/best")
    parser.add_argument("--onnx", type=Path, default=None,
                        help="改用匯出的 ONNX(等同 C# 端會跑的東西)")
    parser.add_argument("--seconds", type=float, default=6.0,
                        help="--checkpoint 模式下每塊多長")
    parser.add_argument("--max-minutes", type=float, default=None)
    args = parser.parse_args(argv)

    if bool(args.checkpoint) == bool(args.onnx):
        print("要指定 --checkpoint 或 --onnx,兩者擇一。")
        return 2

    backend = (OnnxBackend(args.onnx) if args.onnx
               else KerasBackend(args.checkpoint, args.seconds))

    audio = read_audio(args.input, SAMPLE_RATE,
                       args.max_minutes * 60 if args.max_minutes else None)

    print(f"模型   {backend.describe()}")
    print(f"輸入   {args.input.name}  {len(audio) / SAMPLE_RATE / 60:.1f} 分鐘")
    print(f"分塊   {backend.segment} 取樣 = "
          f"{backend.segment / SAMPLE_RATE:.2f} 秒,半塊重疊")

    vocals, accompaniment = separate(backend, audio)

    folder = args.out / args.input.stem
    folder.mkdir(parents=True, exist_ok=True)
    sf.write(folder / "vocals.wav", vocals, SAMPLE_RATE)
    sf.write(folder / "accompaniment.wav", accompaniment, SAMPLE_RATE)

    residual = float(np.abs(vocals + accompaniment - audio[:len(vocals), :2]).max())
    print(f"\n輸出   {folder}")
    print(f"  vocals.wav / accompaniment.wav")
    print(f"  兩軌相加還原誤差 {residual:.2e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
