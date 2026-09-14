"""把訓練好的 TensorFlow 模型匯出成 ONNX,給 C# 端的 ONNX Runtime 用。

    python -m ktisv_tf.export data/runs/tf/best --out data/models/vocals-tf.onnx

契約與 PyTorch 版完全相同
-------------------------
輸入 ``mixture``:float32 ``(batch, 2, segment_samples)``
輸出 ``vocals`` / ``accompaniment``:同形狀

這不是巧合,是刻意的。C# 端載入的時候不該需要知道權重是 PyTorch 還是
TensorFlow 訓出來的 —— 兩邊都吐同一份介面,就能直接換檔案比較。

``segment_samples`` 等呼叫端一定要知道的事寫進 ONNX 的 metadata,而不是
只寫在文件裡:文件會和檔案走散,metadata 不會。

為什麼長度是固定的
------------------
U-Net 的尺寸對齊靠「頻譜幀數整除 2^depth」。做成動態軸的話,遇到不整除的
長度會在 Concat 節點直接炸掉。對呼叫端沒有損失:這類模型本來就要分塊
推論,固定塊長反而讓 ONNX Runtime 能預先配置。整首歌由 C# 端切塊、
重疊相加,最後一塊補零。

驗證不能省
----------
匯出後一定要用 ONNX Runtime 實跑一次,和 TensorFlow 的輸出逐點比對。
運算子語意的差異(補零、捨入、融合)只有在這一步才會現形 —— 而且它們
不會讓程式崩潰,只會讓聲音悄悄變差,是最難查的一種錯。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

if __package__ in (None, ""):  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from . import SAMPLE_RATE
from .model import (aligned_length, build_separator, build_unet,
                    config_from_dict)


def load_checkpoint(prefix: Path):
    """``prefix`` 是不含副檔名的路徑,例如 ``data/runs/tf/best``。"""
    prefix = Path(prefix)
    if prefix.suffix in (".h5", ".json"):
        prefix = prefix.with_suffix("")
    meta_path = prefix.with_suffix(".json")
    weights_path = prefix.with_suffix(".h5")
    if not meta_path.exists() or not weights_path.exists():
        raise FileNotFoundError(
            f"找不到 {meta_path} 或 {weights_path} —— 訓練有跑完嗎?")

    meta = json.loads(meta_path.read_text("utf-8"))
    config = config_from_dict(meta["model_config"])
    unet = build_unet(config)
    unet.load_weights(str(weights_path))
    return config, unet, meta


def build_export_model(prefix: Path, seconds: float):
    """重建一個「匯出用長度」的完整模型。

    訓練片段與匯出片段可以不同:U-Net 是全卷積的、STFT / iSTFT 是常數層,
    所以同一份權重套到任何對齊過的長度都能跑。訓練用短片段省 VRAM,
    匯出用長片段減少 C# 端的重疊成本。

    不過「能跑」不等於「逐點相同」—— 感受野比片段還長,換長度就換了邊界
    條件,輸出會有微小差異(實測兩者相差約 50 dB SI-SDR,聽不出來)。
    要逐點重現的話,匯出長度就設成和訓練時一樣。
    """
    config, unet, meta = load_checkpoint(prefix)
    length = aligned_length(seconds, config, SAMPLE_RATE)
    return build_separator(config, length, unet), config, length, meta


def to_onnx(model, config, length: int, output: Path, opset: int) -> Path:
    import onnx
    import tf2onnx

    output.parent.mkdir(parents=True, exist_ok=True)
    signature = [tf.TensorSpec((None, config.channels, length), tf.float32,
                               name="mixture")]
    proto, _ = tf2onnx.convert.from_keras(model, input_signature=signature,
                                          opset=opset)

    # tf2onnx 會把 Keras 的輸出層名加上後綴。C# 端是照名字取的,所以在這裡
    # 正名 —— 讓兩個框架匯出的檔案有完全一樣的輸入輸出名。
    _rename_outputs(proto, ["vocals", "accompaniment"])

    for key, value in (("samplerate", str(SAMPLE_RATE)),
                       ("segment_samples", str(length)),
                       ("channels", str(config.channels)),
                       ("n_fft", str(config.n_fft)),
                       ("hop_length", str(config.hop_length)),
                       ("framework", "tensorflow")):
        entry = proto.metadata_props.add()
        entry.key, entry.value = key, value

    onnx.save(proto, str(output))
    return output


def _rename_outputs(proto, names: list[str]) -> None:
    """把圖的輸出改名成契約上的名字。

    改名要三個地方一起改:``graph.output``、產生它的節點的 ``output``、
    以及**其他把它當輸入的節點**。最後那一項容易漏 —— 一個張量既是圖的
    輸出、又餵給下游節點是完全合法的(這個模型裡 vocals 就同時被
    ``accompaniment = mixture − vocals`` 用到)。只改前兩處的話,下游節點
    會指向一個不存在的名字,圖就斷了。
    """
    if len(proto.graph.output) != len(names):
        raise RuntimeError(
            f"預期 {len(names)} 個輸出,實際 {len(proto.graph.output)} 個 —— "
            "模型結構和匯出程式不一致。")

    mapping = {out.name: new for out, new in zip(proto.graph.output, names)}
    if len(mapping) != len(names):
        raise RuntimeError("兩個輸出指向同一個張量,無法分別命名。")

    for out in proto.graph.output:
        out.name = mapping[out.name]
    for node in proto.graph.node:
        for group in (node.input, node.output):
            for i, name in enumerate(group):
                if name in mapping:
                    group[i] = mapping[name]
    for info in proto.graph.value_info:
        if info.name in mapping:
            info.name = mapping[info.name]


def verify(model, onnx_path: Path, config, length: int,
           seed: int = 1) -> dict[str, float]:
    """ONNX Runtime 跑一次,和 TensorFlow 逐點比對。"""
    import onnxruntime as ort

    rng = np.random.default_rng(seed)
    waveform = (rng.standard_normal((1, config.channels, length))
                * 0.2).astype(np.float32)

    vocals_tf, accompaniment_tf = model(waveform, training=False)
    session = ort.InferenceSession(str(onnx_path),
                                   providers=["CPUExecutionProvider"])
    vocals_onnx, accompaniment_onnx = session.run(
        ["vocals", "accompaniment"], {"mixture": waveform})

    return {
        "vocals_max_error": float(np.abs(vocals_tf.numpy() - vocals_onnx).max()),
        "accompaniment_max_error": float(
            np.abs(accompaniment_tf.numpy() - accompaniment_onnx).max()),
        # 兩軌相加要還原成混音。這條性質壞掉代表遮罩或 iSTFT 被改壞了。
        "additivity_error": float(
            np.abs(vocals_onnx + accompaniment_onnx - waveform).max()),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="把 TF 檢查點匯出成 ONNX")
    parser.add_argument("checkpoint", type=Path,
                        help="不含副檔名的前綴,例如 data/runs/tf/best")
    parser.add_argument("--out", type=Path,
                        default=Path("data/models/vocals-tf.onnx"))
    parser.add_argument("--seconds", type=float, default=6.0)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--skip-verify", action="store_true")
    args = parser.parse_args(argv)

    tf.config.set_visible_devices([], "GPU")   # 匯出與驗證都在 CPU 上做

    model, config, length, meta = build_export_model(args.checkpoint, args.seconds)
    print(f"檢查點    {args.checkpoint}  (step {meta.get('step')}, "
          f"best {meta.get('best_si_sdr'):+.2f} dB)")
    print(f"片段長度  {length} 取樣 = {length / SAMPLE_RATE:.3f} 秒")

    path = to_onnx(model, config, length, args.out, args.opset)
    print(f"已匯出    {path}  ({path.stat().st_size / 1e6:.1f} MB)")

    if not args.skip_verify:
        print("\n用 ONNX Runtime 實跑一次,與 TensorFlow 逐點比對…")
        errors = verify(model, path, config, length)
        for name, value in errors.items():
            print(f"  {name:26s} {value:.3e}")
        if max(errors.values()) > 1e-3:
            print("\n❌ ONNX 與 TensorFlow 的輸出對不上,這個檔案不能用。")
            return 1
        print("  → 通過")

    print("\nC# 端:Microsoft.ML.OnnxRuntime")
    print(f"  輸入 mixture: float32 (batch, {config.channels}, {length})")
    print("  輸出 vocals / accompaniment:同形狀")
    print("  整首歌要自己切成這個長度的塊、重疊相加,最後一塊補零")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
