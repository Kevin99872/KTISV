"""把訓練好的模型匯出成 ONNX,給 C# 端的 ONNX Runtime 用。

    python -m ktisv_research.export data/runs/latest/best.pt --out data/models/vocals.onnx

為什麼要自己重寫 STFT
--------------------
``torch.stft`` / ``torch.istft`` 匯不出 ONNX —— 它們吐複數張量,而 ONNX 沒有
複數型別;ONNX 從 opset 17 起有 STFT 運算子,卻始終沒有 ISTFT。

於是有兩條路:

  (A) 只匯出 U-Net(幅度譜進、遮罩出),STFT 與 iSTFT 交給 C# 自己做
  (B) 把 STFT / iSTFT 改寫成卷積,整條「波形進、波形出」一起匯出

這裡走 (B)。STFT 本質上就是「一組固定的濾波器」,把 DFT 基底寫成 conv1d 的
權重就行;iSTFT 則是轉置卷積加上窗函數能量的正規化。兩者都是純實數運算,
ONNX 完全支援。

代價是多幾十行程式碼與一次數值驗證。換來的是 C# 端只要餵波形、拿波形 ——
不必在 KTISV 裡實作一套 FFT、不必擔心兩邊的窗函數或正規化慣例對不上。
那種錯誤不會讓程式崩潰,只會讓聲音悄悄變差,是最難查的一種。
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import ModelConfig, Separator, build


class ConvSTFT(nn.Module):
    """用 conv1d 實作的 STFT。輸出實部與虛部兩個實數張量。"""

    def __init__(self, n_fft: int, hop_length: int) -> None:
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        bins = n_fft // 2 + 1

        window = torch.hann_window(n_fft, dtype=torch.float64)
        n = torch.arange(n_fft, dtype=torch.float64)
        k = torch.arange(bins, dtype=torch.float64)[:, None]
        angle = 2.0 * math.pi * k * n / n_fft

        # (bins, 1, n_fft):每個頻格一個濾波器,窗函數直接乘進權重
        real = (torch.cos(angle) * window)[:, None, :]
        imag = (-torch.sin(angle) * window)[:, None, :]
        self.register_buffer("kernel",
                             torch.cat([real, imag], dim=0).float())

    def forward(self, waveform: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(B*C, N) → 兩個 (B*C, bins, T) 的實數張量。"""
        # torch.stft(center=True) 會在兩端各補 n_fft//2,用 reflect 模式
        padded = F.pad(waveform[:, None, :],
                       (self.n_fft // 2, self.n_fft // 2), mode="reflect")
        spec = F.conv1d(padded, self.kernel, stride=self.hop_length)
        bins = self.n_fft // 2 + 1
        return spec[:, :bins], spec[:, bins:]


class ConvISTFT(nn.Module):
    """轉置卷積實作的 iSTFT,含重疊相加的能量正規化。"""

    def __init__(self, n_fft: int, hop_length: int) -> None:
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        bins = n_fft // 2 + 1

        window = torch.hann_window(n_fft, dtype=torch.float64)
        n = torch.arange(n_fft, dtype=torch.float64)
        k = torch.arange(bins, dtype=torch.float64)[:, None]
        angle = 2.0 * math.pi * k * n / n_fft

        # 反 DFT 只用一半的頻格,所以除了 DC 與 Nyquist 之外都要乘 2
        # 才能把被 Hermitian 對稱省掉的那一半補回來。
        scale = torch.full((bins, 1), 2.0, dtype=torch.float64)
        scale[0] = 1.0
        if n_fft % 2 == 0:
            scale[-1] = 1.0
        scale /= n_fft

        real = (torch.cos(angle) * scale * window)[:, None, :]
        imag = (-torch.sin(angle) * scale * window)[:, None, :]
        self.register_buffer("kernel",
                             torch.cat([real, imag], dim=0).float())

        # 窗函數平方的重疊相加包絡。分析與合成各乘一次窗,所以要除掉 w²
        # 的疊加總和才會還原成原始振幅。
        self.register_buffer("window_squared",
                             (window ** 2).float()[None, None, :])

    def forward(self, real: torch.Tensor, imag: torch.Tensor,
                length: int) -> torch.Tensor:
        frames = torch.cat([real, imag], dim=1)
        signal = F.conv_transpose1d(frames, self.kernel, stride=self.hop_length)

        # 同樣的重疊結構套在 w² 上,得到每個取樣被加了多少次窗能量
        ones = torch.ones(1, real.shape[-1], 1, device=real.device,
                          dtype=real.dtype).transpose(1, 2)
        envelope = F.conv_transpose1d(
            ones.expand(1, 1, real.shape[-1]),
            self.window_squared, stride=self.hop_length)

        signal = signal / envelope.clamp_min(1e-8)
        start = self.n_fft // 2
        return signal[:, 0, start:start + length]


class OnnxSeparator(nn.Module):
    """波形進、波形出的完整圖,可匯出成 ONNX。

    輸出人聲**與**伴奏兩軌。伴奏在數學上就是 ``混音 − 人聲``,C# 端自己減
    也行,但一起輸出可以讓呼叫端不必知道這件事,也不會兩邊算法不一致。
    """

    def __init__(self, config: ModelConfig, unet: nn.Module) -> None:
        super().__init__()
        self.config = config
        self.unet = unet
        self.stft = ConvSTFT(config.n_fft, config.hop_length)
        self.istft = ConvISTFT(config.n_fft, config.hop_length)

    def forward(self, waveform: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """waveform: (B, C, N) → (人聲, 伴奏),形狀相同。"""
        batch, channels, length = waveform.shape
        flat = waveform.reshape(batch * channels, length)

        real, imag = self.stft(flat)
        magnitude = torch.sqrt(real * real + imag * imag + 1e-12)

        bins = self.config.freq_bins
        shaped = magnitude.reshape(batch, channels, *magnitude.shape[-2:])
        mask_low = self.unet(shaped[:, :, :bins])

        mask = torch.zeros_like(shaped)
        mask[:, :, :bins] = mask_low
        mask = mask.reshape(batch * channels, *magnitude.shape[-2:])

        vocals = self.istft(real * mask, imag * mask, length)
        vocals = vocals.reshape(batch, channels, length)
        return vocals, waveform - vocals


def aligned_length(seconds: float, config: ModelConfig,
                   samplerate: int = 44100) -> int:
    """挑一個「頻譜幀數剛好被 2^depth 整除」的片段長度。

    為什麼要這麼講究:U-Net 在還原時,若上採樣的尺寸和 skip 對不上,
    ``model.py`` 會用 ``F.interpolate`` 硬湊回去。那在 PyTorch 裡沒問題,
    但匯出 ONNX 時,那個目標尺寸會被**當成常數烙進圖裡** —— 換一個長度
    來跑就會在 Concat 節點炸掉(實測:第 3 軸 129 對不上 128)。

    把長度挑成整除的,那條分支根本不會被觸發,圖裡也就沒有那些常數。
    這比事後想辦法讓 interpolate 支援動態尺寸乾淨得多。
    """
    stride = 2 ** config.depth
    hop = config.hop_length
    # torch.stft(center=True) 給的幀數是 samples // hop + 1
    frames = round(seconds * samplerate / hop) + 1
    frames = max(stride, round(frames / stride) * stride)
    return (frames - 1) * hop


def build_onnx_model(checkpoint: Path, device: str = "cpu") -> OnnxSeparator:
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    config = ModelConfig(**state["model_config"])
    separator = Separator(config)
    separator.load_state_dict(state["model"])
    return OnnxSeparator(config, separator.unet).to(device).eval()


@torch.no_grad()
def verify(checkpoint: Path, seconds: float = 4.0,
           samplerate: int = 44100) -> dict[str, float]:
    """匯出前先確認卷積版 STFT 與 PyTorch 原版算出來的是同一件事。

    這一步不能省。兩邊只要有一個常數對不上(窗函數、正規化、補零方式),
    模型不會報錯,只會安靜地輸出比較差的聲音。
    """
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = ModelConfig(**state["model_config"])
    reference = Separator(config)
    reference.load_state_dict(state["model"])
    reference.eval()

    exportable = OnnxSeparator(config, reference.unet).eval()

    torch.manual_seed(0)
    waveform = torch.randn(1, config.channels, int(seconds * samplerate)) * 0.2

    # ① 卷積 STFT vs torch.stft
    flat = waveform.reshape(-1, waveform.shape[-1])
    real, imag = exportable.stft(flat)
    torch_spec = reference.stft(waveform).reshape(-1, *real.shape[-2:])
    stft_error = (torch.stack([real, imag], -1)
                  - torch.view_as_real(torch_spec)).abs().max().item()

    # ② 卷積 iSTFT 往返
    roundtrip = exportable.istft(real, imag, waveform.shape[-1])
    istft_error = (roundtrip.reshape(waveform.shape) - waveform).abs().max().item()

    # ③ 整個模型:兩條路徑的最終輸出
    vocals_ref = reference(waveform)["vocals"]
    vocals_onnx, accompaniment = exportable(waveform)
    model_error = (vocals_onnx - vocals_ref).abs().max().item()
    additivity = (vocals_onnx + accompaniment - waveform).abs().max().item()

    return {
        "stft_max_error": stft_error,
        "istft_roundtrip_error": istft_error,
        "model_max_error": model_error,
        "additivity_error": additivity,
        "signal_peak": waveform.abs().max().item(),
    }


def export(checkpoint: Path, output: Path, seconds: float = 6.0,
           samplerate: int = 44100, opset: int = 17) -> tuple[Path, int]:
    """匯出成**固定片段長度**的 ONNX。回傳 (檔案路徑, 片段取樣數)。

    長度刻意不設成動態軸。理由見 :func:`aligned_length` —— 這個 U-Net 的
    尺寸對齊邏輯沒辦法安全地匯成動態圖。而且對呼叫端來說也沒有損失:
    這類模型本來就要分塊推論,固定塊長反而讓 ONNX Runtime 能預先配置。
    整首歌由 C# 端切塊、重疊相加,最後一塊補零。
    """
    model = build_onnx_model(checkpoint)
    length = aligned_length(seconds, model.config, samplerate)
    dummy = torch.randn(1, model.config.channels, length) * 0.1

    output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model, (dummy,), str(output),
        input_names=["mixture"],
        output_names=["vocals", "accompaniment"],
        # 只有批次是動態的;長度固定(見上)
        dynamic_axes={"mixture": {0: "batch"},
                      "vocals": {0: "batch"},
                      "accompaniment": {0: "batch"}},
        opset_version=opset,
        dynamo=False,
    )

    # 把呼叫端一定要知道的事寫進模型檔本身,而不是只寫在文件裡 ——
    # 文件會和檔案走散,metadata 不會。
    import onnx

    proto = onnx.load(str(output))
    for key, value in (("samplerate", str(samplerate)),
                       ("segment_samples", str(length)),
                       ("channels", str(model.config.channels)),
                       ("n_fft", str(model.config.n_fft)),
                       ("hop_length", str(model.config.hop_length))):
        entry = proto.metadata_props.add()
        entry.key, entry.value = key, value
    onnx.save(proto, str(output))

    return output, length


@torch.no_grad()
def verify_onnx(checkpoint: Path, onnx_path: Path, length: int,
                samplerate: int = 44100) -> dict[str, float]:
    """真的用 ONNX Runtime 跑一次,和 PyTorch 的輸出逐點比對。

    前面的 verify() 比的是「兩種 PyTorch 寫法」;這裡比的是「PyTorch 對上
    另一套執行引擎」。運算子語意的差異(補零、捨入、融合)只有在這一步
    才會現形,而那正是 C# 端實際會跑到的東西。
    """
    import onnxruntime as ort

    model = build_onnx_model(checkpoint)
    torch.manual_seed(1)
    waveform = torch.randn(1, model.config.channels, length) * 0.2

    vocals_torch, accompaniment_torch = model(waveform)

    session = ort.InferenceSession(str(onnx_path),
                                   providers=["CPUExecutionProvider"])
    vocals_onnx, accompaniment_onnx = session.run(
        None, {"mixture": waveform.numpy()})

    def worst(a: torch.Tensor, b: np.ndarray) -> float:
        return float(np.abs(a.numpy() - b).max())

    return {
        "vocals_max_error": worst(vocals_torch, vocals_onnx),
        "accompaniment_max_error": worst(accompaniment_torch, accompaniment_onnx),
        "additivity_error": float(np.abs(
            vocals_onnx + accompaniment_onnx - waveform.numpy()).max()),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="把檢查點匯出成 ONNX")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--out", type=Path, default=Path("data/models/vocals.onnx"))
    parser.add_argument("--seconds", type=float, default=6.0)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--skip-verify", action="store_true")
    args = parser.parse_args(argv)

    if not args.skip_verify:
        print("驗證卷積版 STFT/iSTFT 與 PyTorch 原版是否一致…")
        errors = verify(args.checkpoint)
        for name, value in errors.items():
            print(f"  {name:24s} {value:.3e}")
        worst = max(errors["stft_max_error"], errors["istft_roundtrip_error"],
                    errors["model_max_error"])
        if worst > 1e-3:
            print(f"\n❌ 誤差 {worst:.3e} 太大,匯出會得到不一樣的模型。已中止。")
            return 1
        print(f"  → 最大誤差 {worst:.3e},通過\n")

    path, length = export(args.checkpoint, args.out, args.seconds, opset=args.opset)
    size = path.stat().st_size / 1e6
    print(f"已匯出 {path}  ({size:.1f} MB)")
    print(f"  固定片段長度 {length} 取樣 = {length / 44100:.3f} 秒")

    if not args.skip_verify:
        print("\n用 ONNX Runtime 實跑一次,與 PyTorch 逐點比對…")
        errors = verify_onnx(args.checkpoint, path, length)
        for name, value in errors.items():
            print(f"  {name:24s} {value:.3e}")
        if max(errors.values()) > 1e-3:
            print("\n❌ ONNX 與 PyTorch 的輸出對不上,這個檔案不能用。")
            return 1
        print("  → 通過")

    print("\nC# 端:Microsoft.ML.OnnxRuntime")
    print(f"  輸入 mixture: float32 (batch, {2}, {length})")
    print("  輸出 vocals / accompaniment:同形狀")
    print("  整首歌要自己切成這個長度的塊、重疊相加,最後一塊補零")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
