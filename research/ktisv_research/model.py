"""輕量頻譜圖 U-Net —— 為 6 GB VRAM 設計的人聲分離模型。

為什麼是這個架構
----------------
Demucs 吃重的原因是它在**波形域**工作,還帶 Transformer。訊號長度是取樣數
(44100/秒),注意力機制的成本隨長度平方成長,所以 VRAM 需求極高。

改在**頻譜圖**上工作,長度立刻降到約 86 幀/秒(hop=512),少了 500 倍。
再用純卷積的 U-Net(沒有注意力),記憶體就變成可預測的線性成長。這就是
Spleeter 能在普通顯示卡上跑的原因。

代價很明確:只預測幅度遮罩、沿用混音的相位。相位不完美會在人聲軌留下
輕微的「相位感」殘影,這是這類架構的固有上限 —— 波形域模型才能超越它。

運作方式
--------
    混音 → STFT → 幅度譜 → U-Net → 遮罩 ∈ [0,1]
                                      ↓
    人聲 ← iSTFT ← 複數譜 × 遮罩 ←────┘

遮罩式輸出有個重要好處:模型只需要決定「這個時頻點有多少比例屬於人聲」,
不必從零合成訊號。輸出天然被限制在輸入的範圍內,不會憑空生出東西,
訓練也比直接回歸頻譜穩定得多。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    """模型與 STFT 設定。

    ``base_channels`` 是控制模型大小的主要旋鈕 —— VRAM 不夠時先調它。
    """

    n_fft: int = 2048
    hop_length: int = 512
    # 送進 U-Net 的頻率 bin 數。這個值真正的用途是**讓頻率軸能被 2^depth
    # 整除** —— rfft 給的是 1025 個 bin,而 1025 除不盡 32,每次下採樣都會
    # 掉一個奇數列,還原時得靠內插硬湊回去。取 1024 就整除了。
    #
    # 順帶一提:1024 個 bin 在 44.1 kHz / n_fft=2048 之下涵蓋到 22028 Hz,
    # 也就是**幾乎整個頻帶**,只丟掉最上面那一個 bin(22028–22050 Hz)。
    # 這裡不是在做低通。0 表示全部保留(但那會失去整除的好處)。
    max_bins: int = 1024
    channels: int = 2               # 立體聲
    base_channels: int = 24         # 每層的基礎寬度
    depth: int = 5                  # 下採樣次數
    growth: float = 2.0             # 每層通道數的成長倍率
    max_channels: int = 384         # 通道數上限,避免最深層爆掉

    @property
    def freq_bins(self) -> int:
        full = self.n_fft // 2 + 1
        return min(self.max_bins, full) if self.max_bins else full


def _conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
    """兩層卷積 + 正規化。

    用 BatchNorm 而非 LayerNorm:頻譜圖的統計特性在 batch 之間相對穩定,
    而 BatchNorm 的記憶體開銷更小。
    """
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.LeakyReLU(0.1, inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.LeakyReLU(0.1, inplace=True),
    )


class SpectrogramUNet(nn.Module):
    """預測人聲遮罩的 U-Net。

    輸入 ``(B, C, F, T)`` 的幅度譜,輸出同形狀、值域 ``[0, 1]`` 的遮罩。
    """

    def __init__(self, config: ModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or ModelConfig()
        cfg = self.config

        widths = []
        width = cfg.base_channels
        for _ in range(cfg.depth):
            widths.append(min(int(width), cfg.max_channels))
            width *= cfg.growth
        self.widths = widths

        self.encoders = nn.ModuleList()
        in_ch = cfg.channels
        for w in widths:
            self.encoders.append(_conv_block(in_ch, w))
            in_ch = w

        bottleneck_width = min(int(widths[-1] * cfg.growth), cfg.max_channels)
        self.bottleneck = _conv_block(widths[-1], bottleneck_width)

        self.upsamples = nn.ModuleList()
        self.decoders = nn.ModuleList()
        in_ch = bottleneck_width
        for w in reversed(widths):
            self.upsamples.append(
                nn.ConvTranspose2d(in_ch, w, kernel_size=2, stride=2))
            # 串接 skip 之後通道數翻倍
            self.decoders.append(_conv_block(w * 2, w))
            in_ch = w

        self.head = nn.Conv2d(widths[0], cfg.channels, 1)

    def forward(self, magnitude: torch.Tensor) -> torch.Tensor:
        """magnitude: (B, C, F, T) → 遮罩 (B, C, F, T),值域 [0, 1]。"""
        # 對數壓縮:頻譜的動態範圍極大(80+ dB),直接餵原始幅度會讓
        # 少數高能量點主導梯度。log1p 讓分佈接近常態,訓練穩定得多。
        x = torch.log1p(magnitude)

        skips = []
        for encoder in self.encoders:
            x = encoder(x)
            skips.append(x)
            x = F.max_pool2d(x, 2)

        x = self.bottleneck(x)

        for upsample, decoder, skip in zip(self.upsamples, self.decoders,
                                           reversed(skips)):
            x = upsample(x)
            # 下採樣的尺寸不一定能整除,還原時對齊到 skip 的大小
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:],
                                  mode="bilinear", align_corners=False)
            x = decoder(torch.cat([x, skip], dim=1))

        return torch.sigmoid(self.head(x))

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


class Separator(nn.Module):
    """把 STFT / iSTFT 包進來,提供波形進、波形出的完整介面。"""

    def __init__(self, config: ModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or ModelConfig()
        self.unet = SpectrogramUNet(self.config)
        self.register_buffer("window", torch.hann_window(self.config.n_fft),
                             persistent=False)

    def stft(self, waveform: torch.Tensor) -> torch.Tensor:
        """(B, C, N) → 複數譜 (B, C, F, T)。"""
        batch, channels, samples = waveform.shape
        flat = waveform.reshape(batch * channels, samples)
        spec = torch.stft(flat, self.config.n_fft,
                          hop_length=self.config.hop_length,
                          window=self.window, return_complex=True,
                          center=True, pad_mode="reflect")
        return spec.reshape(batch, channels, *spec.shape[-2:])

    def istft(self, spec: torch.Tensor, length: int) -> torch.Tensor:
        batch, channels = spec.shape[:2]
        flat = spec.reshape(batch * channels, *spec.shape[-2:])
        waveform = torch.istft(flat, self.config.n_fft,
                               hop_length=self.config.hop_length,
                               window=self.window, center=True, length=length)
        return waveform.reshape(batch, channels, -1)

    def forward(self, waveform: torch.Tensor) -> dict[str, torch.Tensor]:
        """waveform: (B, C, N) → {"vocals", "accompaniment", "mask"}。"""
        length = waveform.shape[-1]
        spec = self.stft(waveform)
        magnitude = spec.abs()

        bins = self.config.freq_bins
        low = magnitude[:, :, :bins]
        mask_low = self.unet(low)

        # 沒進模型的那幾個 bin 遮罩給 0,也就是全部歸伴奏。預設設定下這只是
        # 最頂端一個 bin(22028 Hz 以上),影響可以忽略;真正的意義是在
        # max_bins 被調小時,行為是「保守地把不確定的頻段讓給伴奏」而不是
        # 讓它憑空進人聲軌。
        mask = torch.zeros_like(magnitude)
        mask[:, :, :bins] = mask_low

        vocals_spec = spec * mask
        vocals = self.istft(vocals_spec, length)
        return {
            "vocals": vocals,
            "accompaniment": waveform - vocals,
            "mask": mask,
        }

    @property
    def num_parameters(self) -> int:
        return self.unet.num_parameters


# 幾組預設大小,方便在品質與 VRAM 之間取捨
PRESETS = {
    "tiny":   ModelConfig(base_channels=12, depth=4, max_channels=128),
    "small":  ModelConfig(base_channels=16, depth=5, max_channels=256),
    "medium": ModelConfig(base_channels=24, depth=5, max_channels=384),
    "large":  ModelConfig(base_channels=32, depth=6, max_channels=512),
}


def build(preset: str = "medium") -> Separator:
    if preset not in PRESETS:
        raise ValueError(f"未知的預設值 {preset};可用:{list(PRESETS)}")
    return Separator(PRESETS[preset])
