"""TensorFlow 版的輕量頻譜 U-Net —— 波形進、波形出。

架構與 ``ktisv_research/model.py`` 的 PyTorch 版相同(理由見該檔):在頻譜圖
上工作、只預測幅度遮罩、沿用混音的相位。這裡只講 TF 這一版不一樣的地方。

STFT 從一開始就用卷積寫
-----------------------
PyTorch 那側是「訓練用 ``torch.stft``、匯出時再換成卷積版」,於是必須額外
寫一段驗證去確認兩條路徑算出來的是同一件事(``export.py`` 的 ``verify``)。

這裡不走那條路:**訓練與匯出用的是同一組卷積層**。理由是 ``tf.signal.stft``
依賴 RFFT,而 RFFT 到 ONNX 的轉換並不可靠;既然遲早要換成卷積,不如從頭
就只有一種實作 —— 沒有兩條路徑,就沒有兩條路徑對不上的可能。

STFT 本質上是一組固定的濾波器:把 DFT 基底(乘上窗函數)寫成 conv1d 的權重
即可。iSTFT 是轉置卷積,再除掉窗函數平方的重疊相加包絡。兩者都是純實數
運算,ONNX 完全支援。

正確性仍然要驗,只是驗的對象變成「卷積 STFT 往返之後有沒有還原成原訊號」,
那是一個不需要參考實作的自足檢查(見 ``tests/test_tf_model.py``)。

張量版面
--------
對外(ONNX 契約)是 ``(batch, channels, samples)`` —— 與 PyTorch 版一致,
C# 端不必分辨。對內走 TF 慣例的 channels-last:頻譜圖是
``(batch, frames, bins, channels)``。cuDNN 與 CPU 的卷積都只對
channels-last 有最佳化路徑,逆著用會慢上數倍。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict

import numpy as np
import tensorflow as tf

keras = tf.keras
layers = tf.keras.layers


@dataclass
class ModelConfig:
    """模型與 STFT 設定。欄位與 PyTorch 版同名同義,方便兩邊對照。

    ``base_channels`` 是控制模型大小的主要旋鈕 —— VRAM 不夠時先調它。
    """

    n_fft: int = 2048
    hop_length: int = 512
    # 送進 U-Net 的頻率 bin 數。真正的用途是**讓頻率軸能被 2^depth 整除** ——
    # rfft 給 1025 個 bin,1025 除不盡 32,每次下採樣都會掉一列。取 1024 就
    # 整除了,而且 44.1 kHz / n_fft=2048 之下它涵蓋到 22028 Hz,只丟掉最上面
    # 那一個 bin。這不是低通。
    max_bins: int = 1024
    channels: int = 2               # 立體聲
    base_channels: int = 16
    depth: int = 5
    growth: float = 2.0
    max_channels: int = 256

    @property
    def full_bins(self) -> int:
        return self.n_fft // 2 + 1

    @property
    def freq_bins(self) -> int:
        return min(self.max_bins, self.full_bins) if self.max_bins else self.full_bins

    @property
    def stride(self) -> int:
        return 2 ** self.depth

    def widths(self) -> list[int]:
        out, width = [], float(self.base_channels)
        for _ in range(self.depth):
            out.append(min(int(width), self.max_channels))
            width *= self.growth
        return out


PRESETS = {
    "tiny":   ModelConfig(base_channels=8,  depth=4, max_channels=96),
    "small":  ModelConfig(base_channels=16, depth=5, max_channels=256),
    "medium": ModelConfig(base_channels=24, depth=5, max_channels=384),
    "large":  ModelConfig(base_channels=32, depth=6, max_channels=512),
}


def aligned_length(seconds: float, config: ModelConfig,
                   samplerate: int = 44100) -> int:
    """挑一個「頻譜幀數剛好被 2^depth 整除」的片段長度。

    U-Net 下採樣 depth 次,幀數不整除時上採樣回來的尺寸就會和 skip 差一格,
    得靠內插硬湊。挑一個整除的長度,那條分支根本不會被觸發 —— 訓練與匯出
    都乾淨。這和 PyTorch 版的 ``export.aligned_length`` 是同一個算式。
    """
    frames = round(seconds * samplerate / config.hop_length) + 1
    frames = max(config.stride, round(frames / config.stride) * config.stride)
    return (frames - 1) * config.hop_length


def frames_for(length: int, config: ModelConfig) -> int:
    """``center=True`` 的 STFT 會給幾幀。"""
    return length // config.hop_length + 1


# ── 卷積版 STFT / iSTFT ─────────────────────────────────────────────────
def _dft_kernels(n_fft: int) -> tuple[np.ndarray, np.ndarray]:
    """回傳 (分析核, 合成核),都是 (n_fft, 1, 2*bins) 的 conv1d 權重。

    以 float64 建表再降到 float32:係數本身是解析式,沒有理由讓建表的
    捨入誤差混進來。
    """
    bins = n_fft // 2 + 1
    window = np.hanning(n_fft + 1)[:-1]          # 與 torch.hann_window 一致
    n = np.arange(n_fft, dtype=np.float64)
    k = np.arange(bins, dtype=np.float64)[:, None]
    angle = 2.0 * np.pi * k * n / n_fft

    analysis = np.concatenate([np.cos(angle) * window,
                               -np.sin(angle) * window], axis=0)

    # 反 DFT 只用一半的頻格,所以除了 DC 與 Nyquist 之外都要乘 2 才能把
    # Hermitian 對稱省掉的那一半補回來。
    scale = np.full((bins, 1), 2.0, dtype=np.float64)
    scale[0] = 1.0
    if n_fft % 2 == 0:
        scale[-1] = 1.0
    scale /= n_fft

    synthesis = np.concatenate([np.cos(angle) * scale * window,
                                -np.sin(angle) * scale * window], axis=0)

    # (2*bins, n_fft) → conv1d 要的 (n_fft, in=1, out=2*bins)
    return (analysis.T[:, None, :].astype(np.float32),
            synthesis.T[:, None, :].astype(np.float32))


class ConvSTFT(layers.Layer):
    """conv1d 實作的 STFT。輸入 (B, N),輸出兩個 (B, T, bins) 的實數張量。"""

    def __init__(self, n_fft: int, hop_length: int, **kwargs) -> None:
        super().__init__(trainable=False, **kwargs)
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.bins = n_fft // 2 + 1
        analysis, _ = _dft_kernels(n_fft)
        self.kernel = tf.constant(analysis)

    def call(self, waveform: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        pad = self.n_fft // 2
        # torch.stft(center=True) 的行為:兩端各補 n_fft//2,reflect 模式
        padded = tf.pad(waveform[:, :, None], [[0, 0], [pad, pad], [0, 0]],
                        mode="REFLECT")
        spec = tf.nn.conv1d(padded, self.kernel, stride=self.hop_length,
                            padding="VALID")
        return spec[..., :self.bins], spec[..., self.bins:]


class ConvISTFT(layers.Layer):
    """轉置卷積實作的 iSTFT,含重疊相加的能量正規化。"""

    def __init__(self, n_fft: int, hop_length: int, **kwargs) -> None:
        super().__init__(trainable=False, **kwargs)
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.bins = n_fft // 2 + 1
        _, synthesis = _dft_kernels(n_fft)
        # conv1d_transpose 的權重是 (width, out_channels, in_channels) ——
        # 這裡 out=1(合成回單一波形)、in=2*bins,正好就是 _dft_kernels
        # 給的版面,不需要再轉置。
        self.kernel = tf.constant(synthesis)

        window = np.hanning(n_fft + 1)[:-1].astype(np.float32)
        self.window_squared = tf.constant(
            (window ** 2).reshape(n_fft, 1, 1))

    def call(self, real: tf.Tensor, imag: tf.Tensor,
             length: int) -> tf.Tensor:
        frames = tf.concat([real, imag], axis=-1)
        batch = tf.shape(frames)[0]
        n_frames = tf.shape(frames)[1]
        out_length = (n_frames - 1) * self.hop_length + self.n_fft

        signal = tf.nn.conv1d_transpose(
            frames, self.kernel,
            output_shape=tf.stack([batch, out_length, 1]),
            strides=self.hop_length, padding="VALID")

        # 同樣的重疊結構套在 w² 上 —— 每個取樣被疊加了多少窗能量。
        # 分析與合成各乘一次窗,所以要除掉 w² 的疊加總和才會還原成原振幅。
        ones = tf.ones(tf.stack([1, n_frames, 1]), dtype=frames.dtype)
        envelope = tf.nn.conv1d_transpose(
            ones, self.window_squared,
            output_shape=tf.stack([1, out_length, 1]),
            strides=self.hop_length, padding="VALID")

        signal = signal / tf.maximum(envelope, 1e-8)
        start = self.n_fft // 2
        return signal[:, start:start + length, 0]


# ── U-Net ───────────────────────────────────────────────────────────────
def _conv_block(x: tf.Tensor, width: int, name: str) -> tf.Tensor:
    """兩層卷積 + BatchNorm。

    用 BatchNorm 而非 LayerNorm:頻譜圖的統計特性在 batch 之間相對穩定,
    而 BatchNorm 的記憶體開銷更小。
    """
    for i in (1, 2):
        x = layers.Conv2D(width, 3, padding="same", use_bias=False,
                          name=f"{name}_conv{i}")(x)
        x = layers.BatchNormalization(name=f"{name}_bn{i}")(x)
        x = layers.LeakyReLU(0.1, name=f"{name}_act{i}")(x)
    return x


def build_unet(config: ModelConfig) -> keras.Model:
    """幅度譜 (B, T, F, C) → 遮罩 (B, T, F, C),值域 [0, 1]。"""
    inputs = keras.Input(shape=(None, config.freq_bins, config.channels),
                         name="magnitude")

    # 對數壓縮:頻譜的動態範圍極大(80+ dB),直接餵原始幅度會讓少數高能量
    # 的時頻點主導梯度。log(1+x) 讓分佈接近常態,訓練穩定得多。
    #
    # 寫成 log(1+x) 而不是 log1p(x):tf2onnx 沒有 Log1p 的轉換規則,匯出的
    # 圖會帶著一個 ONNX Runtime 載不進去的節點。兩者在數學上相同,差別只在
    # x 極接近 0 時的精度 —— 而那個量級(1e-7)遠低於本來就存在的量化噪訊。
    x = tf.math.log(inputs + 1.0)

    widths = config.widths()
    skips = []
    for i, width in enumerate(widths):
        x = _conv_block(x, width, f"enc{i}")
        skips.append(x)
        x = layers.MaxPool2D(2, name=f"pool{i}")(x)

    bottleneck = min(int(widths[-1] * config.growth), config.max_channels)
    x = _conv_block(x, bottleneck, "bottleneck")

    for i, (width, skip) in enumerate(zip(reversed(widths), reversed(skips))):
        x = layers.Conv2DTranspose(width, 2, strides=2, name=f"up{i}")(x)
        x = layers.Concatenate(name=f"cat{i}")([x, skip])
        x = _conv_block(x, width, f"dec{i}")

    # 遮罩式輸出:模型只決定「這個時頻點有多少比例屬於人聲」,不必從零合成
    # 訊號。輸出天然被限制在輸入範圍內,訓練比直接回歸頻譜穩定得多。
    mask = layers.Conv2D(config.channels, 1, activation="sigmoid",
                         name="mask")(x)
    return keras.Model(inputs, mask, name="spectrogram_unet")


# ── 完整分離器 ──────────────────────────────────────────────────────────
def build_separator(config: ModelConfig, segment_samples: int,
                    unet: keras.Model | None = None) -> keras.Model:
    """波形進、波形出的完整模型。

    輸入 ``mixture``: (batch, channels, samples);輸出 ``vocals`` 與
    ``accompaniment``,形狀相同。這就是匯出成 ONNX 之後 C# 端看到的介面 ——
    訓練與推論用的是同一個圖,不存在「匯出時換了一套 STFT」的風險。

    片段長度固定。這個 U-Net 的尺寸對齊靠的是「幀數整除 2^depth」,做成
    動態軸會讓 Concat 在不整除的長度上炸掉。對呼叫端也沒有損失:這類模型
    本來就要分塊推論,固定塊長反而讓 ONNX Runtime 能預先配置。
    """
    unet = unet or build_unet(config)
    stft = ConvSTFT(config.n_fft, config.hop_length, name="stft")
    istft = ConvISTFT(config.n_fft, config.hop_length, name="istft")

    channels, bins = config.channels, config.freq_bins
    mixture = keras.Input(shape=(channels, segment_samples), name="mixture")

    # 每個聲道各自做 STFT:(B, C, N) → (B*C, N)
    flat = tf.reshape(mixture, [-1, segment_samples])
    real, imag = stft(flat)
    magnitude = tf.sqrt(real * real + imag * imag + 1e-12)

    n_frames = frames_for(segment_samples, config)
    # (B*C, T, F) → (B, C, T, F) → channels-last 的 (B, T, F, C)
    spec_shape = [-1, channels, n_frames, config.full_bins]
    magnitude = tf.transpose(tf.reshape(magnitude, spec_shape), [0, 2, 3, 1])

    mask_low = unet(magnitude[:, :, :bins, :])
    if bins < config.full_bins:
        # 沒進模型的那幾個 bin 遮罩給 0,也就是全部歸伴奏。預設設定下這只是
        # 最頂端一個 bin(22028 Hz 以上);真正的意義是在 max_bins 被調小時,
        # 行為是「保守地把不確定的頻段讓給伴奏」而不是讓它憑空進人聲軌。
        pad = config.full_bins - bins
        mask_low = tf.pad(mask_low, [[0, 0], [0, 0], [0, pad], [0, 0]])

    # 轉回 (B*C, T, F) 好套回頻譜
    mask = tf.reshape(tf.transpose(mask_low, [0, 3, 1, 2]),
                      [-1, n_frames, config.full_bins])

    vocals = istft(real * mask, imag * mask, segment_samples)
    vocals = tf.reshape(vocals, [-1, channels, segment_samples])

    # 伴奏在數學上就是「混音 − 人聲」。一起輸出可以讓呼叫端不必知道這件事,
    # 也不會兩邊算法不一致。
    accompaniment = mixture - vocals

    model = keras.Model(mixture, [
        layers.Activation("linear", name="vocals")(vocals),
        layers.Activation("linear", name="accompaniment")(accompaniment),
    ], name="ktisv_separator")
    model.unet = unet
    model.config = config
    model.segment_samples = segment_samples
    return model


def build(preset: str = "small", seconds: float = 3.0,
          samplerate: int = 44100) -> keras.Model:
    if preset not in PRESETS:
        raise ValueError(f"未知的預設值 {preset};可用:{list(PRESETS)}")
    config = PRESETS[preset]
    return build_separator(config, aligned_length(seconds, config, samplerate))


def config_to_dict(config: ModelConfig) -> dict:
    return asdict(config)


def config_from_dict(data: dict) -> ModelConfig:
    fields = {f for f in ModelConfig.__dataclass_fields__}
    return ModelConfig(**{k: v for k, v in data.items() if k in fields})


__all__ = [
    "ModelConfig", "PRESETS", "ConvSTFT", "ConvISTFT",
    "aligned_length", "frames_for", "build_unet", "build_separator", "build",
    "config_to_dict", "config_from_dict",
]
