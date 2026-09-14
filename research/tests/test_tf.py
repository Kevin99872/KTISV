"""TensorFlow 管線的正確性測試。

    .venv-tf\\Scripts\\python -m tests.test_tf

三個問題要回答:
  1. 卷積版的 STFT / iSTFT 真的是 STFT 嗎?(往返無損 + 對得上 tf.signal)
  2. 這個架構能在 6 GB VRAM 內訓練嗎?(實測,不是估算)
  3. 訓練機制是對的嗎?(能不能過擬合單一樣本 —— 標準的健全性檢查)

第 1 項是這一版最需要驗的東西。PyTorch 版訓練用 ``torch.stft``、匯出才換成
卷積,所以「兩條路徑一不一致」是它的風險;TF 版從頭到尾只有卷積這一條路,
風險就轉移成「這條路本身對不對」。往返無損是個自足的檢查:它不需要參考
實作,錯了就一定不會過。

要在 ``.venv-tf`` 裡跑,不是 ``.venv``。
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# 這個 import 的順序是有意義的,不要調動:ktisv_tf/__init__.py 會把 CUDA 11
# 的 DLL 目錄掛進 PATH,而 TensorFlow 在**載入時**就會去 dlopen 那些函式庫。
# 反過來先 import tensorflow 的話,它會抓到系統上 CUDA 13 的 cudart,
# 然後在第一次真的用到 GPU 時才炸(cudaGetErrorString symbol not found)。
import ktisv_tf                                                # noqa: E402,F401

import tensorflow as tf                                        # noqa: E402

from ktisv_research.metrics import si_sdr                       # noqa: E402
from ktisv_tf.model import (PRESETS, ConvISTFT, ConvSTFT,  # noqa: E402
                            aligned_length, build_separator, build_unet,
                            frames_for)

FAILURES: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    mark = "ok  " if condition else "FAIL"
    print(f"  [{mark}] {name}{('  — ' + detail) if detail else ''}")
    if not condition:
        FAILURES.append(name)


def device() -> str:
    return "GPU" if tf.config.list_physical_devices("GPU") else "CPU"


# ── STFT ────────────────────────────────────────────────────────────────
def test_stft_roundtrip() -> None:
    """STFT → iSTFT 必須無損,否則模型還沒開始學就已經有誤差。"""
    print("卷積 STFT / iSTFT")
    n_fft, hop = 2048, 512
    stft, istft = ConvSTFT(n_fft, hop), ConvISTFT(n_fft, hop)

    rng = np.random.default_rng(0)
    length = 130560
    signal = (rng.standard_normal((3, length)) * 0.2).astype(np.float32)

    real, imag = stft(tf.constant(signal))
    check("幀數符合 center=True 的公式",
          int(real.shape[1]) == length // hop + 1,
          f"{int(real.shape[1])} 幀")

    recovered = istft(real, imag, length).numpy()
    error = float(np.abs(recovered - signal).max())
    check("往返無損", error < 1e-5, f"最大誤差 {error:.2e}")


def test_stft_matches_reference() -> None:
    """和 tf.signal.stft 比對 —— 卷積核的係數、窗函數、補零方式都要對。

    這裡不是「兩份實作互相驗證」,而是拿一份公認正確的實作當尺。差異只該
    來自浮點累加順序。
    """
    print("與 tf.signal.stft 比對")
    n_fft, hop = 2048, 512
    stft = ConvSTFT(n_fft, hop)

    rng = np.random.default_rng(1)
    signal = (rng.standard_normal((2, 65536)) * 0.2).astype(np.float32)
    real, imag = stft(tf.constant(signal))

    reference = tf.signal.stft(
        tf.pad(tf.constant(signal), [[0, 0], [n_fft // 2, n_fft // 2]],
               mode="REFLECT"),
        frame_length=n_fft, frame_step=hop, fft_length=n_fft,
        window_fn=tf.signal.hann_window).numpy()

    # 頻譜的量級在數百,所以用相對誤差看才有意義
    scale = float(np.abs(reference).max())
    worst = max(float(np.abs(reference.real - real.numpy()).max()),
                float(np.abs(reference.imag - imag.numpy()).max()))
    check("實部與虛部都對得上", worst / scale < 1e-5,
          f"相對誤差 {worst / scale:.2e}")


# ── 模型 ────────────────────────────────────────────────────────────────
def test_shapes_and_additivity() -> None:
    print("形狀與可加性")
    config = PRESETS["tiny"]
    length = aligned_length(1.0, config)
    model = build_separator(config, length)

    rng = np.random.default_rng(2)
    waveform = (rng.standard_normal((2, 2, length)) * 0.1).astype(np.float32)
    vocals, accompaniment = model(waveform, training=False)

    check("人聲輸出形狀正確", tuple(vocals.shape) == waveform.shape,
          str(tuple(vocals.shape)))
    check("伴奏輸出形狀正確", tuple(accompaniment.shape) == waveform.shape)

    # 架構上由 accompaniment = mixture - vocals 保證
    residual = float(np.abs(vocals.numpy() + accompaniment.numpy()
                            - waveform).max())
    check("vocals + accompaniment == 輸入", residual < 1e-5,
          f"最大誤差 {residual:.2e}")


def test_alignment() -> None:
    """對齊長度必須讓幀數整除 2^depth,否則 U-Net 的 skip 會對不上。"""
    print("片段長度對齊")
    for name, config in PRESETS.items():
        for seconds in (1.0, 3.0, 6.0, 10.0):
            length = aligned_length(seconds, config)
            frames = frames_for(length, config)
            if frames % config.stride:
                check(f"{name} @ {seconds}s", False,
                      f"{frames} 幀不能被 {config.stride} 整除")
                return
    check("所有預設值與長度組合都整除", True)


def test_mask_range() -> None:
    """遮罩必須落在 [0,1] —— 這是「不會憑空生出東西」的保證。"""
    print("遮罩值域")
    config = PRESETS["tiny"]
    unet = build_unet(config)
    rng = np.random.default_rng(3)
    magnitude = np.abs(rng.standard_normal(
        (1, 32, config.freq_bins, config.channels)) * 5).astype(np.float32)
    mask = unet(magnitude, training=False).numpy()
    check("遮罩在 [0,1]", bool((mask >= 0).all() and (mask <= 1).all()),
          f"[{mask.min():.3f}, {mask.max():.3f}]")


def test_length_transfer() -> None:
    """同一份權重套到不同片段長度,輸出是不是同一個聲音。

    ``export.py`` 用「訓練 3 秒、匯出 6 秒」的做法,前提是權重與長度無關。
    形狀上確實無關(全卷積 + 常數 STFT),但**數值上不是逐點相同的** ——
    這一點值得說清楚,因為直覺會以為應該相同:

    U-Net 的感受野比片段還長(depth 5 的下採樣把 3 秒的 256 幀壓到 8 幀,
    再算上每層的 3×3 卷積,一個輸出點會看到整段)。所以換一個長度,
    邊界的補零條件就變了,而那個差異會傳遍整段 —— 不是只有頭尾。
    加大要比對的 margin 幾乎沒有幫助,實測可以證實這件事。

    所以這裡驗的不是「逐點相同」,而是**「聽起來是同一個東西」**:
    兩者之間的 SI-SDR。40 dB 以上的差異遠在可聽閾之下;實測隨機初始化的
    模型落在 46–52 dB,門檻設 30 dB 留足餘裕。
    """
    print("換片段長度的一致性")
    for name in ("tiny", "small"):
        config = PRESETS[name]
        unet = build_unet(config)
        short_len = aligned_length(3.0, config)      # 訓練用的長度
        long_len = aligned_length(6.0, config)       # 匯出用的長度

        rng = np.random.default_rng(4)
        waveform = (rng.standard_normal((1, 2, long_len))
                    * 0.1).astype(np.float32)

        long_out = build_separator(config, long_len, unet)(
            waveform, training=False)[0].numpy()
        short_out = build_separator(config, short_len, unet)(
            waveform[:, :, :short_len], training=False)[0].numpy()

        agreement = si_sdr(long_out[0, :, :short_len].T,
                           short_out[0, :, :short_len].T)
        check(f"{name}:3 秒與 6 秒的輸出一致", agreement > 30.0,
              f"SI-SDR {agreement:.1f} dB")


# ── 資源與訓練機制 ──────────────────────────────────────────────────────
def test_memory_footprint() -> None:
    """實測一步訓練要多少 VRAM。目標是 6 GB 的卡裝得下。"""
    print("記憶體用量(訓練一步)")
    if not tf.config.list_physical_devices("GPU"):
        check("跳過(沒有 GPU)", True)
        return

    config = PRESETS["small"]
    length = aligned_length(3.0, config)
    model = build_separator(config, length)
    optimizer = tf.keras.optimizers.Adam(1e-4)

    rng = np.random.default_rng(5)
    batch = (rng.standard_normal((4, 2, length)) * 0.1).astype(np.float32)
    target = (rng.standard_normal((4, 2, length)) * 0.1).astype(np.float32)

    with tf.GradientTape() as tape:
        estimate, _ = model(batch, training=True)
        loss = tf.reduce_mean(tf.abs(estimate - target))
    optimizer.apply_gradients(
        zip(tape.gradient(loss, model.trainable_variables),
            model.trainable_variables))

    peak = tf.config.experimental.get_memory_info("GPU:0")["peak"] / 1e9
    check("small / batch 4 / 3 秒片段在 6 GB 內", peak < 5.5,
          f"尖峰 {peak:.2f} GB")


def test_can_overfit() -> None:
    """能不能把單一樣本學到幾乎完美 —— 學不起來代表訓練機制本身有問題。

    這個檢查抓的是「梯度沒接上」「損失算錯軸」這類錯誤:它們不會讓程式
    崩潰,只會讓訓練看起來在跑、實際什麼都沒學到。
    """
    print("過擬合單一樣本")
    config = PRESETS["tiny"]
    length = aligned_length(1.0, config)
    model = build_separator(config, length)
    optimizer = tf.keras.optimizers.Adam(3e-3)

    # 合成一個好分的例子:人聲是 800 Hz、伴奏是 90 Hz,頻率上完全不重疊
    t = np.arange(length, dtype=np.float32) / 44100.0
    vocals = np.stack([np.sin(2 * np.pi * 800 * t)] * 2)[None] * 0.3
    bass = np.stack([np.sin(2 * np.pi * 90 * t)] * 2)[None] * 0.3
    mixture = (vocals + bass).astype(np.float32)
    vocals = vocals.astype(np.float32)

    @tf.function
    def step():
        with tf.GradientTape() as tape:
            estimate, _ = model(mixture, training=True)
            loss = tf.reduce_mean(tf.abs(estimate - vocals))
        optimizer.apply_gradients(
            zip(tape.gradient(loss, model.trainable_variables),
                model.trainable_variables))
        return loss

    started = time.time()
    first = float(step())
    for _ in range(199):
        last = float(step())
    check("損失明顯下降", last < first * 0.5,
          f"{first:.4f} → {last:.4f}({time.time() - started:.0f}s)")


def main() -> int:
    print(f"TensorFlow {tf.__version__} / 裝置:{device()}\n")
    tests = (test_stft_roundtrip, test_stft_matches_reference,
             test_shapes_and_additivity, test_alignment, test_mask_range,
             test_length_transfer, test_memory_footprint,
             test_can_overfit)
    for fn in tests:
        fn()
        print()

    if FAILURES:
        print(f"{len(FAILURES)} 項未通過: " + ", ".join(FAILURES))
        return 1
    print("全部通過。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
