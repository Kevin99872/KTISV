"""頻譜域分離器:正確性 + 與時域版本的量化比較。

為什麼這裡可以用合成訊號
------------------------
先前用合成音測 Demucs 是無效的 —— 那是學習模型,需要「認得」人聲的音色。

但這兩個分離器是**純空間 DSP**:它們只看左右聲道的相位與振幅關係,
完全不管內容是什麼。所以只要合成訊號的**空間配置**符合真實混音
(人聲置中、樂器分佈兩側),測試就有效。

    python -m tests.test_spectral_separation
"""

from __future__ import annotations

import sys

import numpy as np

from ktisv_engine.dsp.separation import CenterSeparator
from ktisv_engine.dsp.spectral_separation import SpectralCenterSeparator

SR = 48000
BLOCK = 480
FAILURES: list[str] = []
rng = np.random.default_rng(42)


def check(name: str, condition: bool, detail: str = "") -> None:
    mark = "ok  " if condition else "FAIL"
    print(f"  [{mark}] {name}{('  — ' + detail) if detail else ''}")
    if not condition:
        FAILURES.append(name)


def db(x: np.ndarray) -> float:
    value = float(np.sqrt(np.mean(np.square(x, dtype=np.float64))))
    return 20.0 * np.log10(max(value, 1e-12))


def sdr(reference: np.ndarray, estimate: np.ndarray) -> float:
    n = min(len(reference), len(estimate))
    ref, est = reference[:n], estimate[:n]
    error = float(np.sum((ref - est) ** 2))
    signal = float(np.sum(ref ** 2))
    if error < 1e-15:
        return float("inf")
    return 10.0 * np.log10(signal / max(error, 1e-15))


def harmonic(freq: float, seconds: float, amp: float,
             partials: int = 5) -> np.ndarray:
    t = np.arange(int(seconds * SR)) / SR
    signal = sum((0.6 ** k) * np.sin(2 * np.pi * freq * (k + 1) * t)
                 for k in range(partials))
    return (amp * signal / np.max(np.abs(signal))).astype(np.float64)


def pan(mono: np.ndarray, position: float) -> np.ndarray:
    """position: -1 全左, 0 置中, +1 全右(等功率定位)。"""
    angle = (position + 1.0) * np.pi / 4.0
    return np.column_stack([mono * np.cos(angle), mono * np.sin(angle)])


def build_scene(seconds: float = 6.0) -> dict[str, np.ndarray]:
    """模擬真實混音的空間配置。"""
    vocal_mono = harmonic(240.0, seconds, 0.35)          # 主唱
    bass_mono = harmonic(70.0, seconds, 0.35, partials=3)  # 貝斯:置中低頻
    guitar_mono = harmonic(500.0, seconds, 0.25)          # 吉他:偏左
    keys_mono = harmonic(700.0, seconds, 0.25)            # 鍵盤:偏右
    cymbal = (0.12 * rng.standard_normal(int(seconds * SR)))  # 鈸:置中高頻

    vocals = pan(vocal_mono, 0.0)                # 置中 —— 要被移除的
    accompaniment = (pan(bass_mono, 0.0)         # 置中但低頻 —— 要保留
                     + pan(guitar_mono, -0.6)    # 偏左 —— 要保留
                     + pan(keys_mono, 0.6)       # 偏右 —— 要保留
                     + pan(cymbal, 0.0) * 0.5)   # 置中但高頻 —— 要保留

    return {"vocals": vocals, "accompaniment": accompaniment,
            "mixture": vocals + accompaniment}


def run_blocks(separator, audio: np.ndarray) -> np.ndarray:
    """逐 block 送進分離器,模擬即時處理。"""
    out = []
    for start in range(0, len(audio) - BLOCK + 1, BLOCK):
        block = audio[start:start + BLOCK].astype(np.float32)
        out.append(np.asarray(separator.process(block), dtype=np.float64))
    return np.concatenate(out) if out else np.zeros((0, 2))


def measure_latency(make_separator) -> int:
    """用脈衝實測延遲。

    不直接用 ``latency_samples`` —— 那只算演算法本身,block 大小不是 hop
    的整數倍時還會有量化延遲。測試要對齊的是**實際**延遲。
    """
    sep = make_separator()
    sep.mode = "isolate_vocals"
    if hasattr(sep, "sharpness"):
        sep.sharpness = 0.0
        sep._band_weight[:] = 1.0

    n = 1 << 16
    impulse = np.zeros((n, 2), dtype=np.float32)
    position = n // 4
    impulse[position] = 1.0
    out = run_blocks(sep, impulse)
    if not len(out) or float(np.max(np.abs(out))) < 1e-9:
        return 0
    return int(np.argmax(np.abs(out[:, 0]))) - position


def align(reference: np.ndarray, estimate: np.ndarray,
          latency: int, skip: int = 4096) -> tuple[np.ndarray, np.ndarray]:
    """依實測延遲對齊,並跳過開頭的暖機段。"""
    n = min(len(reference) - skip, len(estimate) - skip - latency)
    if n <= 0:
        return reference[:0], estimate[:0]
    return (reference[skip:skip + n],
            estimate[skip + latency:skip + latency + n])


# ── 正確性 ──────────────────────────────────────────────────────────────
def test_bypass() -> None:
    print("旁通")
    sep = SpectralCenterSeparator(SR)
    sep.mode = "off"
    scene = build_scene(1.0)
    out = run_blocks(sep, scene["mixture"])
    reference = scene["mixture"][:len(out)]
    check("mode=off 完全旁通", np.allclose(out, reference, atol=1e-6))


def test_silence_mode() -> None:
    print("靜音模式")
    sep = SpectralCenterSeparator(SR)
    sep.mode = "silence"
    out = run_blocks(sep, build_scene(1.0)["mixture"])
    check("mode=silence 輸出靜音", db(out) < -100, f"{db(out):.1f} dBFS")


def test_reconstruction() -> None:
    """重疊相加必須能無損重建 —— 否則還沒開始分離就已經有失真。"""
    print("重疊相加重建")
    sep = SpectralCenterSeparator(SR)
    sep.mode = "isolate_vocals"
    sep.sharpness = 0.0          # 讓遮罩恆為 1
    sep._band_weight[:] = 1.0

    latency = measure_latency(lambda: SpectralCenterSeparator(SR))
    mono = harmonic(300.0, 3.0, 0.4)
    centered = pan(mono, 0.0)
    out = run_blocks(sep, centered)

    ref, est = align(centered, out, latency)
    score = sdr(ref, est)
    check("置中訊號能完整還原", score > 40.0,
          f"SDR {score:.1f} dB(實測延遲 {latency} 取樣)")


def test_latency_reported() -> None:
    """回報的延遲要和實測相符(容許 block 量化造成的差距)。"""
    print("延遲")
    for n_fft in (512, 1024, 2048):
        sep = SpectralCenterSeparator(SR, n_fft=n_fft)
        measured = measure_latency(
            lambda nf=n_fft: SpectralCenterSeparator(SR, n_fft=nf))
        reported = sep.latency_samples
        extra = measured - reported
        check(f"n_fft={n_fft}: 回報 {sep.latency_ms:.1f} ms,"
              f"實測 {measured / SR * 1000:.1f} ms",
              0 <= extra <= BLOCK,
              f"量化多出 {extra} 取樣(≤ 一個 block {BLOCK})")


# ── 量化比較 ────────────────────────────────────────────────────────────
def compare() -> dict:
    """在同一個場景上比較兩個演算法。"""
    scene = build_scene(8.0)
    results = {}

    for label, factory in (
        ("時域 mid/side",
         lambda: CenterSeparator(SR, low_cut=180.0, high_cut=9000.0)),
        ("頻譜域逐格",
         lambda: SpectralCenterSeparator(SR, n_fft=1024,
                                         low_cut=180.0, high_cut=9000.0)),
    ):
        latency = measure_latency(factory)
        sep = factory()
        sep.mode = "remove_vocals"
        sep.strength = 1.0
        out = run_blocks(sep, scene["mixture"])

        target, estimate = align(scene["accompaniment"], out, latency)
        results[label] = {
            "sdr": sdr(target, estimate),
            "accompaniment_kept": db(estimate) - db(target),
            "residual_db": db(estimate - target),
            "latency": latency,
        }
    return results


def test_comparison() -> None:
    print("演算法比較(去人聲,保留伴奏)")
    results = compare()

    print(f"  {'演算法':<16}{'伴奏 SDR':>10}{'伴奏保留':>10}{'殘差':>10}")
    print("  " + "─" * 46)
    for label, scores in results.items():
        print(f"  {label:<16}{scores['sdr']:>9.2f}dB"
              f"{scores['accompaniment_kept']:>9.2f}dB"
              f"{scores['residual_db']:>9.1f}dB")

    old = results["時域 mid/side"]
    new = results["頻譜域逐格"]

    check("頻譜域的伴奏 SDR 更高",
          new["sdr"] > old["sdr"],
          f"{old['sdr']:.2f} → {new['sdr']:.2f} dB "
          f"(+{new['sdr'] - old['sdr']:.2f})")
    check("頻譜域更能保住伴奏音量",
          abs(new["accompaniment_kept"]) < abs(old["accompaniment_kept"]),
          f"{old['accompaniment_kept']:.2f} → {new['accompaniment_kept']:.2f} dB")


def test_panned_instruments_preserved() -> None:
    """核心訴求:偏一邊的樂器不該被削掉。"""
    print("偏側樂器的保留程度")
    seconds = 6.0
    guitar = pan(harmonic(500.0, seconds, 0.3), -0.6)
    vocal = pan(harmonic(240.0, seconds, 0.35), 0.0)
    mixture = guitar + vocal

    kept_by = {}
    for label, factory in (
        ("時域 mid/side", lambda: CenterSeparator(SR)),
        ("頻譜域逐格", lambda: SpectralCenterSeparator(SR)),
    ):
        latency = measure_latency(factory)
        sep = factory()
        sep.mode = "remove_vocals"
        out = run_blocks(sep, mixture)
        ref, est = align(guitar, out, latency)
        kept_by[label] = db(est) - db(ref)
        print(f"    {label:<16}吉他保留 {kept_by[label]:+.2f} dB")

    time_kept = kept_by["時域 mid/side"]
    spectral_kept = kept_by["頻譜域逐格"]

    check("頻譜域保留偏側樂器優於時域",
          spectral_kept > time_kept,
          f"{time_kept:+.2f} → {spectral_kept:+.2f} dB")
    check("頻譜域的偏側樂器損失在 3 dB 內",
          abs(spectral_kept) < 3.0, f"{spectral_kept:+.2f} dB")


def test_centered_bass_preserved() -> None:
    """置中的貝斯必須靠頻段權重保住。"""
    print("置中低頻的保留")
    seconds = 6.0
    bass = pan(harmonic(70.0, seconds, 0.4, partials=3), 0.0)
    vocal = pan(harmonic(240.0, seconds, 0.35), 0.0)
    mixture = bass + vocal

    latency = measure_latency(lambda: SpectralCenterSeparator(SR, low_cut=180.0))
    sep = SpectralCenterSeparator(SR, low_cut=180.0)
    sep.mode = "remove_vocals"
    out = run_blocks(sep, mixture)
    ref, est = align(bass, out, latency)
    kept = db(est) - db(ref)
    check("置中貝斯幾乎完整保留", kept > -3.0, f"{kept:+.2f} dB")


def main() -> int:
    tests = (test_bypass, test_silence_mode, test_reconstruction,
             test_latency_reported, test_comparison,
             test_panned_instruments_preserved, test_centered_bass_preserved)
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
