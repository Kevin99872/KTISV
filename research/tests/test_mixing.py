"""合成訓練資料的正確性測試。

最關鍵的不變量是 ``mixture == vocals + accompaniment``。若不成立,訓練時
不會有任何錯誤訊息 —— 模型只會安靜地學出爛結果。這種 bug 必須靠測試抓。

    python -m tests.test_mixing
"""

from __future__ import annotations

import sys

import numpy as np

from ktisv_research import SAMPLE_RATE
from ktisv_research.mixing import (
    MixConfig, augment_vocals, is_mostly_silent, mix_stems,
    random_crop, rms, rms_db, set_rms_db,
)

FAILURES: list[str] = []
rng = np.random.default_rng(7)


def check(name: str, condition: bool, detail: str = "") -> None:
    mark = "ok  " if condition else "FAIL"
    print(f"  [{mark}] {name}{('  — ' + detail) if detail else ''}")
    if not condition:
        FAILURES.append(name)


def tone(freq: float, seconds: float = 3.0, amp: float = 0.3) -> np.ndarray:
    t = np.arange(int(seconds * SAMPLE_RATE)) / SAMPLE_RATE
    mono = amp * np.sin(2 * np.pi * freq * t)
    return np.column_stack([mono, mono]).astype(np.float32)


def fake_vocal(seconds: float = 3.0) -> np.ndarray:
    """帶諧波與包絡的假人聲 —— 比純正弦更接近真實訊號。"""
    t = np.arange(int(seconds * SAMPLE_RATE)) / SAMPLE_RATE
    signal = sum(0.3 / (k + 1) * np.sin(2 * np.pi * 220 * (k + 1) * t)
                 for k in range(5))
    envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 1.5 * t)
    mono = signal * envelope
    return np.column_stack([mono, mono]).astype(np.float32)


def fake_accompaniment(seconds: float = 3.0) -> np.ndarray:
    t = np.arange(int(seconds * SAMPLE_RATE)) / SAMPLE_RATE
    bass = 0.4 * np.sin(2 * np.pi * 80 * t)
    hats = 0.1 * rng.standard_normal(len(t)) * (np.sin(2 * np.pi * 4 * t) > 0.9)
    left = bass + hats
    right = bass - hats            # 伴奏有立體聲寬度
    return np.column_stack([left, right]).astype(np.float32)


# ── 核心不變量 ──────────────────────────────────────────────────────────
def test_additivity() -> None:
    """mixture 必須嚴格等於 vocals + accompaniment。"""
    print("可加性(訓練正確性的根本前提)")
    for trial in range(30):
        result = mix_stems(fake_vocal(), fake_accompaniment(),
                           rng=np.random.default_rng(trial))
        residual = result["mixture"] - result["vocals"] - result["accompaniment"]
        max_error = float(np.max(np.abs(residual)))
        if max_error > 1e-6:
            check(f"第 {trial} 次混音可加性", False, f"最大誤差 {max_error:.2e}")
            return
    check("30 次隨機混音都嚴格可加", True, "最大誤差 < 1e-6")


def test_no_clipping() -> None:
    """削波會產生真實音樂裡不存在的失真,模型會去學那個假象。"""
    print("防削波")
    peaks = []
    for trial in range(50):
        # 刻意用很大的輸入來逼出削波
        loud_vocal = fake_vocal() * 3.0
        loud_acc = fake_accompaniment() * 3.0
        result = mix_stems(loud_vocal, loud_acc,
                           rng=np.random.default_rng(trial))
        peaks.append(float(np.max(np.abs(result["mixture"]))))
    check("50 次大音量混音都沒有削波",
          max(peaks) <= 1.0, f"最大峰值 {max(peaks):.4f}")

    # 分軌也不能爆(它們是混音的組成部分)
    result = mix_stems(fake_vocal() * 5.0, fake_accompaniment() * 5.0,
                       rng=np.random.default_rng(99))
    check("分軌峰值也在範圍內",
          max(float(np.max(np.abs(result["vocals"]))),
              float(np.max(np.abs(result["accompaniment"])))) <= 2.0)


def test_snr_variation() -> None:
    """人聲/伴奏比例必須有變化,否則模型會學到固定比例而不是學會分離。"""
    print("SNR 多樣性")
    ratios = []
    for trial in range(60):
        result = mix_stems(fake_vocal(), fake_accompaniment(),
                           rng=np.random.default_rng(trial))
        v, a = rms(result["vocals"]), rms(result["accompaniment"])
        if v > 1e-9 and a > 1e-9:
            ratios.append(20 * np.log10(v / a))

    spread = max(ratios) - min(ratios)
    check("SNR 有足夠的分佈範圍", spread > 10.0,
          f"{min(ratios):.1f} ~ {max(ratios):.1f} dB(跨度 {spread:.1f} dB)")

    config = MixConfig(snr_db_range=(-5.0, 10.0))
    in_range = [r for r in ratios if -6.0 <= r <= 11.0]
    check("SNR 落在設定範圍內",
          len(in_range) >= len(ratios) * 0.95,
          f"{len(in_range)}/{len(ratios)} 在範圍內")


def test_loudness_normalisation() -> None:
    print("響度正規化")
    results = [mix_stems(fake_vocal(), fake_accompaniment(),
                         rng=np.random.default_rng(t)) for t in range(40)]
    levels = [rms_db(r["mixture"]) for r in results]
    config = MixConfig()
    within = [l for l in levels
              if abs(l - config.target_db) <= config.gain_jitter_db + 1.0]
    check("混音響度集中在目標值附近",
          len(within) >= len(levels) * 0.9,
          f"目標 {config.target_db} dB,實測 {min(levels):.1f} ~ {max(levels):.1f}")


def test_silence_detection() -> None:
    print("靜音偵測")
    silent = np.zeros((SAMPLE_RATE, 2), dtype=np.float32)
    check("全靜音被判定為靜音", is_mostly_silent(silent))
    check("正常人聲不被判為靜音", not is_mostly_silent(fake_vocal()))

    # 只有開頭 10% 有聲音 —— 這種樣本對訓練貢獻很低
    sparse = np.zeros((SAMPLE_RATE * 4, 2), dtype=np.float32)
    sparse[:SAMPLE_RATE // 2] = fake_vocal(0.5)
    check("稀疏樣本被判定為靜音", is_mostly_silent(sparse),
          "只有 12.5% 有聲音")


def test_rms_helpers() -> None:
    print("響度工具")
    signal = tone(440, 2.0, amp=0.5)
    adjusted = set_rms_db(signal, -20.0)
    check("set_rms_db 準確", abs(rms_db(adjusted) - (-20.0)) < 0.1,
          f"{rms_db(adjusted):.2f} dB")
    check("靜音訊號不會除以零",
          np.all(set_rms_db(np.zeros((100, 2), dtype=np.float32), -20.0) == 0))


def test_random_crop() -> None:
    print("隨機裁切")
    long_audio = fake_vocal(10.0)
    length = SAMPLE_RATE * 3
    crops = [random_crop(long_audio, length, np.random.default_rng(t))
             for t in range(20)]
    check("裁切長度正確", all(len(c) == length for c in crops))
    starts = {c[:100].tobytes() for c in crops}
    check("裁切位置有變化", len(starts) > 5, f"{len(starts)}/20 個不同起點")

    short = fake_vocal(1.0)
    padded = random_crop(short, length)
    check("過短的片段補零到指定長度", len(padded) == length)


def test_augment_preserves_structure() -> None:
    """增強不能破壞人聲的頻譜結構 —— 那正是模型要學的特徵。"""
    print("人聲增強")
    vocal = fake_vocal()
    augmented = [augment_vocals(vocal, np.random.default_rng(t))
                 for t in range(20)]

    def spectrum(x):
        return np.abs(np.fft.rfft(x[:, 0]))

    original = spectrum(vocal)
    check("頻譜幅度不變(只做極性/聲道操作)",
          all(np.allclose(spectrum(a), original, rtol=1e-4) for a in augmented))
    check("確實有產生變化",
          any(not np.array_equal(a, vocal) for a in augmented))


def test_length_mismatch() -> None:
    print("長度不一致")
    result = mix_stems(fake_vocal(5.0), fake_accompaniment(3.0))
    n = len(result["mixture"])
    check("以較短者為準", n == len(fake_accompaniment(3.0)), f"{n} 取樣")
    check("三者長度一致",
          len(result["vocals"]) == n and len(result["accompaniment"]) == n)
    residual = result["mixture"] - result["vocals"] - result["accompaniment"]
    check("長度不同時仍維持可加性", float(np.max(np.abs(residual))) < 1e-6)


def test_mono_input() -> None:
    print("單聲道輸入")
    mono_vocal = fake_vocal()[:, :1]
    result = mix_stems(mono_vocal, fake_accompaniment())
    check("單聲道自動轉立體聲", result["mixture"].shape[1] == 2)
    residual = result["mixture"] - result["vocals"] - result["accompaniment"]
    check("轉換後仍維持可加性", float(np.max(np.abs(residual))) < 1e-6)


def main() -> int:
    tests = (
        test_additivity, test_no_clipping, test_snr_variation,
        test_loudness_normalisation, test_silence_detection, test_rms_helpers,
        test_random_crop, test_augment_preserves_structure,
        test_length_mismatch, test_mono_input,
    )
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
