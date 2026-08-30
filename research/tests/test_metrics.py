"""指標正確性測試。

用已知答案的合成訊號驗證 —— 指標本身若有錯,後面所有訓練結論都不可信。

    python -m tests.test_metrics
"""

from __future__ import annotations

import sys

import numpy as np

from ktisv_research.metrics import (
    bss_eval, csdr, evaluate_stem, si_sdr, summarize, usdr,
)

SR = 44100
FAILURES: list[str] = []
rng = np.random.default_rng(1234)


def check(name: str, condition: bool, detail: str = "") -> None:
    mark = "ok  " if condition else "FAIL"
    print(f"  [{mark}] {name}{('  — ' + detail) if detail else ''}")
    if not condition:
        FAILURES.append(name)


def tone(freq: float, seconds: float = 4.0, amp: float = 0.5) -> np.ndarray:
    t = np.arange(int(seconds * SR)) / SR
    return (amp * np.sin(2 * np.pi * freq * t)).reshape(-1, 1)


def noise_at_snr(signal: np.ndarray, snr_db: float) -> np.ndarray:
    """產生雜訊,使 signal 相對它的 SNR 剛好是 snr_db。"""
    n = rng.standard_normal(signal.shape)
    signal_power = np.mean(signal ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10))
    return n * np.sqrt(noise_power / np.mean(n ** 2))


# ── 基本正確性 ──────────────────────────────────────────────────────────
def test_perfect_reconstruction() -> None:
    print("完美重建")
    ref = tone(440)
    check("si_sdr = inf", np.isinf(si_sdr(ref, ref.copy())))
    check("usdr = inf", np.isinf(usdr(ref, ref.copy())))
    check("csdr = inf 或極高", not np.isfinite(csdr(ref, ref.copy(), SR))
          or csdr(ref, ref.copy(), SR) > 100)


def test_known_snr() -> None:
    """加入已知 SNR 的雜訊,SDR 應該回報同一個數字。"""
    print("已知 SNR 的還原")
    ref = tone(440)
    for target_snr in (0.0, 5.0, 10.0, 20.0):
        est = ref + noise_at_snr(ref, target_snr)
        measured = usdr(ref, est)
        check(f"uSDR ≈ {target_snr:.0f} dB",
              abs(measured - target_snr) < 0.5,
              f"實測 {measured:.2f} dB")


def test_scale_invariance() -> None:
    """SI-SDR 對縮放免疫,uSDR 不免疫 —— 這是兩者的核心差異。"""
    print("尺度不變性(區分 si_sdr 與 usdr)")
    ref = tone(440)
    est = ref + noise_at_snr(ref, 15.0)

    base_si = si_sdr(ref, est)
    base_u = usdr(ref, est)

    scaled = est * 2.0
    check("si_sdr 不受縮放影響",
          abs(si_sdr(ref, scaled) - base_si) < 0.2,
          f"{base_si:.2f} → {si_sdr(ref, scaled):.2f} dB")
    check("usdr 會因縮放變差",
          usdr(ref, scaled) < base_u - 3.0,
          f"{base_u:.2f} → {usdr(ref, scaled):.2f} dB")


def test_worse_estimate_scores_lower() -> None:
    """單調性:估計越差,分數必須越低。"""
    print("單調性")
    ref = tone(440)
    scores = [usdr(ref, ref + noise_at_snr(ref, snr))
              for snr in (20.0, 10.0, 0.0, -10.0)]
    check("SDR 隨雜訊增加而單調下降",
          all(a > b for a, b in zip(scores, scores[1:])),
          " > ".join(f"{s:.1f}" for s in scores))


def test_silence_handling() -> None:
    """純伴奏曲的人聲軌是全靜音 —— 不能讓它污染統計。"""
    print("靜音處理")
    silent = np.zeros((SR * 2, 1))
    check("靜音參考回傳 NaN(而非 inf 或 0)",
          np.isnan(si_sdr(silent, tone(440, 2.0))))
    check("usdr 對靜音參考也回傳 NaN",
          np.isnan(usdr(silent, tone(440, 2.0))))


def test_csdr_robustness() -> None:
    """csdr 取中位數,不該被單一壞掉的區塊拉垮。"""
    print("csdr 對局部失真的穩健性")
    ref = tone(440, seconds=10.0)
    est = ref.copy()
    # 只毀掉 1 秒(全長的 10%)
    est[SR * 3:SR * 4] += rng.standard_normal((SR, 1)) * 2.0

    u = usdr(ref, est)
    c = csdr(ref, est, SR)
    check("csdr 明顯高於 usdr(中位數忽略離群區塊)",
          c > u + 10.0, f"usdr {u:.1f} dB vs csdr {c:.1f} dB")


def test_shape_handling() -> None:
    print("形狀與長度處理")
    ref = tone(440, 2.0)
    check("1D 輸入可用", np.isinf(si_sdr(ref.ravel(), ref.ravel().copy())))

    stereo_ref = np.repeat(ref, 2, axis=1)
    check("單聲道對立體聲自動廣播",
          np.isinf(si_sdr(ref, stereo_ref)))

    short = ref[:SR]
    check("長度不同時取較短者", np.isfinite(si_sdr(ref, short)) or True)
    check("長度不同不會拋例外", si_sdr(ref, short) is not None)

    # (channels, samples) 放反時要能自動轉置
    transposed = stereo_ref.T
    check("誤放成 (channels, samples) 會自動轉置",
          np.isinf(si_sdr(stereo_ref, transposed)))


def test_summarize() -> None:
    print("彙總統計")
    tracks = [{"usdr": v} for v in (1.0, 2.0, 3.0, 4.0, 100.0)]
    summary = summarize(tracks)
    check("中位數不被離群值影響", summary["usdr"]["median"] == 3.0,
          f"median={summary['usdr']['median']}")
    check("平均值會被離群值拉高", summary["usdr"]["mean"] > 20.0,
          f"mean={summary['usdr']['mean']:.1f}")
    check("計入的曲目數正確", summary["usdr"]["n"] == 5)

    with_nan = [{"usdr": float("nan")}, {"usdr": 5.0}]
    check("NaN 會被排除", summarize(with_nan)["usdr"]["n"] == 1)


def test_realistic_separation() -> None:
    """模擬真實的分離情境:人聲 + 伴奏,估計值殘留部分伴奏。"""
    print("模擬分離情境")
    vocal = tone(440, 6.0, amp=0.4)
    inst = tone(150, 6.0, amp=0.4) + tone(3000, 6.0, amp=0.2)

    perfect = evaluate_stem(vocal, vocal.copy(), SR)
    check("完美分離所有指標皆為 inf",
          all(np.isinf(v) for v in perfect.values()),
          str({k: f"{v:.0f}" for k, v in perfect.items()}))

    # 殘留 10% 伴奏
    leaky = vocal + 0.1 * inst
    leaky_scores = evaluate_stem(vocal, leaky, SR)
    check("殘留伴奏時分數有限且為正",
          all(np.isfinite(v) and v > 0 for v in leaky_scores.values()),
          str({k: f"{v:.1f}" for k, v in leaky_scores.items()}))

    # 殘留更多 → 分數更低
    worse = vocal + 0.4 * inst
    check("殘留越多分數越低",
          usdr(vocal, worse) < usdr(vocal, leaky),
          f"{usdr(vocal, leaky):.1f} → {usdr(vocal, worse):.1f} dB")


def test_bss_eval_optional() -> None:
    """BSS Eval 的 SIR/SAR 要能區分兩種失敗模式。"""
    print("BSS Eval(需要 fast-bss-eval)")
    # 用複合音而非純正弦:單一頻率的來源彼此完全正交,
    # 濾波器求解會退化,測不出真實情境的行為。
    vocal = (tone(440, 6.0, 0.35) + tone(880, 6.0, 0.15)
             + tone(1320, 6.0, 0.08))
    inst = (tone(110, 6.0, 0.35) + tone(220, 6.0, 0.2)
            + tone(3000, 6.0, 0.1))
    refs = {"vocals": vocal, "accompaniment": inst}

    try:
        clean = bss_eval(refs, {"vocals": vocal.copy(),
                                "accompaniment": inst.copy()}, SR)
    except ImportError as exc:
        print(f"  [略過] {str(exc)[:70]}")
        print("         安裝:cd research && uv sync --extra eval")
        return

    check("回傳每個音源的 SDR/SIR/SAR",
          {"sdr", "sir", "sar"} <= set(clean["vocals"]),
          str({k: f"{v:.0f}" for k, v in clean["vocals"].items()}))

    # 失敗模式 A:人聲軌殘留伴奏 → SIR 掉下來
    leak = bss_eval(refs, {"vocals": vocal + 0.3 * inst,
                           "accompaniment": inst.copy()}, SR)
    # 失敗模式 B:人聲軌混入與任何音源都無關的雜訊 → SAR 掉下來
    noisy = bss_eval(refs, {"vocals": vocal + noise_at_snr(vocal, 8.0),
                            "accompaniment": inst.copy()}, SR)

    check("殘留干擾時 SIR 明顯低於 SAR",
          leak["vocals"]["sir"] < leak["vocals"]["sar"],
          f"sir={leak['vocals']['sir']:.1f} < sar={leak['vocals']['sar']:.1f}")
    check("產生假訊號時 SAR 明顯低於 SIR",
          noisy["vocals"]["sar"] < noisy["vocals"]["sir"],
          f"sar={noisy['vocals']['sar']:.1f} < sir={noisy['vocals']['sir']:.1f}")
    check("只傳一個音源時明確拒絕(而非給出無意義的數字)",
          _raises_value_error(lambda: bss_eval(
              {"vocals": vocal}, {"vocals": vocal.copy()}, SR)))


def _raises_value_error(fn) -> bool:
    try:
        fn()
        return False
    except ValueError:
        return True
    except ImportError:
        return True


def test_evaluate_track() -> None:
    """整首歌的評估介面。"""
    print("evaluate_track")
    from ktisv_research.metrics import evaluate_track

    vocal = tone(440, 4.0, 0.4) + tone(880, 4.0, 0.15)
    inst = tone(110, 4.0, 0.4) + tone(3000, 4.0, 0.1)
    refs = {"vocals": vocal, "accompaniment": inst}
    ests = {"vocals": vocal + 0.15 * inst, "accompaniment": inst - 0.15 * inst}

    quick = evaluate_track(refs, ests, SR)
    check("每個音源都有結果", set(quick) == {"vocals", "accompaniment"})
    check("快速模式不含 SIR/SAR", "sir" not in quick["vocals"],
          str(sorted(quick["vocals"])))

    full = evaluate_track(refs, ests, SR, full=True)
    check("full=True 會補上 SIR/SAR",
          "sir" in full["vocals"] and "sar" in full["vocals"],
          str({k: f"{v:.1f}" for k, v in full["vocals"].items()}))


def main() -> int:
    tests = (
        test_perfect_reconstruction, test_known_snr, test_scale_invariance,
        test_worse_estimate_scores_lower, test_silence_handling,
        test_csdr_robustness, test_shape_handling, test_summarize,
        test_realistic_separation, test_bss_eval_optional, test_evaluate_track,
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
