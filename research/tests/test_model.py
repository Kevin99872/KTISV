"""模型正確性與資源用量測試。

兩個問題要回答:
  1. 這個架構真的能在 6 GB VRAM 內訓練嗎?(實測,不是估算)
  2. 訓練機制是對的嗎?(能不能過擬合單一樣本 —— 這是標準的健全性檢查)

    python -m tests.test_model
"""

from __future__ import annotations

import sys
import time

import numpy as np
import torch

from ktisv_research.model import PRESETS, ModelConfig, Separator, build

FAILURES: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    mark = "ok  " if condition else "FAIL"
    print(f"  [{mark}] {name}{('  — ' + detail) if detail else ''}")
    if not condition:
        FAILURES.append(name)


def device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


# ── 正確性 ──────────────────────────────────────────────────────────────
def test_shapes() -> None:
    print("形狀與值域")
    model = build("tiny")
    wav = torch.randn(2, 2, 44100)          # 2 個樣本、立體聲、1 秒
    out = model(wav)

    check("人聲輸出形狀正確", out["vocals"].shape == wav.shape,
          str(tuple(out["vocals"].shape)))
    check("伴奏輸出形狀正確", out["accompaniment"].shape == wav.shape)
    check("遮罩值域在 [0,1]",
          bool((out["mask"] >= 0).all() and (out["mask"] <= 1).all()))


def test_additivity() -> None:
    """人聲 + 伴奏必須等於輸入 —— 架構上由 accompaniment = input - vocals 保證。"""
    print("可加性")
    model = build("tiny")
    wav = torch.randn(1, 2, 44100)
    out = model(wav)
    residual = (out["vocals"] + out["accompaniment"] - wav).abs().max().item()
    check("vocals + accompaniment == 輸入", residual < 1e-4,
          f"最大誤差 {residual:.2e}")


def test_stft_roundtrip() -> None:
    """STFT → iSTFT 必須無損,否則模型還沒開始學就已經有誤差。"""
    print("STFT 往返")
    model = build("tiny")
    wav = torch.randn(1, 2, 44100 * 2)
    spec = model.stft(wav)
    back = model.istft(spec, wav.shape[-1])
    error = (back - wav).abs().max().item()
    check("往返誤差可忽略", error < 1e-4, f"最大誤差 {error:.2e}")


def test_mask_semantics() -> None:
    """遮罩全 0 應該輸出靜音,全 1 應該輸出原訊號。"""
    print("遮罩語意")
    model = build("tiny")
    wav = torch.randn(1, 2, 44100)

    with torch.no_grad():
        spec = model.stft(wav)
        zeros = model.istft(spec * 0.0, wav.shape[-1])
        ones = model.istft(spec * 1.0, wav.shape[-1])

    check("遮罩=0 → 靜音", zeros.abs().max().item() < 1e-5)
    check("遮罩=1 → 原訊號", (ones - wav).abs().max().item() < 1e-4)


def test_high_band_policy() -> None:
    """超出 max_bins 的頻段應該完全歸伴奏(遮罩為 0)。"""
    print("高頻段策略")
    config = ModelConfig(max_bins=256)
    model = Separator(config)
    out = model(torch.randn(1, 2, 44100))
    high = out["mask"][:, :, 256:]
    check("高頻遮罩為 0(保守地全給伴奏)",
          bool((high == 0).all()), f"{high.shape[2]} 個 bin")


# ── 資源用量(實測)──────────────────────────────────────────────────────
def measure(preset: str, seconds: float, batch: int,
            dev: str) -> dict | None:
    """實際跑一次前向+反向,量峰值記憶體與耗時。"""
    model = build(preset).to(dev)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    samples = int(seconds * 44100)
    wav = torch.randn(batch, 2, samples, device=dev)
    target = torch.randn(batch, 2, samples, device=dev)

    if dev == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    try:
        started = time.perf_counter()
        for _ in range(3):
            optimizer.zero_grad(set_to_none=True)
            out = model(wav)
            loss = torch.nn.functional.l1_loss(out["vocals"], target)
            loss.backward()
            optimizer.step()
        if dev == "cuda":
            torch.cuda.synchronize()
        elapsed = (time.perf_counter() - started) / 3
    except torch.cuda.OutOfMemoryError:
        return None
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            return None
        raise

    peak = (torch.cuda.max_memory_allocated() / 1024**3
            if dev == "cuda" else float("nan"))
    result = {"preset": preset, "params": model.num_parameters,
              "peak_gb": peak, "step_s": elapsed,
              "seconds": seconds, "batch": batch}
    del model, optimizer, wav, target
    if dev == "cuda":
        torch.cuda.empty_cache()
    return result


def test_memory_footprint() -> None:
    print("訓練資源實測(前向 + 反向 + 更新)")
    dev = device()
    if dev == "cpu":
        print("  (沒有 CUDA,略過 VRAM 量測)")
        for preset in PRESETS:
            model = build(preset)
            print(f"  {preset:<8} 參數 {model.num_parameters/1e6:>6.2f} M")
        check("所有預設值都能建立", True)
        return

    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"  GPU: {torch.cuda.get_device_name(0)}  ({total:.1f} GB)\n")
    print(f"  {'預設':<8}{'參數':>10}{'片段':>7}{'batch':>7}"
          f"{'峰值VRAM':>11}{'每步':>9}")
    print("  " + "─" * 54)

    # Windows 的 WDDM 允許溢出到系統記憶體,所以「沒有 OOM」不代表裝得下 ——
    # 溢出時速度會掉一個數量級。留 10% 餘裕作為真正可用的判準。
    budget = total * 0.90
    fits = []
    for preset in PRESETS:
        for seconds, batch in ((6.0, 4), (6.0, 2)):
            result = measure(preset, seconds, batch, dev)
            if result is None:
                continue
            usable = result["peak_gb"] <= budget
            verdict = "可用" if usable else "溢出到系統記憶體"
            print(f"  {preset:<8}{result['params']/1e6:>9.2f}M"
                  f"{seconds:>6.0f}s{batch:>7}"
                  f"{result['peak_gb']:>10.2f}G{result['step_s']:>8.2f}s"
                  f"   {verdict}")
            if usable:
                fits.append(result)
            break
        else:
            print(f"  {preset:<8}{'—':>9} {'OOM(即使 batch=2)':>30}")

    check("有設定能真正在 VRAM 內訓練", len(fits) > 0,
          f"{len(fits)}/{len(PRESETS)} 個預設值裝得下 {budget:.1f} GB 預算")
    if fits:
        best = max(fits, key=lambda r: r["params"])
        print(f"\n  → 這張卡上最大的實用設定:{best['preset']}"
              f"({best['params']/1e6:.1f}M 參數,"
              f"{best['peak_gb']:.2f} GB,{best['step_s']:.2f} 秒/步)")
        check("最大可用設定確實留有餘裕", total - best["peak_gb"] > 0.5,
              f"剩 {total - best['peak_gb']:.2f} GB")


# ── 訓練機制健全性 ──────────────────────────────────────────────────────
def test_can_overfit() -> None:
    """能不能記住單一樣本?

    這是判斷訓練機制正確與否的標準檢查。若連一個樣本都學不會,
    表示梯度流、損失函數或架構有錯 —— 那麼餵再多資料也不會有結果。
    """
    print("過擬合測試(驗證梯度真的有流動)")
    dev = device()
    torch.manual_seed(0)

    model = build("tiny").to(dev)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)

    # 造一個結構明確的目標:人聲在中頻,伴奏在低頻
    n = 44100
    t = torch.arange(n, device=dev) / 44100.0
    vocals = 0.3 * torch.sin(2 * np.pi * 800 * t)
    accompaniment = 0.3 * torch.sin(2 * np.pi * 90 * t)
    vocals = vocals.expand(1, 2, n).contiguous()
    accompaniment = accompaniment.expand(1, 2, n).contiguous()
    mixture = vocals + accompaniment

    losses = []
    for step in range(120):
        optimizer.zero_grad(set_to_none=True)
        out = model(mixture)
        loss = torch.nn.functional.l1_loss(out["vocals"], vocals)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    first, last = np.mean(losses[:10]), np.mean(losses[-10:])
    check("損失明顯下降", last < first * 0.5,
          f"{first:.5f} → {last:.5f}(降低 {(1-last/first)*100:.0f}%)")
    check("損失沒有發散", np.isfinite(losses[-1]) and losses[-1] < first)

    # 學到的遮罩應該在人聲頻率附近較高、在伴奏頻率附近較低
    with torch.no_grad():
        mask = model(mixture)["mask"][0, 0]
    bin_hz = 44100 / 2048
    vocal_bin, bass_bin = int(800 / bin_hz), int(90 / bin_hz)
    vocal_mask = mask[vocal_bin - 2:vocal_bin + 3].mean().item()
    bass_mask = mask[bass_bin - 2:bass_bin + 3].mean().item()
    check("遮罩在人聲頻率高於伴奏頻率", vocal_mask > bass_mask,
          f"800Hz={vocal_mask:.3f} vs 90Hz={bass_mask:.3f}")


def main() -> int:
    print(f"裝置:{device()}")
    if torch.cuda.is_available():
        print(f"      {torch.cuda.get_device_name(0)}\n")
    else:
        print()

    tests = (test_shapes, test_additivity, test_stft_roundtrip,
             test_mask_semantics, test_high_band_policy,
             test_memory_footprint, test_can_overfit)
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
