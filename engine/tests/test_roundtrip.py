"""真實環回延遲量測。

為什麼需要這個
--------------
``latency_report()`` 是把各段緩衝的**回報值**相加算出來的。但 PortAudio 回報的
只是它自己設定的緩衝,WASAPI 共享模式下 Windows 音訊引擎還有一層混音緩衝
不在其中 —— 所以那個數字可能低估真實延遲。

這支測試不相信任何回報值:實際播一段訊號、實際錄回來、數它晚了幾幀。

為什麼一定要用「雙工」串流
--------------------------
第一版用兩條獨立的 OutputStream / InputStream,結果變異高達 66 ms —— 因為
兩條串流的啟動時機與時脈各自獨立,量到的其實是它們的隨機偏移,不是延遲。

改用單一 ``sd.Stream``:輸入與輸出共用同一個回呼、同一個時脈。在第 N 個
回呼幀寫出訊號、在第 M 個回呼幀收到,``M - N`` 就是精確的來回延遲。

需要虛擬音效卡把輸出接回輸入(VB-CABLE 之類)::

    python -m tests.test_roundtrip
    python -m tests.test_roundtrip --blocksize 128 --exclusive
"""

from __future__ import annotations

import argparse
import sys
import threading

import numpy as np
import sounddevice as sd

from ktisv_engine.audio import devices as devices_mod

SR = 48000


def find_loopback() -> tuple[int, int] | None:
    """找一組能把輸出接回輸入的虛擬裝置(輸出端, 輸入端)。"""
    info = devices_mod.list_devices()

    def pick(items, *keywords):
        for want in keywords:
            for d in items:
                if want in d["name"].lower() and d["hostapi"] == "Windows WASAPI":
                    return d["index"]
        return None

    out = pick(info["outputs"], "cable input")
    inp = pick(info["inputs"], "cable output")
    if out is not None and inp is not None:
        return out, inp

    out = pick(info["outputs"], "voicemeeter input", "virtual speaker")
    inp = pick(info["inputs"], "voicemeeter output", "virtual mic")
    if out is not None and inp is not None:
        return out, inp
    return None


def make_probe(length: int = 256) -> np.ndarray:
    """短促的啁啾。互相關的峰值比單一脈衝銳利,也不容易被系統處理吃掉。"""
    t = np.arange(length) / SR
    sweep = np.sin(2 * np.pi * (2000 + 8000 * t / (length / SR)) * t)
    return (sweep * np.hanning(length)).astype(np.float32)


def measure_once(out_device: int, in_device: int, blocksize: int,
                 exclusive: bool, timeout: float = 6.0) -> float | None:
    """量一次來回延遲(毫秒)。失敗回傳 None。"""
    probe = make_probe()
    settings = None
    if exclusive:
        try:
            settings = sd.WasapiSettings(exclusive=True)
        except Exception:
            pass

    state = {
        "frame": 0,          # 已處理的回呼幀數
        "emit_at": None,     # 送出訊號的幀位置
        "captured": [],      # 收錄的音訊
        "emitted": False,
    }
    done = threading.Event()

    # 先讓串流跑一段暖機再送訊號 —— 剛開串流時緩衝還在填,不具代表性
    warmup_frames = int(SR * 0.3)
    capture_frames = int(SR * 0.8)

    def callback(indata, outdata, frames, time_info, status):
        position = state["frame"]
        outdata.fill(0)

        # 暖機結束後,在這個 block 的開頭送出探測訊號
        if not state["emitted"] and position >= warmup_frames:
            count = min(len(probe), frames)
            outdata[:count, 0] = probe[:count]
            if outdata.shape[1] > 1:
                outdata[:count, 1] = probe[:count]
            state["emit_at"] = position
            state["emitted"] = True

        if state["emitted"]:
            state["captured"].append(indata[:, 0].copy())
            if sum(len(c) for c in state["captured"]) >= capture_frames:
                done.set()

        state["frame"] = position + frames

    common = dict(samplerate=SR, dtype="float32", blocksize=blocksize,
                  latency="low")
    if settings is not None:
        # 雙工串流的 extra_settings 要給 (輸入, 輸出) 兩份
        common["extra_settings"] = (settings, settings)

    try:
        with sd.Stream(device=(in_device, out_device),
                       channels=(1, 2), callback=callback, **common):
            done.wait(timeout)
    except Exception:
        return None

    if not state["captured"] or state["emit_at"] is None:
        return None

    signal = np.concatenate(state["captured"]).astype(np.float64)
    if float(np.max(np.abs(signal))) < 1e-4:
        return None

    correlation = np.correlate(signal, probe.astype(np.float64), mode="valid")
    peak = int(np.argmax(np.abs(correlation)))

    # captured 是從送出訊號的那個 block 開始收的,所以 peak 直接就是延遲幀數
    return peak / SR * 1000.0


def measure(out_device: int, in_device: int, blocksize: int,
            exclusive: bool, repeats: int = 5) -> dict:
    results = [measure_once(out_device, in_device, blocksize, exclusive)
               for _ in range(repeats)]
    values = [v for v in results if v is not None]
    if not values:
        return {"ok": False, "error": "錄不到訊號"}
    return {
        "ok": True,
        "median_ms": float(np.median(values)),
        "min_ms": float(np.min(values)),
        "max_ms": float(np.max(values)),
        "spread_ms": float(np.max(values) - np.min(values)),
        "samples": len(values),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="真實環回延遲量測")
    parser.add_argument("--blocksize", type=int, default=None)
    parser.add_argument("--exclusive", action="store_true")
    parser.add_argument("--shared-only", action="store_true")
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args(argv)

    pair = find_loopback()
    if pair is None:
        print("找不到可用的環回裝置。需要 VB-CABLE 之類的虛擬音效卡。")
        return 1

    out_device, in_device = pair
    print(f"輸出 → {devices_mod.describe(out_device)}")
    print(f"輸入 ← {devices_mod.describe(in_device)}")
    print()

    blocksizes = ([args.blocksize] if args.blocksize
                  else [64, 128, 192, 240, 320, 480, 960])
    if args.exclusive:
        modes = [True]
    elif args.shared_only:
        modes = [False]
    else:
        modes = [False, True]

    print(f"{'模式':<6}{'block':>7}{'中位數':>10}{'最小':>9}{'最大':>9}{'變異':>8}")
    print("-" * 50)

    best = None
    for exclusive in modes:
        for blocksize in blocksizes:
            result = measure(out_device, in_device, blocksize, exclusive,
                             args.repeats)
            label = "獨佔" if exclusive else "共享"
            if not result["ok"]:
                print(f"{label:<6}{blocksize:>7}   {result['error'][:36]}")
                continue
            print(f"{label:<6}{blocksize:>7}{result['median_ms']:>9.1f}ms"
                  f"{result['min_ms']:>8.1f}ms{result['max_ms']:>8.1f}ms"
                  f"{result['spread_ms']:>7.1f}ms")
            if best is None or result["median_ms"] < best[0]:
                best = (result["median_ms"], label, blocksize)

    if best:
        print()
        print(f"最低來回延遲:{best[1]} · block {best[2]} → {best[0]:.1f} ms")

    print()
    print("這是完整的軟體來回延遲(輸出→虛擬線→輸入),含 OS 混音層。")
    print("「變異」欄若很大表示量測不穩,數字不可盡信。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
