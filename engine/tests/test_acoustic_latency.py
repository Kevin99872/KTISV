"""耳返路徑的真實延遲量測(聲學環回)。

為什麼不能用 VB-CABLE 量
------------------------
``test_roundtrip.py`` 走的是「輸出 → VB-CABLE → 輸入」,那條路含了一段虛擬
驅動的緩衝,而**耳返根本不經過 VB-CABLE**:

    耳返實際路徑:實體麥克風 → KTISV → 實體耳機

所以那個數字對耳返而言是高估的。要量真的,只能讓麥克風**實際聽到**耳機。

怎麼跑
------
把耳機的耳罩靠近麥克風(頭戴式耳麥通常本來就夠近,戴著就行),然後::

    python -m tests.test_acoustic_latency

會從耳機播出短促的啁啾,由麥克風收回來,用互相關算出延遲。
訊號音量刻意壓低,但仍建議先把耳機音量調小再跑。

量到的是什麼
------------
完整的軟體來回延遲,加上約 1 ms 的空氣傳播(30 公分)。空氣那段可以忽略,
所以結果就是**你實際會感受到的耳返延遲**。
"""

from __future__ import annotations

import argparse
import sys
import threading

import numpy as np
import sounddevice as sd

from ktisv_engine.audio import devices as devices_mod

SR = 48000


def make_probe(length: int = 384) -> np.ndarray:
    """短促啁啾。掃頻範圍避開太低的頻率 —— 小型耳機放不出來。"""
    t = np.arange(length) / SR
    sweep = np.sin(2 * np.pi * (1500 + 6000 * t / (length / SR)) * t)
    return (sweep * np.hanning(length)).astype(np.float32)


def measure_once(in_device: int, out_device: int, blocksize: int,
                 exclusive: bool, level: float = 0.25,
                 timeout: float = 6.0) -> float | None:
    """量一次。回傳毫秒,失敗回傳 None。"""
    probe = make_probe()
    settings = None
    if exclusive:
        try:
            settings = sd.WasapiSettings(exclusive=True)
        except Exception:
            pass

    state = {"frame": 0, "cap": [], "sent": False}
    done = threading.Event()
    warmup = int(SR * 0.4)
    capture = int(SR * 0.6)

    def callback(indata, outdata, frames, time_info, status):
        position = state["frame"]
        outdata.fill(0)

        if not state["sent"] and position >= warmup:
            count = min(len(probe), frames)
            for channel in range(outdata.shape[1]):
                outdata[:count, channel] = probe[:count] * level
            state["sent"] = True

        if state["sent"]:
            state["cap"].append(indata[:, 0].copy())
            if sum(len(c) for c in state["cap"]) >= capture:
                done.set()

        state["frame"] = position + frames

    common = dict(samplerate=SR, dtype="float32", blocksize=blocksize,
                  latency="low")
    if settings is not None:
        common["extra_settings"] = (settings, settings)

    try:
        with sd.Stream(device=(in_device, out_device),
                       channels=(1, 2), callback=callback, **common):
            done.wait(timeout)
    except Exception:
        return None

    if not state["cap"]:
        return None

    signal = np.concatenate(state["cap"]).astype(np.float64)
    peak_level = float(np.max(np.abs(signal)))
    if peak_level < 2e-3:          # 麥克風沒收到 —— 太遠或音量太小
        return None

    correlation = np.correlate(signal, probe.astype(np.float64), mode="valid")
    return int(np.argmax(np.abs(correlation))) / SR * 1000.0


def measure(in_device: int, out_device: int, blocksize: int,
            exclusive: bool, repeats: int, level: float) -> dict:
    values = [v for v in
              (measure_once(in_device, out_device, blocksize, exclusive, level)
               for _ in range(repeats))
              if v is not None]
    if not values:
        return {"ok": False}
    return {
        "ok": True,
        "median": float(np.median(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "spread": float(np.max(values) - np.min(values)),
        "n": len(values),
    }


def pick_physical(items, prefer: str = "") -> int | None:
    """挑一個實體的 WASAPI 裝置(排除虛擬音效卡)。"""
    pool = [d for d in items
            if d["hostapi"] == "Windows WASAPI" and not d["virtual"]]
    if prefer:
        for d in pool:
            if prefer.lower() in d["name"].lower():
                return d["index"]
    return pool[0]["index"] if pool else None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="耳返路徑的聲學延遲量測")
    parser.add_argument("--mic", type=int, default=None)
    parser.add_argument("--out", type=int, default=None)
    parser.add_argument("--blocksize", type=int, default=None)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--level", type=float, default=0.25,
                        help="測試訊號音量 0~1,收不到訊號時可調高")
    args = parser.parse_args(argv)

    info = devices_mod.list_devices()
    mic = args.mic if args.mic is not None else pick_physical(info["inputs"], "headset")
    out = args.out if args.out is not None else pick_physical(info["outputs"], "headset")

    if mic is None or out is None:
        print("找不到合適的實體麥克風或耳機。用 --mic / --out 指定裝置編號。")
        return 1

    print(f"耳機 → {devices_mod.describe(out)}")
    print(f"麥克風 ← {devices_mod.describe(mic)}")
    print()
    print("請把耳機的耳罩靠近麥克風(戴著頭戴式耳麥通常就夠)。")
    print("測試訊號會從耳機播出,音量已壓低,但建議先把耳機音量調小。")
    print()

    blocksizes = [args.blocksize] if args.blocksize else [64, 128, 240, 480]

    print(f"{'模式':<6}{'block':>7}{'中位數':>10}{'最小':>9}{'最大':>9}"
          f"{'變異':>8}{'成功':>6}")
    print("-" * 56)

    any_ok = False
    results = []
    for exclusive in (False, True):
        for blocksize in blocksizes:
            result = measure(mic, out, blocksize, exclusive,
                             args.repeats, args.level)
            label = "獨佔" if exclusive else "共享"
            if not result["ok"]:
                print(f"{label:<6}{blocksize:>7}   收不到訊號"
                      f"{'':>26}0/{args.repeats}")
                continue

            # 只成功一兩次時「變異 0」不代表精確,只代表樣本不足 ——
            # 明確標出來,免得把不可靠的數字當成結論。
            note = "" if result["n"] >= 3 else "  ⚠ 樣本不足"
            any_ok = True
            if result["n"] >= 3:
                results.append((result["median"], label, blocksize))
            print(f"{label:<6}{blocksize:>7}{result['median']:>9.1f}ms"
                  f"{result['min']:>8.1f}ms{result['max']:>8.1f}ms"
                  f"{result['spread']:>7.1f}ms{result['n']:>4}/{args.repeats}"
                  f"{note}")

    print()
    if not results:
        print("沒有任何設定取得足夠的樣本,結果不可採信。")
        print()
        print("最可能的原因是**麥克風的回音消除(AEC)**:")
        print("  Windows 與音效驅動的「音訊增強」會主動消除從喇叭傳來的聲音,")
        print("  那正是這個測試依賴的訊號 —— 等於測試被系統反制了。")
        print()
        print("請到「設定 → 系統 → 音效 → 更多音效設定 → 錄製 → 你的麥克風")
        print("  → 內容 → 進階/增強」把所有音訊增強關掉,再跑一次。")
        print()
        print("順帶一提:那些增強本身也會增加延遲,關掉對耳返有實質幫助。")
        print()
        print("其他可能:耳機離麥克風太遠、耳機音量太小、或訊號太弱(試 --level 0.5)。")
        return 1

    best = min(results)
    print(f"最低耳返延遲:{best[1]} · block {best[2]} → {best[0]:.1f} ms")
    print()
    print("這是實際的耳返路徑(麥克風 → 耳機),含約 1 ms 的空氣傳播。")
    print("和 test_roundtrip 的差別:那個走 VB-CABLE,含耳返不會經過的虛擬驅動緩衝。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
