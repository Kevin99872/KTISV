"""耳返延遲量測 —— 自己的聲音繞一圈回到耳朵要多久。

耳返能不能用,唯一的指標就是延遲。人對自己的聲音特別敏感:
  < 10 ms   察覺不到
  10–20 ms  可以接受,唱歌沒問題
  20–30 ms  明顯,講話會有點怪
  > 30 ms   會干擾發聲,基本上不能用

全程不載入音源,所以不會發出聲音。

    python -m tests.test_latency
"""

from __future__ import annotations

import sys
import time

from ktisv_engine.audio import devices as devices_mod
from ktisv_engine.audio.engine import AudioEngine

FAILURES: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    mark = "ok  " if condition else "FAIL"
    print(f"  [{mark}] {name}{('  — ' + detail) if detail else ''}")
    if not condition:
        FAILURES.append(name)


def verdict(ms: float) -> str:
    if ms < 10:
        return "察覺不到"
    if ms < 20:
        return "可以接受,唱歌沒問題"
    if ms < 30:
        return "明顯,講話會有點怪"
    return "會干擾發聲,不建議使用"


def physical(items: list[dict]) -> list[dict]:
    wasapi = [d for d in items if d["hostapi"] == "Windows WASAPI"]
    pool = wasapi or items
    return [d for d in pool if not d["virtual"]] or pool


def measure(headphone: int, mic: int, virtual: int | None,
            blocksize: int, label: str, exclusive: bool = False) -> dict | None:
    engine = AudioEngine(blocksize=blocksize)
    engine.headphone_device = headphone
    engine.mic_device = mic
    engine.virtual_device = virtual
    engine.exclusive_mode = exclusive
    engine.params.monitor_self = True

    try:
        engine.start()
    except Exception as exc:
        print(f"  {label}: 無法開啟 —— {exc}")
        return None

    try:
        # 自適應修剪每秒收斂一次,多等幾輪讓它穩定下來
        time.sleep(4.0)
        report = engine.latency_report()
        report["xruns"] = engine.xruns
        report["cpu"] = engine.status()["cpu"]
        report["exclusive"] = list(engine.exclusive_active)
        report["notes"] = list(engine.exclusive_notes)
        return report
    finally:
        engine.stop()


def main() -> int:
    info = devices_mod.list_devices()
    outputs = physical(info["outputs"])
    inputs = physical(info["inputs"])

    if not outputs or not inputs:
        print("找不到可用的實體輸出或輸入裝置。")
        return 1

    headphone = outputs[0]["index"]
    mic = inputs[0]["index"]
    virtual = devices_mod.suggest_virtual_output(info)

    print("裝置")
    print(f"  耳機     : {devices_mod.describe(headphone)}")
    print(f"  麥克風   : {devices_mod.describe(mic)}")
    print(f"  虛擬音效卡: {devices_mod.describe(virtual) if virtual is not None else '(無)'}")
    print()

    # ── 預設 block 大小 ─────────────────────────────────────────────────
    print("耳返延遲(預設 480 frames / 10 ms block)")
    report = measure(headphone, mic, virtual, 480, "預設")
    if report is None:
        return 1

    print(f"  麥克風輸入緩衝  {report['mic_in_ms']:>7.2f} ms")
    print(f"  跨時脈環形緩衝  {report['mic_buffer_ms']:>7.2f} ms")
    print(f"  耳機輸出緩衝    {report['hp_out_ms']:>7.2f} ms")
    print(f"  ─────────────────────────")
    monitor = report["monitor_ms"]
    print(f"  耳返總延遲      {monitor:>7.2f} ms   ← {verdict(monitor)}")
    if report["mic_to_peer_ms"] is not None:
        print(f"  送到對方(不含網路) {report['mic_to_peer_ms']:>4.2f} ms")
    print()

    # 驅動回報的輸入/輸出緩衝是硬體與 WASAPI 模式決定的,程式改不了;
    # 這裡只斷言「本程式自己加上去的延遲」有壓到最低,總延遲則據實回報。
    block_ms = report["block_ms"]
    # 存量在 1~2 個 block 之間來回,平均落在 1.5 個 block 附近
    check("環形緩衝只留一個 block 左右的餘裕",
          report["mic_buffer_ms"] <= block_ms * 2.0 + 0.1,
          f"平均 {report['mic_buffer_ms']:.2f} ms(block = {block_ms:.1f} ms)")
    # DSP 唯一會加延遲的是限幅器的前瞻(一個 block)。分離、EQ、對時、
    # 降噪都是逐 block 或 IIR,零延遲。這裡斷言「多出來的剛好就是前瞻,
    # 沒有別的東西偷偷加延遲」—— 而且它必須有被算進 monitor_ms,
    # 否則介面顯示的耳返會比實際低。
    limiter_ms = report.get("limiter_ms", 0.0)
    accounted = (report["mic_in_ms"] + report["mic_buffer_ms"]
                 + report["hp_out_ms"] + limiter_ms)
    check("DSP 只有限幅器前瞻這一項延遲",
          abs(monitor - accounted) < 0.01,
          f"前瞻 {limiter_ms:.2f} ms,其餘全部零延遲")
    check("前瞻不超過一個 block",
          limiter_ms <= block_ms + 0.01,
          f"{limiter_ms:.2f} ms vs block {block_ms:.2f} ms")
    check("沒有 xrun", report["xruns"] == 0, f"{report['xruns']} 次")

    driver_ms = report["mic_in_ms"] + report["hp_out_ms"]
    print(f"  (其中 {driver_ms:.1f} ms 來自驅動的輸入/輸出緩衝,"
          f"本程式只佔 {report['mic_buffer_ms']:.1f} ms)")
    print()

    # ── 比較不同 block 大小 ─────────────────────────────────────────────
    print("不同 block 大小的取捨")
    print(f"  {'block':>6}  {'耳返':>9}  {'CPU':>7}  {'xrun':>5}")
    results = []
    for blocksize in (128, 240, 480, 960):
        r = measure(headphone, mic, virtual, blocksize, f"{blocksize}")
        if r is None or r["monitor_ms"] is None:
            continue
        results.append((blocksize, r))
        block_ms = blocksize / 48.0
        print(f"  {blocksize:>4} ({block_ms:>4.1f}ms) {r['monitor_ms']:>7.2f} ms  "
              f"{r['cpu'] * 100:>5.1f}%  {r['xruns']:>5}")

    best_shared = None
    if results:
        clean = [(b, r) for b, r in results if r["xruns"] == 0]
        if clean:
            best_shared = min(clean, key=lambda x: x[1]["monitor_ms"])
            print()
            print(f"  → 共享模式下不掉幀的最低延遲:block {best_shared[0]} "
                  f"({best_shared[1]['monitor_ms']:.2f} ms,"
                  f"CPU {best_shared[1]['cpu'] * 100:.1f}%)")
    print()

    # ── WASAPI 獨佔模式 ─────────────────────────────────────────────────
    print("WASAPI 獨佔模式(裝置會被本程式獨佔)")
    blocksize = best_shared[0] if best_shared else 480
    exclusive = measure(headphone, mic, virtual, blocksize, "獨佔", exclusive=True)
    if exclusive is None:
        print("  無法開啟。")
    else:
        if exclusive["exclusive"]:
            print(f"  獨佔生效:{'、'.join(exclusive['exclusive'])}")
        else:
            print("  沒有任何裝置能進獨佔模式,全部退回共享模式。")
        for note in exclusive["notes"]:
            print(f"  · {note}")

        print(f"  耳返總延遲      {exclusive['monitor_ms']:>7.2f} ms   "
              f"← {verdict(exclusive['monitor_ms'])}")
        if best_shared:
            saved = best_shared[1]["monitor_ms"] - exclusive["monitor_ms"]
            print(f"  比共享模式最佳值{'省下' if saved > 0 else '多出'} {abs(saved):.2f} ms")
        check("獨佔模式沒有造成 xrun", exclusive["xruns"] == 0,
              f"{exclusive['xruns']} 次")
        check("獨佔模式不會比共享模式差", best_shared is None
              or exclusive["monitor_ms"] <= best_shared[1]["monitor_ms"] + 0.1,
              f"{exclusive['monitor_ms']:.2f} ms")
    print()

    if FAILURES:
        print(f"{len(FAILURES)} 項未通過: " + ", ".join(FAILURES))
        return 1
    print("全部通過。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
