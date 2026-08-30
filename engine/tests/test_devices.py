"""實體裝置測試 —— 真的開啟 WASAPI 串流,驗證雙路輸出與麥克風輸入。

不載入任何音源,所以輸出全程是靜音,不會發出聲音。

    python -m tests.test_devices                  (自動挑裝置)
    python -m tests.test_devices --hp 14 --vc 16 --mic 18
"""

from __future__ import annotations

import argparse
import sys
import time

import sounddevice as sd

from ktisv_engine.audio import devices as devices_mod
from ktisv_engine.audio.engine import AudioEngine

FAILURES: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    mark = "ok  " if condition else "FAIL"
    print(f"  [{mark}] {name}{('  — ' + detail) if detail else ''}")
    if not condition:
        FAILURES.append(name)


def _pool(items: list[dict]) -> list[dict]:
    wasapi = [d for d in items if d["hostapi"] == "Windows WASAPI"]
    return wasapi or items


def pick_outputs(info: dict) -> tuple[int | None, int | None]:
    """挑「耳機 + 虛擬音效卡」兩路。

    耳機一定要挑實體裝置:虛擬音效卡的某些端點(例如 CABLE In 16ch)不接受
    2 聲道開啟,拿它當耳機只會測到該裝置本身的限制,不是引擎的問題。
    """
    pool = _pool(info["outputs"])
    if not pool:
        return None, None

    physical = [d["index"] for d in pool if not d["virtual"]]
    virtual = devices_mod.suggest_virtual_output(info)

    headphone = physical[0] if physical else pool[0]["index"]
    if virtual is None:
        # 沒有虛擬音效卡時,改用第二個實體輸出來驗證雙路架構
        virtual = next((i for i in physical if i != headphone), None)
    return headphone, virtual


def pick_input(info: dict) -> int | None:
    """挑實體麥克風 —— 選到 CABLE Output 會和 CABLE Input 構成回授迴圈。"""
    pool = _pool(info["inputs"])
    physical = [d for d in pool if not d["virtual"]]
    chosen = physical or pool
    return chosen[0]["index"] if chosen else None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hp", type=int, default=None)
    parser.add_argument("--vc", type=int, default=None)
    parser.add_argument("--mic", type=int, default=None)
    parser.add_argument("--seconds", type=float, default=3.0)
    args = parser.parse_args(argv)

    info = devices_mod.list_devices()
    auto_hp, auto_vc = pick_outputs(info)
    headphone = args.hp if args.hp is not None else auto_hp
    virtual = args.vc if args.vc is not None else auto_vc
    mic = args.mic if args.mic is not None else pick_input(info)

    print("選用的裝置")
    print(f"  耳機     : {devices_mod.describe(headphone) if headphone is not None else '—'}")
    print(f"  虛擬音效卡: {devices_mod.describe(virtual) if virtual is not None else '—'}")
    print(f"  麥克風   : {devices_mod.describe(mic) if mic is not None else '—'}")
    if not info["has_virtual_output"]:
        print("  (未安裝虛擬音效卡,改用第二個實體輸出來驗證雙路架構)")
    print()

    # ── 單路輸出 ────────────────────────────────────────────────────────
    print("單路輸出(只有耳機)")
    engine = AudioEngine()
    engine.headphone_device = headphone
    try:
        engine.start()
        check("串流可開啟", engine.running)
        time.sleep(1.0)
        status = engine.status()
        check("回呼有在跑(CPU 負載 > 0)", status["cpu"] > 0, f"{status['cpu'] * 100:.1f}%")
        check("沒有 xrun", status["xruns"] == 0, f"{status['xruns']} 次")
    except Exception as exc:
        check("串流可開啟", False, str(exc))
    finally:
        engine.stop()
    check("可以停止", not engine.running)
    print()

    # ── 雙路輸出 + 麥克風 ───────────────────────────────────────────────
    print(f"雙路輸出 + 麥克風({args.seconds:.0f} 秒)")
    engine = AudioEngine()
    engine.headphone_device = headphone
    engine.virtual_device = virtual
    engine.mic_device = mic
    engine.params.monitor_self = True
    engine.params.mic_fader.snap(1.0)

    started = False
    try:
        engine.start()
        started = engine.running
        check("三條串流同時開啟", started)
    except Exception as exc:
        check("三條串流同時開啟", False, str(exc))

    if started:
        time.sleep(args.seconds)
        status = engine.status()

        check("主時脈回呼有在跑", status["cpu"] > 0, f"CPU {status['cpu'] * 100:.1f}%")
        check("CPU 負載在安全範圍", status["cpu"] < 0.5,
              f"{status['cpu'] * 100:.1f}% of block budget")
        check("沒有 xrun", status["xruns"] == 0, f"{status['xruns']} 次")

        if mic is not None:
            mic_level = engine.meters["mic_in"].peak_db
            check("麥克風有取得資料", engine._mic_ring.underflows < 5,
                  f"underflow {engine._mic_ring.underflows} 次,輸入峰值 {mic_level:.1f} dBFS")

        # 虛擬音效卡那一路靠環形緩衝跨時脈,檢查它沒有失控
        vc_fill = status["vc_ring"]
        check("虛擬音效卡緩衝穩定", 0 <= vc_fill <= engine.blocksize * 8,
              f"{vc_fill} frames(約 {vc_fill / 48000 * 1000:.1f} ms)")
        check("緩衝沒有大量溢位", status["ring_overflows"] < 10,
              f"{status['ring_overflows']} 次")

        engine.stop()
        check("可以停止", not engine.running)
    print()

    # ── 錯誤處理 ────────────────────────────────────────────────────────
    print("錯誤處理")
    engine = AudioEngine()
    try:
        engine.start()
        check("沒選裝置時會拒絕啟動", False)
    except Exception as exc:
        check("沒選裝置時會拒絕啟動", "至少要選一個" in str(exc), str(exc)[:50])

    engine = AudioEngine()
    engine.headphone_device = 9999
    try:
        engine.start()
        check("無效裝置索引會回報錯誤", False)
        engine.stop()
    except Exception as exc:
        check("無效裝置索引會回報錯誤", True, str(exc)[:60])
    check("失敗後沒有殘留串流", not engine.running)
    print()

    if FAILURES:
        print(f"{len(FAILURES)} 項未通過: " + ", ".join(FAILURES))
        return 1
    print("全部通過。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
