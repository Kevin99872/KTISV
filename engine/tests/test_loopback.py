"""VB-CABLE 回送測試 —— 驗證「Discord 實際會聽到什麼」。

引擎只開虛擬音效卡那一路(不開耳機),把測試訊號送進 CABLE Input,
同時從 CABLE Output 錄回來比對。CABLE Output 正是 Discord 要選的輸入裝置,
所以錄到什麼,對方就聽到什麼。

因為沒有開耳機輸出,整個過程你不會聽到任何聲音。

    python -m tests.test_loopback
"""

from __future__ import annotations

import sys
import time

import numpy as np
import sounddevice as sd

from ktisv_engine import SAMPLE_RATE
from ktisv_engine.audio import devices as devices_mod
from ktisv_engine.audio.engine import AudioEngine

FAILURES: list[str] = []
SKIPPED = False


def check(name: str, condition: bool, detail: str = "") -> None:
    mark = "ok  " if condition else "FAIL"
    print(f"  [{mark}] {name}{('  — ' + detail) if detail else ''}")
    if not condition:
        FAILURES.append(name)


def find(items: list[dict], needle: str) -> dict | None:
    """在 WASAPI 裝置中找名稱相符的那一個。"""
    wasapi = [d for d in items if d["hostapi"] == "Windows WASAPI"]
    for d in wasapi or items:
        if needle.lower() in d["name"].lower():
            return d
    return None


def band_energy(stereo: np.ndarray, f0: float, width: float = 60.0) -> float:
    """左聲道在 f0 附近的能量(dB)。

    刻意只看單一聲道:去人聲之後,硬左右分離的樂器會變成 L=+side、R=-side,
    把兩聲道平均會讓它再次相消,量出來的結果就不代表對方實際聽到的東西。
    """
    mono = stereo[:, 0] if stereo.ndim > 1 else stereo
    n = len(mono)
    if n < 1024:
        return -120.0
    spec = np.abs(np.fft.rfft(mono * np.hanning(n)))
    freqs = np.fft.rfftfreq(n, 1.0 / SAMPLE_RATE)
    sel = (freqs > f0 - width) & (freqs < f0 + width)
    return 20.0 * np.log10(max(float(np.sum(spec[sel])), 1e-12))


def make_song(seconds: float) -> np.ndarray:
    """置中的 1 kHz「人聲」+ 硬左分的 300 Hz「樂器」。"""
    t = np.arange(int(seconds * SAMPLE_RATE)) / SAMPLE_RATE
    vocal = 0.30 * np.sin(2 * np.pi * 1000 * t)
    inst = 0.30 * np.sin(2 * np.pi * 300 * t)
    return np.column_stack([vocal + inst, vocal]).astype(np.float32)


def capture(device_index: int, seconds: float) -> np.ndarray:
    """從指定輸入裝置錄音,回傳 (frames, 2) 立體聲 float32。"""
    frames = int(seconds * SAMPLE_RATE)
    return sd.rec(frames, samplerate=SAMPLE_RATE, channels=2,
                  dtype="float32", device=device_index, blocking=True)


def run_case(engine: AudioEngine, cable_out: int, seconds: float = 1.6) -> np.ndarray:
    """讓引擎從頭播一次,同時錄下 CABLE Output。"""
    engine.player.seek(0.0)
    engine.player.play()
    time.sleep(0.25)          # 跳過傳輸淡入
    recorded = capture(cable_out, seconds)
    engine.player.pause()
    time.sleep(0.1)
    return recorded


def main() -> int:
    global SKIPPED

    info = devices_mod.list_devices()
    cable_in = find(info["outputs"], "cable input")
    cable_out = find(info["inputs"], "cable output")

    print("VB-CABLE 裝置")
    if cable_in is None or cable_out is None:
        print("  找不到 VB-CABLE。請先從 https://vb-audio.com/Cable/ 安裝並重開機。")
        SKIPPED = True
        return 0

    print(f"  送出 → {cable_in['name']}  (index {cable_in['index']}, "
          f"裝置預設 {cable_in['samplerate']} Hz)")
    print(f"  錄回 ← {cable_out['name']}  (index {cable_out['index']}, "
          f"裝置預設 {cable_out['samplerate']} Hz)")

    mismatched = [d for d in (cable_in, cable_out) if d["samplerate"] != SAMPLE_RATE]
    if mismatched:
        print()
        print(f"  注意:引擎跑 {SAMPLE_RATE} Hz,但這些端點是 "
              f"{', '.join(str(d['samplerate']) for d in mismatched)} Hz。")
        print("  請到 Windows「音效」設定,把 CABLE Input 與 CABLE Output 的預設格式")
        print("  都改成 48000 Hz —— 兩端不一致會造成破音或音調不對。")
    print()

    # ── Windows 預設播放裝置 ────────────────────────────────────────────
    # VB-CABLE 的安裝程式常常會把系統預設輸出換成 CABLE Input。若沒改回來,
    # 所有系統聲音都會灌進這條線路(對方會聽到你的全部音效),而且你自己
    # 什麼都聽不到。
    print("Windows 預設播放裝置")
    try:
        default_out = sd.query_devices(sd.default.device[1])["name"]
    except Exception:
        default_out = ""
    hijacked = "cable" in default_out.lower()
    check("預設播放裝置不是 CABLE Input", not hijacked,
          default_out or "(查詢失敗)")

    if hijacked:
        # 這是前置條件,不是普通的檢查項。系統預設輸出還指著 CABLE Input 時,
        # 任何系統聲音(通知、瀏覽器、其他程式)都會混進這條線路,量測窗口
        # 內出現一次就會讓結果偏掉好幾 dB。與其量出不可信的數字,不如停下來。
        print()
        print("  這個狀態下量不準,先修設定再跑:")
        print("    設定 → 系統 → 音效 → 輸出,改回你的耳機")
        print()
        print("  KTISV 走的是自己選的裝置,不受這個設定影響 —— 改回來不會影響程式運作。")
        return 1
    print()

    # ── 建立引擎:只走虛擬音效卡,不開耳機(所以完全靜音)──────────────
    engine = AudioEngine()
    engine.virtual_device = cable_in["index"]
    engine.headphone_device = None
    engine.mic_device = None

    engine.params.send_music_vc.snap(1.0)
    engine.params.master_vc.snap(1.0)
    engine.params.music_fader.snap(1.0)

    engine.player.load({"mix": make_song(6.0)}, title="loopback")

    print("開啟串流")
    try:
        engine.start()
        check("虛擬音效卡可開啟", engine.running)
    except Exception as exc:
        check("虛擬音效卡可開啟", False, str(exc))
        return 1
    print()

    try:
        # ── 靜音底線 ────────────────────────────────────────────────────
        print("靜音底線(播放器停住)")
        silence = capture(cable_out["index"], 0.8)
        noise_floor = band_energy(silence, 1000.0)
        silence_peak = float(np.max(np.abs(silence)))
        check("沒播放時線路是安靜的", silence_peak < 0.02, f"峰值 {silence_peak:.4f}")
        if silence_peak >= 0.02:
            print("         → 有別的程式正在對 CABLE Input 播音。最常見的原因就是")
            print("           Windows 預設播放裝置還停在 CABLE Input。")
        print()

        # ── 訊號真的穿過 VB-CABLE ───────────────────────────────────────
        print("訊號穿透(Discord 會聽到的東西)")
        engine.apply_separation_flags(False, False, realtime=True)
        clean = run_case(engine, cable_out["index"])

        vocal_level = band_energy(clean, 1000.0)
        inst_level = band_energy(clean, 300.0)
        peak = float(np.max(np.abs(clean)))

        check("音訊確實通過了 VB-CABLE", peak > 0.05, f"錄到峰值 {peak:.3f}")
        check("1 kHz 人聲有到達對端", vocal_level > noise_floor + 30.0,
              f"{vocal_level:.1f} dB(底噪 {noise_floor:.1f} dB)")
        check("300 Hz 伴奏有到達對端", inst_level > noise_floor + 30.0,
              f"{inst_level:.1f} dB")
        check("沒有削波", peak < 0.99, f"峰值 {peak:.3f}")
        print()

        # ── 勾選框在對端真的生效 ────────────────────────────────────────
        print("「剝離人聲」在對端生效")
        engine.apply_separation_flags(True, False, realtime=True)
        time.sleep(0.2)
        removed = run_case(engine, cable_out["index"])

        vocal_after = band_energy(removed, 1000.0)
        inst_after = band_energy(removed, 300.0)
        check("對端聽到的人聲被壓下去", vocal_after < vocal_level - 20.0,
              f"{vocal_level:.1f} → {vocal_after:.1f} dB")
        # 硬左右分離的樂器經 mid/side 相消後會少掉約一半能量(-6 dB),
        # 再加上低頻保留濾波器在 300 Hz 的殘留相位,實測約落在 -6 ~ -9 dB。
        check("對端聽到的伴奏還在", inst_after > inst_level - 10.0,
              f"{inst_level:.1f} → {inst_after:.1f} dB")
        print()

        # ── 音量推桿在對端真的生效 ──────────────────────────────────────
        print("音量推桿在對端生效")
        engine.apply_separation_flags(False, False, realtime=True)
        engine.params.master_vc.snap(10 ** (-12.0 / 20.0))   # -12 dB
        time.sleep(0.2)
        quiet = run_case(engine, cable_out["index"])

        quiet_level = band_energy(quiet, 1000.0)
        delta = vocal_level - quiet_level
        check("總輸出 -12 dB 在對端量得到", 8.0 < delta < 16.0,
              f"實測衰減 {delta:.1f} dB")
        print()

        # ── 串流健康度 ──────────────────────────────────────────────────
        print("串流健康度")
        status = engine.status()
        check("沒有 xrun", status["xruns"] == 0, f"{status['xruns']} 次")
        check("CPU 負載在安全範圍", status["cpu"] < 0.5,
              f"{status['cpu'] * 100:.1f}% of block budget")
        check("緩衝沒有大量溢位", status["ring_overflows"] < 10,
              f"{status['ring_overflows']} 次")

    finally:
        engine.stop()

    print()
    if FAILURES:
        print(f"{len(FAILURES)} 項未通過: " + ", ".join(FAILURES))
        return 1
    print("全部通過 —— Discord 選 CABLE Output 就會聽到這些內容。")
    return 0


if __name__ == "__main__":
    code = main()
    if SKIPPED:
        print("(已跳過:未安裝 VB-CABLE)")
    sys.exit(code)
