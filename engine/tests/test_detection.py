"""虛擬裝置偵測 —— 名稱比對邏輯測試。

不依賴實機硬體:用合成的裝置名稱驗證各驅動家族都認得出來,
而且實體裝置不會被誤判成虛擬。

    python -m tests.test_detection
"""

from __future__ import annotations

import sys

from ktisv_engine.audio import devices as devices_mod

FAILURES: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    mark = "ok  " if condition else "FAIL"
    print(f"  [{mark}] {name}{('  — ' + detail) if detail else ''}")
    if not condition:
        FAILURES.append(name)


def dev(name: str, hostapi: str = "Windows WASAPI") -> dict:
    return {"name": name, "hostapi": hostapi,
            "virtual": devices_mod._looks_virtual(name), "index": 0}


# 各家族在真實系統上會出現的裝置名稱
VIRTUAL_SAMPLES = {
    "VB-CABLE": [
        "CABLE Input (VB-Audio Virtual Cable)",
        "CABLE Output (VB-Audio Virtual Cable)",
    ],
    "VoiceMeeter": [
        "VoiceMeeter Input (VB-Audio VoiceMeeter VAIO)",
        "VoiceMeeter Aux Output (VB-Audio VoiceMeeter AUX VAIO)",
    ],
    "Virtual Audio Driver": [
        "Virtual Speaker (Virtual Audio Driver)",
        "Virtual Mic (Virtual Audio Driver)",
    ],
    "AudioMirror": [
        "Audio Mirror Speaker",
        "Audio Mirror Microphone",
    ],
    "Elgato Wave Link": [
        "Wave Link Stream (Elgato Wave Link)",
    ],
    "NVIDIA Broadcast": [
        "Microphone (NVIDIA Broadcast)",
    ],
    "VAC": [
        "Line 1 (Virtual Audio Cable)",
    ],
}

# 這些是實體裝置,絕不能被判成虛擬
PHYSICAL_SAMPLES = [
    "Headset Earphone on SoundWire Device (6- Realtek XU)",
    "Microphone Array on SoundWire Device (2- Realtek XU)",
    "喇叭 (2- SoundWire Speakers)",
    "Smart M70D (4- HD Audio Driver for Display Audio)",
    "Realtek High Definition Audio",
    "Scarlett 2i2 USB",              # 實體錄音介面,名字裡沒有虛擬字樣
    "Focusrite USB Audio",
    "Microsoft 音效對應表 - Output",
]


def test_virtual_recognised() -> None:
    print("虛擬裝置名稱辨識")
    for family, names in VIRTUAL_SAMPLES.items():
        for name in names:
            check(f"{family}: {name[:44]}", devices_mod._looks_virtual(name))


def test_physical_not_flagged() -> None:
    print("實體裝置不被誤判")
    for name in PHYSICAL_SAMPLES:
        check(f"不是虛擬: {name[:46]}", not devices_mod._looks_virtual(name))


def test_family_detection() -> None:
    print("家族辨識")
    for family, names in VIRTUAL_SAMPLES.items():
        found = devices_mod.detect_families([dev(n) for n in names])
        # 只能認出這一個家族 —— 標錯家族會讓引導畫面給出錯誤的指示
        check(f"認出 {family}", found == [family], str(found))

    mixed = [dev(n) for n in PHYSICAL_SAMPLES]
    check("只有實體裝置時不報任何家族",
          devices_mod.detect_families(mixed) == [],
          str(devices_mod.detect_families(mixed)))


def test_output_preference() -> None:
    print("輸出端自動選擇")

    def pick(names_with_host):
        outputs = []
        for i, (name, host) in enumerate(names_with_host):
            outputs.append({**dev(name, host), "index": i})
        return devices_mod.suggest_virtual_output(
            {"outputs": outputs, "inputs": []})

    # 有 VB-CABLE 時優先選它的輸入端(那是 KTISV 要寫入的喇叭)
    idx = pick([
        ("Virtual Speaker (Virtual Audio Driver)", "Windows WASAPI"),
        ("CABLE Input (VB-Audio Virtual Cable)", "Windows WASAPI"),
    ])
    check("VB-CABLE 優先於其他家族", idx == 1, f"選了 #{idx}")

    # 同一裝置有多個 host API 時偏好 WASAPI(延遲較低)
    idx = pick([
        ("CABLE Input (VB-Audio Virtual Cable)", "MME"),
        ("CABLE Input (VB-Audio Virtual Cable)", "Windows WASAPI"),
    ])
    check("同名裝置偏好 WASAPI", idx == 1, f"選了 #{idx}")

    # 只有開源驅動時也要選得到
    idx = pick([
        ("Virtual Speaker (Virtual Audio Driver)", "Windows WASAPI"),
    ])
    check("只有 Virtual Audio Driver 也選得到", idx == 0, f"選了 #{idx}")

    # 完全沒有虛擬裝置
    check("沒有虛擬裝置時回傳 None",
          devices_mod.suggest_virtual_output({"outputs": [], "inputs": []}) is None)


def test_live_system() -> None:
    """實機掃描 —— 只回報,不斷言(取決於使用者裝了什麼)。"""
    print("實機狀態(僅供參考)")
    info = devices_mod.list_devices()
    families = info["virtual_families"]
    if families:
        print(f"  偵測到:{'、'.join(families)}")
        print(f"  虛擬輸出:{info['has_virtual_output']}  "
              f"虛擬輸入:{info['has_virtual_input']}")
    else:
        print("  這台機器目前沒有安裝任何虛擬音訊驅動")
        print("  → 第一次啟動精靈會顯示安裝引導")
    check("list_devices 回報完整欄位",
          {"has_virtual_input", "has_virtual_output", "virtual_families"}
          <= set(info))


def main() -> int:
    for fn in (test_virtual_recognised, test_physical_not_flagged,
               test_family_detection, test_output_preference, test_live_system):
        fn()
        print()
    if FAILURES:
        print(f"{len(FAILURES)} 項未通過: " + ", ".join(FAILURES))
        return 1
    print("全部通過。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
