"""音訊裝置列舉與虛擬音效卡偵測。"""

from __future__ import annotations

import sounddevice as sd

PREFERRED_HOSTAPIS = ("Windows WASAPI", "Windows DirectSound", "MME",
                      "Core Audio", "ALSA", "PulseAudio")

# 只列 WASAPI 的裝置 —— 目標是讓清單長得跟 Windows 的音效設定一樣簡潔。
#
# PortAudio 會把同一個端點在**每個** host API 底下各列一次。這台機器上
# 17 個「輸入裝置」其實只有 3 支真的麥克風,其餘全是同一支的重複項:
#
#   MME              走 legacy waveOut,系統必定多插一層重取樣,延遲高、
#                    音質隨端點設定飄,而且裝置名稱被截到 31 個字元
#                    (「Microphone Array on SoundWire D」)
#   DirectSound      同樣是相容層,沒有任何優勢
#   WDM-KS           直接接驅動的 KS pin,繞過整個 Windows 音訊引擎。
#                    它列出來的多半不是使用者認得的「麥克風」而是驅動內部
#                    的 pin —— 實測 8 個 WDM-KS 輸入裡有 Headset Earphone
#                    與 Speaker 被當成「輸入」列出來(那是監聽 pin),
#                    選了只會錄到無聲。而且 KS 是獨佔的,開了別的程式就搶不到。
#
# WASAPI 是 Windows 現在的原生音訊 API,音效控制台看到的每一個端點都在
# 這裡,所以濾掉其他三個不會讓任何真實裝置消失,只會讓重複項消失。
#
# 使用者從清單裡選到哪一份純屬運氣,選錯就得回控制台調參數 —— 這正是
# 「每次都要重調」的來源。直接不給選,問題就不存在。
PRIMARY_HOSTAPI = "Windows WASAPI"

VIRTUAL_HINTS = (
    # VB-Audio 家族
    "cable input", "cable output", "vb-audio", "vb audio", "voicemeeter",
    "virtual cable", "virtual audio cable", "vac ",
    # VirtualDrivers/Virtual-Audio-Driver(開源,MIT)
    "virtual audio", "virtual speaker", "virtual mic",
    # JannesP/AudioMirror
    "audio mirror", "audiomirror",
    # 其他常見虛擬音訊裝置
    "blackhole", "soundflower", "loopback", "obs virtual",
    "wave link", "nvidia broadcast", "steam streaming",
)

# 已知驅動家族:顯示名稱 → 可辨識的名稱關鍵字。
# 給引導畫面顯示「偵測到 X」,也用來判斷該家族是否兩端俱全。
KNOWN_FAMILIES = (
    ("VB-CABLE", ("cable input", "cable output")),
    ("VoiceMeeter", ("voicemeeter",)),
    # 注意:這裡不能用寬鬆的 "virtual audio",否則 VAC 的
    # "Line 1 (Virtual Audio Cable)" 會被誤標成這個家族。
    ("Virtual Audio Driver", ("virtual speaker", "virtual mic",
                              "virtual audio driver")),
    ("AudioMirror", ("audio mirror", "audiomirror")),
    ("Elgato Wave Link", ("wave link",)),
    ("NVIDIA Broadcast", ("nvidia broadcast",)),
    ("VAC", ("vac ", "virtual audio cable")),
)

# KTISV 要寫入的那一端(虛擬「喇叭」)的名稱關鍵字,依偏好排序。
_OUTPUT_PRIORITY = (
    "cable input",            # VB-CABLE
    "voicemeeter input",
    "voicemeeter aux input",
    "virtual speaker",        # Virtual-Audio-Driver
    "virtual audio",
    "audio mirror",           # AudioMirror
    "wave link",              # Elgato
    "nvidia broadcast",
)


def _hostapi_name(index: int) -> str:
    try:
        return sd.query_hostapis(index)["name"]
    except Exception:
        return "?"


def refresh() -> None:
    """讓 PortAudio 重新掃描裝置(插拔耳機後呼叫)。"""
    try:
        sd._terminate()
        sd._initialize()
    except Exception:
        pass


def list_devices() -> dict:
    """回傳所有輸入 / 輸出裝置,附上 host API 與虛擬音效卡標記。"""
    inputs: list[dict] = []
    outputs: list[dict] = []

    try:
        default_in, default_out = sd.default.device
    except Exception:
        default_in = default_out = -1

    for index, dev in enumerate(sd.query_devices()):
        host = _hostapi_name(dev["hostapi"])
        entry = {
            "index": index,
            "name": dev["name"],
            "hostapi": host,
            "samplerate": int(dev["default_samplerate"]),
            "virtual": _looks_virtual(dev["name"]),
            # 清單濾到只剩 WASAPI 之後,標上 host API 是純粹的雜訊 ——
            # 每一列都印同一個字。跟 Windows 的音效設定一樣,只顯示名字。
            "label": dev["name"],
        }
        if dev["max_input_channels"] > 0:
            inputs.append({**entry, "channels": dev["max_input_channels"],
                           "default": index == default_in})
        if dev["max_output_channels"] > 0:
            outputs.append({**entry, "channels": dev["max_output_channels"],
                            "default": index == default_out})

    families = detect_families(inputs + outputs)
    # 家族偵測在過濾之前做:被濾掉的重複項目不影響「裝了哪些驅動」的判斷。
    inputs = _primary_only(inputs)
    outputs = _primary_only(outputs)
    return {
        "inputs": _sorted(inputs),
        "outputs": _sorted(outputs),
        "hostapis": [h["name"] for h in sd.query_hostapis()],
        # 輸出端 = KTISV 要寫入的虛擬喇叭
        "has_virtual_output": any(d["virtual"] for d in outputs),
        # 輸入端 = Discord 要選的虛擬麥克風。兩者都有才算真的可用,
        # 所以引導畫面看的是這個。
        "has_virtual_input": any(d["virtual"] for d in inputs),
        "virtual_families": families,
    }


def detect_families(devices: list[dict]) -> list[str]:
    """從裝置名稱認出已安裝了哪些虛擬音訊驅動家族。"""
    lowered = [d["name"].lower() for d in devices]
    found: list[str] = []
    for label, hints in KNOWN_FAMILIES:
        if any(hint in name for name in lowered for hint in hints):
            found.append(label)
    return found


def _sorted(items: list[dict]) -> list[dict]:
    def key(item: dict):
        try:
            rank = PREFERRED_HOSTAPIS.index(item["hostapi"])
        except ValueError:
            rank = len(PREFERRED_HOSTAPIS)
        return (rank, item["name"].lower())

    return sorted(items, key=key)


def _looks_virtual(name: str) -> bool:
    lowered = name.lower()
    return any(hint in lowered for hint in VIRTUAL_HINTS)


def _primary_only(items: list[dict]) -> list[dict]:
    """只保留 PRIMARY_HOSTAPI 的裝置。見該常數上方的說明。

    保險:該方向若一個 WASAPI 裝置都沒有就原樣退回。非 Windows 平台
    (Core Audio / ALSA)本來就不會有 "Windows WASAPI",不能把清單清空。
    """
    kept = [d for d in items if d["hostapi"] == PRIMARY_HOSTAPI]
    return kept if kept else items


def suggest_virtual_output(devices: dict | None = None) -> int | None:
    """猜一個最可能是「送進 Discord 的虛擬喇叭」的輸出裝置。

    以 _OUTPUT_PRIORITY 的順序比對,讓不同驅動家族都能被自動選中;
    WASAPI 的同名裝置優先(延遲較低)。
    """
    devices = devices or list_devices()
    candidates = [d for d in devices["outputs"] if d["virtual"]]
    if not candidates:
        return None

    def rank(item: dict) -> tuple[int, int]:
        name = item["name"].lower()
        for i, wanted in enumerate(_OUTPUT_PRIORITY):
            if wanted in name:
                return (i, 0 if item["hostapi"] == "Windows WASAPI" else 1)
        return (len(_OUTPUT_PRIORITY),
                0 if item["hostapi"] == "Windows WASAPI" else 1)

    return min(candidates, key=rank)["index"]


def describe(index: int) -> str:
    try:
        dev = sd.query_devices(index)
        return f"{dev['name']} ({_hostapi_name(dev['hostapi'])})"
    except Exception:
        return f"#{index}"
