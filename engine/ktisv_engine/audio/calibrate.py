"""音遊式的對時校準。

做法
----
耳機放出「1 2 3 4」的語音節拍,你跟著念。引擎記下每一拍**送出**的取樣位置,
再從麥克風收到的訊號裡找出你每一次出聲的位置,兩者相減就是你的歌聲實際
落後(或超前)節拍多少 —— 那正是送給 Discord 那一路要補的對時量。

為什麼這比用耳朵調準
--------------------
用耳朵調要嘛靠對方回報、要嘛靠監聽送出訊號慢慢逼近,兩種都在猜。這裡量到
的是同一件事的實際數字:你聽到拍子、開口、聲音被收進混音,整個迴圈花了多久。
中間包含耳機輸出延遲、你自己的反應、麥克風輸入延遲 —— 這些本來就該一起補掉,
不需要分開估。

量到的值可能是負的
------------------
不少人習慣搶拍,開口早於聽到的節拍。那時量出來就是負值,代表要延後的是歌聲
而不是音樂。這是正常結果,不是量錯。
"""

from __future__ import annotations

import os
import subprocess
import tempfile

import numpy as np

from ..paths import cache_dir

# 節拍間隔。600 ms(100 BPM)是念得完又不拖沓的速度 —— 太快會來不及發完一個
# 音節,太慢則每一拍的起點容易被自己拉長的尾音蓋掉。
BEAT_MS = 600.0

# 總拍數,以及前面要丟掉幾拍。前兩拍通常還在抓節奏,納入只會拉大變異。
BEATS = 8
WARMUP_BEATS = 2

_CREATE_NO_WINDOW = 0x08000000 if os.name == "nt" else 0


# ── 節拍語音 ────────────────────────────────────────────────────────────
def _trim_silence(clip: np.ndarray, floor: float = 0.02) -> np.ndarray:
    """切掉前後的靜音。

    語音合成出來的檔案前面常有幾十毫秒的空白。不切的話,「這一拍的聲音什麼
    時候開始」會整個往後偏,量出來的對時就跟著偏一樣多。
    """
    loud = np.abs(clip) > floor
    if not loud.any():
        return clip
    first = int(np.argmax(loud))
    last = len(clip) - int(np.argmax(loud[::-1]))
    return clip[first:last]


def _speak_to_wav(words: list[str], paths: list[str]) -> bool:
    """用 Windows 內建的語音合成把數字念成 wav。做不到就回 False。"""
    if os.name != "nt":
        return False

    script = [
        "$ErrorActionPreference='Stop'",
        "Add-Type -AssemblyName System.Speech",
        "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer",
        "$s.Rate = 1",
    ]
    for word, path in zip(words, paths):
        escaped = path.replace("'", "''")
        script.append("$s.SetOutputToWaveFile('" + escaped + "')")
        script.append("$s.Speak('" + word + "')")
    script.append("$s.Dispose()")

    try:
        done = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command",
             "; ".join(script)],
            capture_output=True, timeout=60, creationflags=_CREATE_NO_WINDOW)
    except Exception:
        return False

    return done.returncode == 0 and all(
        os.path.isfile(p) and os.path.getsize(p) > 44 for p in paths)


def _synth_beep(samplerate: int, index: int) -> np.ndarray:
    """備援節拍聲:沒有語音合成可用時改放這個。

    第一拍高一個八度,聽得出四拍一循環的起點 —— 少了這個提示,純粹等距的
    嗶聲很容易跟丟。
    """
    freq = 880.0 if index == 0 else 587.0
    length = int(samplerate * 0.12)
    t = np.arange(length) / samplerate
    envelope = np.exp(-t * 22.0)
    return (np.sin(2 * np.pi * freq * t) * envelope * 0.5).astype(np.float32)


def count_clips(samplerate: int) -> tuple[list[np.ndarray], bool]:
    """1..4 的節拍聲(單聲道)。回傳 (片段, 是否為語音)。

    語音合成要跑一次外部行程,拿到之後就快取起來 —— 校準隨時可能再按一次,
    每次都等一兩秒不合理。
    """
    from ..media import ffmpeg as ffmpeg_mod

    folder = cache_dir("calibration")
    words = ["1", "2", "3", "4"]
    paths = [os.path.join(folder, "count-" + w + ".wav") for w in words]

    if not all(os.path.isfile(p) for p in paths):
        with tempfile.TemporaryDirectory() as tmp:
            staged = [os.path.join(tmp, "count-" + w + ".wav") for w in words]
            if _speak_to_wav(words, staged):
                for src, dst in zip(staged, paths):
                    try:
                        os.replace(src, dst)
                    except Exception:
                        pass

    clips: list[np.ndarray] = []
    for path in paths:
        if not os.path.isfile(path):
            break
        try:
            data = ffmpeg_mod.decode_to_array(path, samplerate, 1)
        except Exception:
            break
        clip = _trim_silence(np.asarray(data, dtype=np.float32).reshape(-1))
        if len(clip) == 0:
            break
        peak = float(np.max(np.abs(clip)))
        if peak > 0:
            clip = clip * (0.5 / peak)
        clips.append(clip.astype(np.float32))

    if len(clips) == len(words):
        return clips, True
    return [_synth_beep(samplerate, i) for i in range(len(words))], False


# ── 校準器 ──────────────────────────────────────────────────────────────
class BeatCalibrator:
    """放節拍、收麥克風,最後算出你的歌聲落在哪裡。

    render() 與 observe() 都在音訊回呼裡跑,所以只做搬移與累加;真正的偵測
    留到結束後在 result() 裡一次做完。
    """

    def __init__(self, samplerate: int, clips: list[np.ndarray],
                 voiced: bool, beats: int = BEATS,
                 beat_ms: float = BEAT_MS) -> None:
        self.samplerate = int(samplerate)
        self.beats = int(beats)
        self.interval = int(round(beat_ms * samplerate / 1000.0))
        self.voiced = voiced
        self._clips = clips

        # 開頭留一拍空白:讓使用者聽到第一聲之前先進入狀況,也給底噪一段
        # 乾淨的取樣。
        self._lead_in = self.interval
        self.beat_positions = [self._lead_in + i * self.interval
                               for i in range(self.beats)]
        # 最後一拍念完還要留時間收尾,否則最後一拍的聲音會被切掉
        self.total = self.beat_positions[-1] + self.interval * 2

        self._cursor = 0
        self._captured: list[np.ndarray] = []
        self.finished = False

    # -- 音訊回呼裡呼叫 --
    def render(self, frames: int) -> np.ndarray:
        """這個 block 要混進耳機的節拍聲,形狀 (frames, 1)。"""
        out = np.zeros((frames, 1), dtype=np.float32)
        start = self._cursor
        stop = start + frames

        for index, position in enumerate(self.beat_positions):
            clip = self._clips[index % len(self._clips)]
            clip_stop = position + len(clip)
            if clip_stop <= start or position >= stop:
                continue
            take_from = max(position, start)
            take_to = min(clip_stop, stop)
            out[take_from - start:take_to - start, 0] += \
                clip[take_from - position:take_to - position]

        return out

    def observe(self, mic_mono: np.ndarray | None, frames: int) -> None:
        """收下這個 block 的麥克風,並推進時間軸。"""
        if mic_mono is None or len(mic_mono) != frames:
            self._captured.append(np.zeros(frames, dtype=np.float32))
        else:
            self._captured.append(
                np.asarray(mic_mono, dtype=np.float32).reshape(-1).copy())
        self._cursor += frames
        if self._cursor >= self.total:
            self.finished = True

    @property
    def progress(self) -> float:
        return min(1.0, self._cursor / max(self.total, 1))

    @property
    def current_beat(self) -> int:
        """目前念到第幾拍(1 起算;還在前奏時是 0)。"""
        if self._cursor < self._lead_in:
            return 0
        return min(self.beats,
                   (self._cursor - self._lead_in) // self.interval + 1)

    def _cue_track(self, length: int) -> np.ndarray:
        """把整段節拍聲重建出來 —— 用來估耳機漏音有多大。"""
        track = np.zeros(length, dtype=np.float32)
        for index, position in enumerate(self.beat_positions):
            if position >= length:
                break
            clip = self._clips[index % len(self._clips)]
            end = min(position + len(clip), length)
            track[position:end] += clip[:end - position]
        return track

    # -- 結束後呼叫 --
    def result(self) -> dict:
        """算出對時建議值。"""
        signal = np.concatenate(self._captured) if self._captured \
            else np.zeros(0, dtype=np.float32)
        if len(signal) < self.interval:
            return {"ok": False, "reason": "沒有收到麥克風訊號"}

        hop = max(1, int(self.samplerate * 0.002))       # 2 ms 解析度
        usable = (len(signal) // hop) * hop
        envelope = np.abs(signal[:usable]).reshape(-1, hop).max(axis=1)
        if not envelope.size:
            return {"ok": False, "reason": "沒有收到麥克風訊號"}

        # 門檻取相對值:底噪與說話音量因人因裝置而異,寫死的絕對門檻在安靜
        # 的麥克風上會全部漏抓、在吵的上面全部誤觸。
        #
        # 參考點用 99 百分位而不是 95:節拍之間大多是靜音,95 百分位會被
        # 那些安靜的格子拉低,門檻跟著變低,結果**耳機漏音**(也就是節拍聲
        # 自己漏進麥克風)就會被當成你開口的瞬間 —— 量出來的偏移會整個
        # 偏向 0。0.4 的比例讓門檻穩穩落在漏音之上、人聲之下。
        floor = float(np.percentile(envelope, 20))
        peak = float(np.percentile(envelope, 99))
        # 兩道門都要過:
        #   絕對值 —— 安靜房間的底噪其實很尖(實測 floor 0.0001、peak 0.019,
        #             比值高達 140 倍),所以「相對比值」完全分不出有沒有人
        #             在講話。真的對著麥克風念出來會遠高於 0.03。
        #   相對值 —— 擋住整段都是穩定大聲的情況(例如麥克風一直破音)。
        if peak < 0.03 or peak - floor < 0.01:
            return {"ok": False,
                    "reason": "沒聽到你的聲音 —— 要跟著念出聲,"
                              "並確認麥克風選對了、沒有靜音、音量夠大"}
        threshold = floor + 0.4 * (peak - floor)

        # 估一下耳機漏音有多大。節拍聲是我們自己放的,所以可以把麥克風訊號
        # 投影到它身上,看漏回來多少。
        #
        # 這個估計刻意只當警訊、不當否決:你唸的正好也是「1234」,跟節拍聲
        # 有部分相關,投影會把你的聲音算進去一些,所以數字不夠準到能拿來
        # 判生死。但它足以認出「漏音大到跟人聲同一個量級」這種危險情況 ——
        # 那時偵測到的起點可能是節拍聲漏回來的,量出來的偏移會往 0 靠,
        # 看起來正常卻是錯的。
        cue = self._cue_track(usable)
        energy = float(np.dot(cue, cue))
        bleed_peak = 0.0
        if energy > 0:
            gain = float(np.dot(signal[:usable], cue) / energy)
            bleed_peak = abs(gain) * float(np.max(np.abs(cue)))
        bleed_risky = bleed_peak > threshold * 0.7

        deltas: list[float] = []
        for index, position in enumerate(self.beat_positions):
            if index < WARMUP_BEATS:
                continue
            lo = max(0, int((position - self.interval * 0.45) / hop))
            hi = min(len(envelope), int((position + self.interval * 0.55) / hop))
            if hi <= lo:
                continue
            above = np.flatnonzero(envelope[lo:hi] > threshold)
            if not above.size:
                continue
            onset = (lo + int(above[0])) * hop
            deltas.append((onset - position) / self.samplerate * 1000.0)

        if len(deltas) < 3:
            return {"ok": False,
                    "reason": f"只抓到 {len(deltas)} 拍 —— 請大聲一點,每一拍都要念出來"}

        values = np.array(deltas, dtype=np.float64)
        median = float(np.median(values))
        spread = float(np.percentile(values, 75) - np.percentile(values, 25))
        return {
            "ok": True,
            "offset_ms": round(median, 1),
            "spread_ms": round(spread, 1),
            "beats_detected": len(deltas),
            "beats_used": self.beats - WARMUP_BEATS,
            # 四分位距超過 60 ms 表示每拍念的位置差很多。中位數仍是最好的
            # 估計,但使用者該知道這次量得不穩,值得再測一次。
            "bleed": round(bleed_peak, 4),
            "bleed_risky": bleed_risky,
            "reliable": spread <= 60.0 and not bleed_risky,
            "warning": ("耳機漏音偏大,節拍聲可能被當成你的聲音 —— "
                        "把耳機音量調小或讓麥克風遠離耳機,再測一次比較保險")
                       if bleed_risky else "",
        }
