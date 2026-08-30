"""即時混音引擎。

訊號流
------
                  ┌── 分離 ──┐   ┌ 變調 ┐   ┌ 音樂 EQ ┐
  播放器(分軌) ──┤          ├──▶│      ├──▶│         ├─▶ 音樂推桿 ┬▶ ×送耳機 ┐
                  └──────────┘   └──────┘   └─────────┘            │          │
                                                                   └▶ ×送虛擬 ┼▶ 虛擬音效卡
  麥克風輸入 ──▶ 麥克風 EQ ──▶ 回音(可選)──▶ 麥克風推桿 ─┬─▶ ×送虛擬 ──────┘   (→ Discord)
                                                           │
                                                           └─▶ ×監聽(可勾選)▶ 耳機

回音接在推桿之前、兩路分岔之前,所以自己的耳返與對方聽到的是同一份效果。

送給 Discord 那一路可以對時(見 ``set_vc_sync_ms``):你是跟著耳機裡的
音樂唱的,歌聲混進去時天生就比音樂晚一個耳返延遲。對時可正可負 ——
延後音樂或延後歌聲,兩個方向都做得到。耳機那一路本身不受影響,除非
打開 ``monitor_send``,那會讓耳機直接聽送出去的同一份訊號。

時脈處理
--------
耳機輸出的回呼是主時脈:所有混音都在裡面完成,虛擬音效卡那一路寫進環形
緩衝、由它自己的回呼取用。麥克風同樣經過環形緩衝進來。兩者的裝置時脈會
慢慢飄移,靠緩衝的預留量吸收,超出上限就丟掉最舊的資料。
"""

from __future__ import annotations

import threading
import time

import numpy as np
import sounddevice as sd

from .. import BLOCK_SIZE, SAMPLE_RATE
from ..dsp.delay import CEILING_DELAY_MS, DelayLine
from ..dsp.denoise import MicDenoiser
from ..dsp.drift import DriftCorrector
from ..dsp.echo import Echo
from ..dsp.eq import GraphicEQ
from ..dsp.gain import SmoothGain, db_to_lin
from ..dsp.limiter import Limiter
from ..dsp.meters import MeterBank
from ..dsp.pitch import MAX_SEMITONES, PitchShifter
from ..dsp.separation import CenterSeparator, mode_from_flags
from ..dsp.spectral_separation import SpectralCenterSeparator
from ..dsp.ring import RingBuffer
from .calibrate import BeatCalibrator, count_clips
from .player import StemPlayer

METER_POINTS = ("music_in", "music_out", "mic_in", "mic_out", "hp_out", "vc_out")


class EngineError(RuntimeError):
    pass


class MixerParams:
    """所有可即時調整的參數。單純的屬性存取,音訊回呼直接讀。"""

    def __init__(self, samplerate: int = SAMPLE_RATE) -> None:
        r = lambda v: SmoothGain(v, 25.0, samplerate)  # noqa: E731

        self.music_fader = r(1.0)
        self.mic_fader = r(1.0)
        self.send_music_hp = r(1.0)
        self.send_music_vc = r(0.8)
        self.send_mic_vc = r(1.0)
        self.send_mic_monitor = r(0.5)
        self.master_hp = r(0.8)
        self.master_vc = r(0.8)

        self.monitor_self = False
        # 耳機改聽「送給 Discord 的那一份」。校 DC 對時時才聽得到效果。
        self.monitor_send = False
        self.mic_muted = False
        self.music_muted = False
        self.mic_to_headphone_enabled = False  # = monitor_self 的實際開關
        self.limiter = True

    def as_dict(self) -> dict:
        return {
            "music_fader": self.music_fader.target,
            "mic_fader": self.mic_fader.target,
            "send_music_hp": self.send_music_hp.target,
            "send_music_vc": self.send_music_vc.target,
            "send_mic_vc": self.send_mic_vc.target,
            "send_mic_monitor": self.send_mic_monitor.target,
            "master_hp": self.master_hp.target,
            "master_vc": self.master_vc.target,
            "monitor_self": self.monitor_self,
            "monitor_send": self.monitor_send,
            "mic_muted": self.mic_muted,
            "music_muted": self.music_muted,
            "limiter": self.limiter,
        }


class AudioEngine:
    def __init__(self, samplerate: int = SAMPLE_RATE, blocksize: int = BLOCK_SIZE) -> None:
        self.samplerate = samplerate
        self.blocksize = blocksize

        self.player = StemPlayer(samplerate)
        self.params = MixerParams(samplerate)
        # 兩種即時分離演算法並存,可即時切換:
        #   fast     時域 mid/side —— 零延遲,但對整個頻段做同一決策
        #   quality  頻譜域逐格 —— 約 20 ms 延遲,實測伴奏 SDR 高 7 dB
        self._separators: dict[str, object] = {
            "fast": CenterSeparator(samplerate),
            "quality": SpectralCenterSeparator(samplerate),
        }
        self.separator_quality = "quality"
        self.separator = self._separators["quality"]
        # 送給 Discord 那一路的「音樂 vs 歌聲」對時,可正可負。
        #
        # 訊號是即時的,沒辦法真的把任何一路「提前」——「提前音樂」只能
        # 靠「延後歌聲」達成,反之亦然。所以這裡放兩條延遲線,依號誌決定
        # 延遲哪一條,永遠只有一條在作用:
        #
        #   vc_sync_ms > 0   延後音樂 → 歌聲相對變早
        #   vc_sync_ms < 0   延後歌聲 → 音樂相對變早
        #
        # 這樣任何方向都做得到,而且加進去的絕對延遲永遠是 |vc_sync_ms|,
        # 不會為了「提前」而先墊一段基準延遲。
        self._vc_music_delay = DelayLine(samplerate, 2)
        self._vc_mic_delay = DelayLine(samplerate, 2)
        self._vc_sync_ms = 0.0
        # 音遊式校準器。只有按下校準時才存在,平常是 None,
        # _produce 靠這個判斷要不要接管耳機。
        self._calibrator: BeatCalibrator | None = None
        self._calibration_clips: list | None = None
        self._calibration_voiced = False
        # 升 key / 降 key。0 半音時整條旁通,不花 CPU 也不加延遲。
        self.music_pitch = PitchShifter(samplerate, 2)
        self.music_eq = GraphicEQ(samplerate, 2)
        # 麥克風底噪處理(電源哼聲 + 嘶聲)。預設關閉,零延遲。
        # 位置在 EQ 之前:先把哼聲切掉,EQ 才不會又把它推回來。
        self.mic_denoise = MicDenoiser(samplerate, 1)
        self.mic_eq = GraphicEQ(samplerate, 1)
        # 麥克風回音是效果器,預設關閉;開了之後耳機與 Discord 兩邊都聽得到,
        # 因為它接在麥克風那一路的共用段落上。
        self.mic_echo = Echo(samplerate, 1)
        # 兩路各自的限幅器。前瞻一個 block,所以會多一個 block 的延遲,
        # 換來的是不再對正常響度的訊號做波形整形(實測 0 dBFS 的 THD+N
        # 從 −30.9 dB 進步到 −99.7 dB)。
        self.hp_limiter = Limiter(samplerate, 2)
        self.vc_limiter = Limiter(samplerate, 2)
        self.meters = MeterBank(METER_POINTS, samplerate)

        self.headphone_device: int | None = None
        self.virtual_device: int | None = None
        self.mic_device: int | None = None
        self.mic_channels = 1
        # WASAPI 獨佔模式:延遲低很多,但該裝置會被本程式獨佔,
        # 其他程式(含 Discord 本身的提示音)就發不出聲音。
        self.exclusive_mode = False
        self.exclusive_active: list[str] = []
        self.exclusive_notes: list[str] = []
        # 端點取樣率與引擎不同、由 WASAPI 代為轉換的紀錄。不是錯誤,
        # 只是讓「聽起來怪」的時候有跡可循。
        self.format_notes: list[str] = []

        # 環回實測得到的真實延遲(毫秒)。None 表示還沒校正過。
        # latency_report() 的估計值會低估約一倍,只有這個是可信的絕對值。
        self.calibrated_monitor_ms: float | None = None

        self._streams: list[sd.Stream] = []
        self._mic_stream: sd.InputStream | None = None
        self._hp_stream: sd.OutputStream | None = None
        self._vc_stream: sd.OutputStream | None = None
        self._pump_thread: threading.Thread | None = None
        self._pump_stop = threading.Event()

        self._mic_ring = RingBuffer(samplerate, 1)          # 1 秒
        # 虛擬音效卡那一路走漂移補償重取樣,不是單純的環形緩衝。
        # 它的資料由耳機回呼產生、卻由虛擬卡自己的回呼取用,兩個時脈差
        # 幾十 ppm;單純丟樣本補不了速率差,虛擬卡驅動只好自己插補,
        # 聽起來就是斷續與電流聲(實測 −40.5 dB 的寬頻雜訊)。
        self._vc_ring = DriftCorrector(samplerate, 2,
                                       target_fill=self._vc_target(blocksize))

        # 跨時脈緩衝只需要一個 block 的抖動餘裕。多出來的都是純延遲,
        # 由 _trim_slack() 持續削掉 —— 對耳返來說這是最大的可控延遲來源。
        self._slack_target = blocksize
        self._mic_fill_min: int | None = None
        self._vc_fill_min: int | None = None
        # 存量在一個 block 的範圍內來回跳,瞬時值不能代表實際延遲,取平滑值
        self._mic_fill_avg: float | None = None
        self._vc_fill_avg: float | None = None
        self._trim_ticks = 0
        self._trim_interval = self._interval_for(blocksize)
        # 主時脈停住時的安全上限。刻意用時間而非 block 數 —— 用 block 數的話
        # 上限會跟著 block 大小放大(480 取樣時是 80 ms),那是純浪費。
        self._ring_cap = self._cap_for(blocksize)

        self._lock = threading.RLock()
        self._running = False
        self.last_error = ""
        self.xruns = 0
        self._cpu_load = 0.0

    # ── 生命週期 ────────────────────────────────────────────────────────
    def _interval_for(self, blocksize: int) -> int:
        """修剪週期(以 block 為單位)= 一秒。

        試過縮短到 0.25 秒讓延遲收斂快一點,實機量測反而變差
        (獨佔 / 128 取樣:12.9 ms → 13.8 ms)。取最低存量的視窗越短,
        看到的最低值就越高,削掉的自然越少 —— 這個週期要長到足以
        看見真正的谷底。
        """
        return max(1, int(self.samplerate / blocksize))

    def _vc_target(self, blocksize: int) -> int:
        """虛擬卡緩衝的目標存量。

        虛擬卡在共享模式下是**成串呼叫**的:WASAPI 共享的週期約 10 ms,
        會連續叫好幾次再等一陣子。緩衝必須蓋得住一整串,否則串到一半就
        見底 —— 實測 3 個 block(bs=64 時只有 4 ms)有 25% 的呼叫讀不到
        資料,存量在 2 到 515 之間劇烈擺盪。

        實測:4 個 block / 10 ms 在輕載下是 0 次,但把分離、變調、回音、
        降噪全開之後(CPU 32%)產生端更抖,又出現 0.6% 的欠載。
        6 個 block 且至少 20 ms 才在滿載下也是 0 次。這一路餵的是 Discord,
        後面本來就有網路與抖動緩衝,多這十幾毫秒沒有實質損失。
        """
        return max(6 * blocksize, int(self.samplerate * 0.020))

    def _cap_for(self, blocksize: int) -> int:
        """環形緩衝的安全上限(取樣)。至少四個 block,至少 25 ms。"""
        return max(blocksize * 4, int(self.samplerate * 0.025))

    @property
    def running(self) -> bool:
        return self._running

    def start(self) -> dict:
        with self._lock:
            if self._running:
                return self.status()
            if self.headphone_device is None and self.virtual_device is None:
                raise EngineError("至少要選一個輸出裝置(耳機或虛擬音效卡)。")

            self.last_error = ""
            self._mic_ring.clear()
            self._vc_ring.prime()
            self._mic_fill_min = None
            self._vc_fill_min = None
            self._mic_fill_avg = None
            self._vc_fill_avg = None
            self._trim_ticks = 0
            self.music_eq.clear_state()
            self.mic_eq.clear_state()
            self.mic_denoise.reset()
            self.music_pitch.reset()
            self.mic_echo.reset()
            self._vc_music_delay.reset()
            self._vc_mic_delay.reset()
            self.hp_limiter.reset()
            self.vc_limiter.reset()
            self.separator.reset()
            self.meters.reset()

            try:
                self._open_streams()
            except Exception as exc:
                self._close_streams()
                raise EngineError(f"開啟音訊裝置失敗: {exc}") from exc

            self._running = True
            return self.status()

    def stop(self) -> dict:
        with self._lock:
            self._close_streams()
            self._running = False
            return self.status()

    def restart(self) -> dict:
        was = self._running
        self.stop()
        if was:
            return self.start()
        return self.status()

    def _is_wasapi(self, device: int) -> bool:
        try:
            host = sd.query_hostapis(sd.query_devices(device)["hostapi"])["name"]
        except Exception:
            return False
        return host == "Windows WASAPI"

    def _shared_settings(self, device: int):
        """共享模式的 extra_settings;非 WASAPI 裝置回傳 None。

        WASAPI 共享模式下,串流的取樣率必須跟端點的 mix format 一致,否則
        根本開不起來。而 mix format 是使用者在「音效 → 進階 → 裝置內容」
        設的,虛擬卡出廠常常不是 48 kHz,兩端也未必一樣 —— 原本的做法是
        寫死 48000 開下去,對不上就得叫使用者自己回控制台調。

        ``auto_convert`` 讓 WASAPI 自己插入系統的取樣率轉換器與聲道矩陣。
        這就是一般應用程式播音樂時走的路徑:程式給什麼格式,Windows 就轉
        什麼格式,不需要任何人去對齊設定。獨佔模式沒有這一層(也不該有),
        但虛擬卡本來就不走獨佔。
        """
        if not self._is_wasapi(device):
            return None
        return sd.WasapiSettings(auto_convert=True)

    def _note_format(self, device: int, label: str) -> None:
        """端點取樣率與引擎不同時記一筆,讓「記錄」分頁看得到轉換發生在哪。

        只對 WASAPI 講 —— 其他 host API 的轉換是 PortAudio 自己做的,
        機制不同,不該掛上 WASAPI 的名字。
        """
        if not self._is_wasapi(device):
            return
        try:
            rate = int(sd.query_devices(device)["default_samplerate"])
        except Exception:
            return
        if rate != self.samplerate:
            self.format_notes.append(
                f"{label}:端點為 {rate} Hz,引擎為 {self.samplerate} Hz,"
                "已由 WASAPI 自動轉換。")

    def _exclusive_settings(self, device: int):
        """獨佔模式的 extra_settings;非 WASAPI 裝置回傳 None。

        虛擬音效卡一律不套獨佔,即使使用者開了獨佔模式。
        """
        if not self.exclusive_mode:
            return None

        # 虛擬音效卡走獨佔會爛掉。實測(注入純音,從 CABLE Output 錄回):
        #
        #     獨佔   每秒 111 次掉落、34% 的時間是空洞、雜訊與主音同量級
        #     共享   0 次掉落、雜訊 −98.5 dB
        #
        # 那 111 Hz 的週期性掉落就是使用者聽到的「斷續 + 電流感」。虛擬卡
        # 是軟體裝置,它的時脈由驅動自己產生,獨佔那條路顯然沒有被好好
        # 實作。而且它本來就不該獨佔:這一路餵的是 Discord,後面還有網路
        # 與抖動緩衝,省那幾毫秒毫無意義,卻會擋住其他程式使用同一張卡。
        if self.virtual_device is not None and device == self.virtual_device:
            return None
        if not self._is_wasapi(device):
            return None
        return sd.WasapiSettings(exclusive=True)

    def _make_stream(self, kind: str, device: int, channels: int, callback, label: str):
        """開串流,獨佔模式失敗時自動退回共享模式。

        獨佔模式對格式很挑(取樣率、位元深度必須被裝置原生支援),
        失敗是常態而不是例外,所以一定要有退路。
        """
        cls = sd.InputStream if kind == "input" else sd.OutputStream
        common = dict(samplerate=self.samplerate, dtype="float32",
                      blocksize=self.blocksize, latency="low")

        settings = self._exclusive_settings(device)
        if settings is not None:
            try:
                stream = cls(device=device, channels=channels, callback=callback,
                             extra_settings=settings, **common)
                self.exclusive_active.append(label)
                return stream
            except Exception as exc:
                note = f"{label}:無法使用獨佔模式,已退回共享模式({exc})"
                self.exclusive_notes.append(note)
                self.last_error = note

        # 共享模式。取樣率的協商交給 WASAPI(見 _shared_settings)——
        # 不再假設端點是 48 kHz,也不再要求使用者手動對齊。
        self._note_format(device, label)
        return cls(device=device, channels=channels, callback=callback,
                   extra_settings=self._shared_settings(device), **common)

    def _open_streams(self) -> None:
        self.exclusive_active = []
        self.exclusive_notes = []
        self.format_notes = []

        if self.mic_device is not None:
            channels = max(1, min(2, int(self.mic_channels)))
            self._mic_stream = self._make_stream(
                "input", self.mic_device, channels, self._mic_callback, "麥克風")
            self._mic_stream.start()

        # 主時脈:優先耳機;沒選耳機時由虛擬音效卡兼任
        hp_is_master = self.headphone_device is not None

        if hp_is_master:
            self._hp_stream = self._make_stream(
                "output", self.headphone_device, 2, self._hp_callback, "耳機")

        if self.virtual_device is not None:
            self._vc_stream = self._make_stream(
                "output", self.virtual_device, 2,
                self._vc_callback if hp_is_master else self._vc_master_callback,
                "虛擬音效卡")

        # 先開虛擬卡再開耳機:耳機的回呼是產生端,虛擬卡是取用端。
        # 反過來的話產生端會先跑一段沒人取用的時間,存量一口氣衝高。
        if self._vc_stream is not None:
            self._vc_stream.start()
        if self._hp_stream is not None:
            self._hp_stream.start()

    def _close_streams(self) -> None:
        for stream in (self._hp_stream, self._vc_stream, self._mic_stream):
            if stream is None:
                continue
            try:
                stream.stop()
            except Exception:
                pass
            try:
                stream.close()
            except Exception:
                pass
        self._hp_stream = self._vc_stream = self._mic_stream = None

    # ── 音訊回呼 ────────────────────────────────────────────────────────
    def _mic_callback(self, indata, frames, time_info, status) -> None:
        if status:
            self.xruns += 1
        if indata.shape[1] == 1:
            mono = indata.reshape(frames, 1)
        else:
            mono = np.mean(indata[:, :2], axis=1, keepdims=True)
        self._mic_ring.write(np.ascontiguousarray(mono, dtype=np.float32))

        # 安全網:主時脈若整個停住,別讓緩衝無限長大
        excess = self._mic_ring.available() - self._ring_cap
        if excess > 0:
            self._mic_ring.drop(excess)

    def _hp_callback(self, outdata, frames, time_info, status) -> None:
        if status:
            self.xruns += 1
        started = time.perf_counter()
        hp, vc = self._produce(frames)
        outdata[:] = hp
        if self._vc_stream is not None:
            self._vc_ring.write(vc)
        self._track_load(started, frames)

    def _vc_master_callback(self, outdata, frames, time_info, status) -> None:
        if status:
            self.xruns += 1
        started = time.perf_counter()
        _, vc = self._produce(frames)
        outdata[:] = vc
        self._track_load(started, frames)

    def _vc_callback(self, outdata, frames, time_info, status) -> None:
        if status:
            self.xruns += 1
        # 存量由 DriftCorrector 自己用重取樣速率調節,這裡不要插手 ——
        # 丟樣本正是原本產生雜訊的行為。
        outdata[:] = self._vc_ring.read(frames)

    def _trim_slack(self) -> None:
        """削掉環形緩衝裡持續閒置的存量。

        每個 block 記錄一次存量,累積約一秒後取這段期間的**最低值** —— 那是
        「無論如何都沒被用到」的部分,削掉它不會造成 underrun。保留一個 block
        當抖動餘裕。這讓延遲自動收斂到當下裝置組合能達到的最低值,
        而不是寫死一個保守的預留量。
        """
        mic_fill = self._mic_ring.available()
        vc_fill = self._vc_ring.available()
        self._mic_fill_min = mic_fill if self._mic_fill_min is None \
            else min(self._mic_fill_min, mic_fill)
        self._vc_fill_min = vc_fill if self._vc_fill_min is None \
            else min(self._vc_fill_min, vc_fill)

        alpha = 0.02
        self._mic_fill_avg = float(mic_fill) if self._mic_fill_avg is None \
            else self._mic_fill_avg + alpha * (mic_fill - self._mic_fill_avg)
        self._vc_fill_avg = float(vc_fill) if self._vc_fill_avg is None \
            else self._vc_fill_avg + alpha * (vc_fill - self._vc_fill_avg)

        self._trim_ticks += 1
        if self._trim_ticks < self._trim_interval:
            return
        self._trim_ticks = 0

        if self._mic_stream is not None and self._mic_fill_min is not None:
            slack = self._mic_fill_min - self._slack_target
            if slack > 0:
                self._mic_ring.drop(slack)

        self._mic_fill_min = None
        self._vc_fill_min = None

    def _track_load(self, started: float, frames: int) -> None:
        budget = frames / self.samplerate
        self._cpu_load = 0.9 * self._cpu_load + 0.1 * ((time.perf_counter() - started) / budget)

    # ── 混音核心 ────────────────────────────────────────────────────────
    def _produce(self, frames: int) -> tuple[np.ndarray, np.ndarray]:
        p = self.params

        # --- 音樂路徑 ---
        music = self.player.read(frames)
        self.meters["music_in"].process(music)
        music = self.separator.process(music)
        # 變調排在分離之後、EQ 之前:分離要看的是原始的左右相位關係,而 EQ
        # 的頻率標的是使用者聽到的最終音高。
        music = self.music_pitch.process(music)
        music = self.music_eq.process(music)
        music = music * p.music_fader.envelope(frames)
        if p.music_muted:
            music = np.zeros_like(music)
        self.meters["music_out"].process(music)

        # --- 麥克風路徑 ---
        mic_mono = self._mic_ring.read(frames) if self._mic_stream is not None \
            else np.zeros((frames, 1), dtype=np.float32)
        self.meters["mic_in"].process(mic_mono)
        # 校準要看的是「進到混音的原始聲音」,所以在 EQ、回音、推桿、
        # 靜音之前先留一份。走完那些之後才取的話,推桿位置與靜音狀態
        # 都會影響量測結果,而那是量測不該在意的東西。
        mic_capture = mic_mono
        mic_mono = self.mic_denoise.process(mic_mono)
        mic_mono = self.mic_eq.process(mic_mono)
        mic_mono = self.mic_echo.process(mic_mono)
        mic_mono = mic_mono * p.mic_fader.envelope(frames)
        if p.mic_muted:
            mic_mono = np.zeros_like(mic_mono)
        self.meters["mic_out"].process(mic_mono)
        mic = np.repeat(mic_mono, 2, axis=1)

        # --- 送給 Discord 的匯流排 ---
        # 對時永遠只延遲其中一路,號誌決定是哪一路(見 _vc_sync_ms)。
        # 兩條延遲線都要每個 block 餵到,不作用的那條才有歷史音訊可讀,
        # 對時號誌翻面的瞬間才不會出現一段空白。
        vc = self._vc_music_delay.process(music) * p.send_music_vc.envelope(frames)
        vc = vc + self._vc_mic_delay.process(mic) * p.send_mic_vc.envelope(frames)
        vc = vc * p.master_vc.envelope(frames)

        if p.limiter:
            vc = self.vc_limiter.process(vc)
        else:
            np.clip(vc, -1.0, 1.0, out=vc)

        # --- 耳機 ---
        # 三個 envelope 無論走哪個分支都各取一次。少取的那個平滑狀態會停在
        # 原地,切換監聽模式的瞬間就會跳一下音量。
        music_hp_env = p.send_music_hp.envelope(frames)
        monitor_env = p.send_mic_monitor.envelope(frames)
        master_hp_env = p.master_hp.envelope(frames)

        if p.monitor_send:
            # 耳機直接聽「送出去的那一份」—— 同一個來源分兩端,一端進耳機
            # 一端進虛擬音效卡,差別只有這裡的耳機主音量。
            #
            # 要用耳朵校 DC 對時就得這樣聽:平常耳機聽的是還沒對時的本地
            # 混音,對時調了也感覺不到,只能靠對方回報。
            hp = vc * master_hp_env
            # vc 已經限幅過了,這裡不能再軟限一次 —— 再壓一遍就不是對方
            # 收到的那份波形了,而這條路存在的意義就是「聽到一模一樣的」。
            # 只留硬性夾位當安全網,因為耳機主音量可以推到 +12 dB。
            np.clip(hp, -1.0, 1.0, out=hp)
        else:
            hp = music * music_hp_env
            if p.monitor_self:
                hp = hp + mic * monitor_env
            hp = hp * master_hp_env

            if p.limiter:
                hp = self.hp_limiter.process(hp)
            else:
                np.clip(hp, -1.0, 1.0, out=hp)

        # --- 校準(音遊式對時)---
        # 蓋掉耳機內容而不是疊上去:量的是「聽到拍子 → 開口 → 被收進混音」
        # 這一圈,伴奏漏音進麥克風會被誤判成人聲起點,把結果整個帶偏。
        # 送出那一路也一併靜音,免得把數數聲丟給通話中的對方。
        calibrator = self._calibrator
        if calibrator is not None and not calibrator.finished:
            cue = np.repeat(calibrator.render(frames), 2, axis=1)
            hp = cue * master_hp_env
            np.clip(hp, -1.0, 1.0, out=hp)
            vc = np.zeros_like(vc)
            calibrator.observe(mic_capture, frames)

        self.meters["hp_out"].process(hp)
        self.meters["vc_out"].process(vc)
        self._trim_slack()
        return hp, vc

    # ── 參數 ────────────────────────────────────────────────────────────
    def set_gain_db(self, name: str, db: float) -> None:
        gain = getattr(self.params, name, None)
        if isinstance(gain, SmoothGain):
            gain.target = db_to_lin(float(db))
        else:
            raise EngineError(f"未知的增益: {name}")

    def set_gain_linear(self, name: str, value: float) -> None:
        gain = getattr(self.params, name, None)
        if isinstance(gain, SmoothGain):
            gain.target = max(0.0, min(4.0, float(value)))
        else:
            raise EngineError(f"未知的增益: {name}")

    def set_blocksize(self, blocksize: int) -> None:
        """調整 block 大小。需要重開串流才會生效。"""
        # 下限 32:實機量測 32 取樣在獨佔模式下是 8.8 ms、0 xrun,DSP 只吃掉
        # 15% 的預算。再低就沒有量測支持了,而爆音比幾毫秒難聽得多。
        blocksize = max(32, min(2048, int(blocksize)))
        self.blocksize = blocksize
        self._slack_target = blocksize
        self._trim_interval = self._interval_for(blocksize)
        self._ring_cap = self._cap_for(blocksize)
        self._vc_ring.target_fill = self._vc_target(blocksize)

    @property
    def vc_sync_ms(self) -> float:
        return self._vc_sync_ms

    def set_vc_sync_ms(self, value: float) -> float:
        """送給 Discord 那一路的「音樂 vs 歌聲」對時(毫秒,可正可負)。

        正值延後音樂、負值延後歌聲 —— 同一個相對位移的兩個方向。任何時候
        只有一條延遲線在作用,所以加進去的絕對延遲就是 |value|,不會為了
        能「提前」而先墊一段基準延遲。

        沒有功能上的上下限 —— 延遲線的緩衝會跟著設定值長大。唯一擋住的是
        ±CEILING_DELAY_MS(60 秒),那是記憶體的保險絲而不是功能限制:
        少打一個小數點的話會當場配置掉幾百 GB。
        """
        value = float(value)
        if value != value:                       # NaN
            value = 0.0
        value = max(-CEILING_DELAY_MS, min(CEILING_DELAY_MS, value))
        self._vc_sync_ms = value

        self._vc_music_delay.delay_ms = value if value > 0 else 0.0
        self._vc_mic_delay.delay_ms = -value if value < 0 else 0.0
        return value

    def set_music_pitch(self, semitones: float) -> dict:
        """升 key / 降 key(半音;0 = 原調)。即時生效,不用重開串流。"""
        self.music_pitch.semitones = max(-MAX_SEMITONES,
                                         min(MAX_SEMITONES, float(semitones)))
        return self.music_pitch.as_dict()

    def set_mic_echo(self, enabled: bool | None = None, delay_ms: float | None = None,
                     feedback: float | None = None, mix: float | None = None,
                     damping: float | None = None) -> dict:
        """調整麥克風回音。傳 None 表示該項不動。"""
        echo = self.mic_echo
        if delay_ms is not None:
            echo.delay_ms = delay_ms
        if feedback is not None:
            echo.feedback = feedback
        if mix is not None:
            echo.mix = mix
        if damping is not None:
            echo.damping = damping
        if enabled is not None and bool(enabled) != echo.enabled:
            echo.enabled = bool(enabled)
            # 關掉時把緩衝清乾淨。留著的話下次開啟會先聽到幾百毫秒前的
            # 舊聲音,像是有人在旁邊插話。
            if not echo.enabled:
                echo.reset()
        return echo.as_dict()

    def suggested_vc_sync_ms(self) -> float | None:
        """建議**起點** = 目前的耳返延遲(正值 = 延後音樂)。

        歌聲比音樂晚的量,正好就是「聽到音樂 → 唱出來 → 被收進混音」這一圈,
        也就是耳返延遲。但那只是個起點:實際還會疊上你自己的節奏習慣,
        而且可能往反方向 —— 所以介面上要能兩邊調,套了不代表就結束。

        沒同時選好麥克風與耳機時算不出來,回 None 讓介面自己決定顯示什麼。
        """
        report = self.latency_report()
        monitor = report.get("monitor_ms")
        if not isinstance(monitor, (int, float)):
            return None
        return round(min(float(monitor), CEILING_DELAY_MS), 1)

    # ── 音遊式校準 ──────────────────────────────────────────────────────
    def start_calibration(self) -> dict:
        """開始校準:耳機放「1 2 3 4」,同時錄下麥克風。

        需要麥克風與耳機都選好 —— 少一邊就沒有迴圈可量。
        """
        if not self._running:
            raise EngineError("請先啟動音訊再校準。")
        if self.mic_device is None:
            raise EngineError("校準需要選好麥克風。")
        if self.headphone_device is None:
            raise EngineError("校準需要選好耳機。")

        # 語音合成要跑外部行程,不能在音訊回呼裡做;先在這裡備好並快取。
        if self._calibration_clips is None:
            clips, voiced = count_clips(self.samplerate)
            self._calibration_clips = clips
            self._calibration_voiced = voiced

        calibrator = BeatCalibrator(self.samplerate, self._calibration_clips,
                                    self._calibration_voiced)
        self._calibrator = calibrator
        return {
            "beats": calibrator.beats,
            "beat_ms": calibrator.interval / self.samplerate * 1000.0,
            "duration_ms": calibrator.total / self.samplerate * 1000.0,
            "voiced": calibrator.voiced,
        }

    def cancel_calibration(self) -> None:
        self._calibrator = None

    def calibration_state(self) -> dict:
        calibrator = self._calibrator
        if calibrator is None:
            return {"active": False}
        return {
            "active": not calibrator.finished,
            "finished": calibrator.finished,
            "progress": round(calibrator.progress, 3),
            "beat": calibrator.current_beat,
            "beats": calibrator.beats,
        }

    def calibration_result(self) -> dict:
        """取出結果並收掉校準器。只有跑完才拿得到。"""
        calibrator = self._calibrator
        if calibrator is None:
            return {"ok": False, "reason": "沒有進行中的校準"}
        if not calibrator.finished:
            return {"ok": False, "reason": "校準還沒結束"}
        self._calibrator = None
        return calibrator.result()

    def set_mic_denoise(self, **kwargs) -> dict:
        """調整麥克風降噪。傳 None 的項目不動。"""
        d = self.mic_denoise
        if kwargs.get("enabled") is not None:
            d.enabled = bool(kwargs["enabled"])
        if kwargs.get("gate_enabled") is not None:
            d.gate_enabled = bool(kwargs["gate_enabled"])
        if kwargs.get("hum_hz") is not None:
            d.hum_hz = kwargs["hum_hz"]
        if kwargs.get("auto_hum") is not None:
            want = bool(kwargs["auto_hum"])
            if want != d.auto_hum:
                d.auto_hum = want
                # 關掉時要把已偵測到的陷波一起撤掉,否則它們會留在濾波器裡
                if not want:
                    d._auto_freqs = []
                d._rebuild()
        if kwargs.get("highpass_hz") is not None:
            d.highpass_hz = kwargs["highpass_hz"]
        if kwargs.get("margin_db") is not None:
            d.margin_db = kwargs["margin_db"]
        if kwargs.get("reduction_db") is not None:
            d.reduction_db = kwargs["reduction_db"]
        if kwargs.get("learn"):
            d.learn_floor(float(kwargs.get("learn_seconds", 1.0)))
        return d.as_dict()

    def set_flag(self, name: str, value: bool) -> None:
        if not hasattr(self.params, name):
            raise EngineError(f"未知的開關: {name}")
        setattr(self.params, name, bool(value))

    def set_separator_quality(self, quality: str) -> dict:
        """切換即時分離演算法。

        兩個分離器都常駐,切換時把目前的參數(模式、強度、頻段)搬過去,
        使用者不會因為切換而丟失設定。舊的那個要 reset,免得下次切回來時
        殘留舊的緩衝內容。
        """
        if quality not in self._separators:
            raise EngineError(
                f"未知的分離品質 {quality};可用:{list(self._separators)}")
        if quality == self.separator_quality:
            return self.separator_info()

        old = self.separator
        new = self._separators[quality]

        new.mode = old.mode
        new.strength = old.strength
        new.low_cut = old.low_cut
        new.high_cut = old.high_cut
        if hasattr(old, "reset"):
            old.reset()
        if hasattr(new, "reset"):
            new.reset()

        self.separator_quality = quality
        self.separator = new
        return self.separator_info()

    def separator_info(self) -> dict:
        sep = self.separator
        latency = getattr(sep, "latency_samples", 0)
        return {
            "quality": self.separator_quality,
            "mode": sep.mode,
            "strength": sep.strength,
            "low_cut": sep.low_cut,
            "high_cut": sep.high_cut,
            "sharpness": getattr(sep, "sharpness", None),
            "latency_samples": latency,
            "latency_ms": round(latency / self.samplerate * 1000.0, 1),
            "available": list(self._separators),
        }

    def apply_separation_flags(self, remove_vocals: bool, remove_instrumental: bool,
                               realtime: bool) -> None:
        """勾選框 → 分軌增益(Demucs 模式)或即時分離模式。"""
        if realtime or "mix" in self.player.stem_names:
            self.separator.mode = mode_from_flags(remove_vocals, remove_instrumental)
            self.player.set_stem_gain("mix", 1.0)
        else:
            self.separator.mode = "off"
            self.player.set_stem_gains({
                "vocals": 0.0 if remove_vocals else 1.0,
                "instrumental": 0.0 if remove_instrumental else 1.0,
            })

    # ── 狀態 ────────────────────────────────────────────────────────────
    @staticmethod
    def _stream_latency(stream) -> float:
        """PortAudio 回報的串流延遲(秒)。雙向串流回傳的是 tuple。"""
        if stream is None:
            return 0.0
        try:
            latency = stream.latency
        except Exception:
            return 0.0
        if isinstance(latency, (tuple, list)):
            return float(latency[0]) if latency else 0.0
        return float(latency)

    # 延遲差距大到不能只靠使用者自己看裝置名稱:同一支耳麥在 MME 下
    # PortAudio 回報 90 ms、WASAPI 只有 3 ms。選錯 host API 的話,底下
    # 再怎麼調 block 都是白費工,所以要能在介面上明講。
    LOW_LATENCY_HOSTAPIS = ("Windows WASAPI", "Windows WDM-KS", "ASIO",
                            "Core Audio", "ALSA", "JACK Audio Connection Kit")

    def _hostapi_of(self, device: int | None) -> str:
        if device is None:
            return ""
        try:
            return sd.query_hostapis(sd.query_devices(device)["hostapi"])["name"]
        except Exception:
            return ""

    def hostapi_report(self) -> dict:
        """三個裝置各自走哪個 host API,以及哪些是高延遲的路徑。

        虛擬音效卡不列入警告 —— 它那一路的延遲影響的是對方聽到你的時間,
        不是耳返,而且虛擬驅動本來就不一定有 WASAPI 端點。
        """
        used = {
            "mic": self._hostapi_of(self.mic_device),
            "headphone": self._hostapi_of(self.headphone_device),
            "virtual": self._hostapi_of(self.virtual_device),
        }
        slow = [role for role in ("mic", "headphone")
                if used[role] and used[role] not in self.LOW_LATENCY_HOSTAPIS]
        return {"hostapis": used, "slow_roles": slow}

    def latency_report(self) -> dict:
        """各條路徑的端到端延遲估計(毫秒)。

        耳返延遲 = 麥克風輸入緩衝 + 環形緩衝存量 + 耳機輸出緩衝 + 限幅器前瞻。

        DSP 幾乎都是零延遲的(逐 block 或 IIR),唯一的例外是限幅器 ——
        它要先看下一個 block 的峰值才能決定增益,所以固定欠一個 block。
        對時的延遲線不算在這裡:它只作用在送給 Discord 那一路。
        """
        sr = float(self.samplerate)
        mic_in = self._stream_latency(self._mic_stream)
        hp_out = self._stream_latency(self._hp_stream)
        vc_out = self._stream_latency(self._vc_stream)
        mic_fill = self._mic_fill_avg if self._mic_fill_avg is not None \
            else self._mic_ring.available()
        vc_fill = self._vc_fill_avg if self._vc_fill_avg is not None \
            else self._vc_ring.available()
        mic_ring = mic_fill / sr
        vc_ring = vc_fill / sr

        report = {
            "mic_in_ms": round(mic_in * 1000, 2),
            "hp_out_ms": round(hp_out * 1000, 2),
            "vc_out_ms": round(vc_out * 1000, 2),
            "mic_buffer_ms": round(mic_ring * 1000, 2),
            "vc_buffer_ms": round(vc_ring * 1000, 2),
            "block_ms": round(self.blocksize / sr * 1000, 2),
        }

        # 耳返:自己的聲音繞一圈回到耳朵。
        #
        # ⚠️ 這是**下限估計**,不是真實延遲。
        #
        # PortAudio 回報的 stream.latency 只涵蓋它自己配置的緩衝。WASAPI
        # 共享模式下,Windows 音訊引擎(audiodg.exe)還有一層混音緩衝完全
        # 不在其中。環回實測顯示真實來回延遲約為此估計值的兩倍:
        #
        #     共享模式   估計 ~55 ms   實測 ~104 ms
        #     獨佔模式   估計 ~14 ms   實測  ~52 ms
        #
        # 所以這個欄位只適合做**相對比較**(哪個設定比較好),絕對值請以
        # tests/test_roundtrip.py 的環回實測為準。
        # 不去猜「真實值是估計的幾倍」—— 那個比例因驅動與機器而異,用單一
        # 機器的數據硬湊一個係數,只會產生看起來精確的錯誤數字。
        # 要真實值就得實際量(見 calibrated_monitor_ms)。
        # 限幅器的前瞻是真的延遲,一定要算進去 —— 不算的話介面上顯示的
        # 耳返會比實際低一個 block,使用者照著那個數字去對時就會偏掉。
        limiter_ms = self.hp_limiter.latency_samples / sr
        report["limiter_ms"] = round(limiter_ms * 1000, 2)

        if self._mic_stream is not None and self._hp_stream is not None:
            report["monitor_ms"] = round(
                (mic_in + mic_ring + hp_out + limiter_ms) * 1000, 2)
            report["monitor_is_estimate"] = True
        else:
            report["monitor_ms"] = None
            report["monitor_is_estimate"] = True

        # 若做過環回校正就一併回報,那才是可信的絕對值
        report["calibrated_monitor_ms"] = self.calibrated_monitor_ms

        # 麥克風送到對方耳朵(不含網路);沒開耳機時虛擬那路自己就是主時脈
        if self._mic_stream is not None and self._vc_stream is not None:
            to_peer = (mic_in + mic_ring + vc_out
                       + self.vc_limiter.latency_samples / sr)
            if self._hp_stream is not None:
                to_peer += vc_ring        # 虛擬那路多一層跨時脈緩衝
            report["mic_to_peer_ms"] = round(to_peer * 1000, 2)
        else:
            report["mic_to_peer_ms"] = None

        return report

    # ── 自動調校 ────────────────────────────────────────────────────────
    #
    # 最佳設定因機器而異:同一組參數在這台 0 xrun,在較慢的機器上可能掉幀。
    # 所以不猜,直接在實機上把候選組合各跑一遍再挑。
    #
    # 由低延遲往高延遲試,取「不掉幀的最低延遲」。刻意不取絕對最低 ——
    # 掉幀的爆音比多幾毫秒延遲難聽得多。
    #
    # 只留四個級距。實機量測顯示相鄰級距(192 / 320 / 640)之間的差異
    # 遠小於量測本身的抖動,多試它們只是讓掃描變長;而 960 在任何模式下
    # 都不曾勝出過,純粹是把延遲往上推。
    TUNE_BLOCKSIZES = (64, 128, 240, 480)

    def measure_config(self, exclusive: bool, blocksize: int,
                       settle: float = 3.0) -> dict:
        """用指定設定實際開一次串流,量延遲與 xrun。

        ``settle`` 不能太短:環形緩衝的自適應修剪約每秒收斂一次,只等一輪
        會量到還沒修剪完的數字,把表現低估好幾毫秒。3 秒能讓它跑完數輪。
        整輪掃描的時間改用「少試幾組設定」來省(見 TUNE_BLOCKSIZES),
        不是靠縮短這裡。
        """
        import time as _time

        was_running = self._running
        old_exclusive, old_blocksize = self.exclusive_mode, self.blocksize

        result: dict = {"exclusive": exclusive, "blocksize": blocksize,
                        "block_ms": round(blocksize / self.samplerate * 1000, 2)}
        try:
            self.stop()
            self.exclusive_mode = exclusive
            self.set_blocksize(blocksize)
            self.xruns = 0
            self.start()

            # 等緩衝與自適應修剪穩定下來,否則量到的是暖機中的數字
            _time.sleep(settle)

            report = self.latency_report()
            result.update({
                "ok": True,
                "monitor_ms": report.get("monitor_ms"),
                "mic_to_peer_ms": report.get("mic_to_peer_ms"),
                "xruns": self.xruns,
                "cpu": round(self._cpu_load, 3),
                "exclusive_active": list(self.exclusive_active),
            })
        except Exception as exc:
            result.update({"ok": False, "error": str(exc)[:160]})
        finally:
            try:
                self.stop()
            except Exception:
                pass
            self.exclusive_mode = old_exclusive
            self.set_blocksize(old_blocksize)
            if was_running:
                try:
                    self.start()
                except Exception:
                    pass
        return result

    def tune_latency(self, target_ms: float = 30.0,
                     allow_exclusive: bool = True,
                     progress=None) -> dict:
        """掃描設定組合,挑出不掉幀且延遲最低的一組。

        ``allow_exclusive`` 讓呼叫端能排除獨佔模式 —— 它延遲最低,但會鎖住
        裝置,其他程式(含 Discord 自己的提示音)發不出聲音,不是每個人都要。
        """
        if self.headphone_device is None or self.mic_device is None:
            raise EngineError("自動調校需要同時選好麥克風與耳機。")

        modes = [False]
        if allow_exclusive:
            modes.insert(0, True)      # 先試獨佔,通常直接就達標

        trials: list[dict] = []
        total = len(modes) * len(self.TUNE_BLOCKSIZES)
        done = 0

        for exclusive in modes:
            for blocksize in self.TUNE_BLOCKSIZES:
                if progress:
                    label = "獨佔" if exclusive else "共享"
                    progress(f"測試 {label} / {blocksize} 取樣",
                             done / max(total, 1))
                trials.append(self.measure_config(exclusive, blocksize))
                done += 1

                # 已經明顯達標就不必把更大的 block 試完 —— 它們只會更慢
                last = trials[-1]
                if (last.get("ok") and last.get("xruns") == 0
                        and isinstance(last.get("monitor_ms"), (int, float))
                        and last["monitor_ms"] <= target_ms * 0.5):
                    done = ((modes.index(exclusive) + 1)
                            * len(self.TUNE_BLOCKSIZES))
                    break

        usable = [t for t in trials
                  if t.get("ok") and t.get("xruns") == 0
                  and isinstance(t.get("monitor_ms"), (int, float))]

        best = min(usable, key=lambda t: t["monitor_ms"]) if usable else None
        if best:
            self.exclusive_mode = best["exclusive"]
            self.set_blocksize(best["blocksize"])
            if self._running:
                self.restart()

        if progress:
            progress("完成", 1.0)

        return {
            "trials": trials,
            "best": best,
            "target_ms": target_ms,
            "met_target": bool(best and best["monitor_ms"] <= target_ms),
            "applied": bool(best),
        }

    def status(self) -> dict:
        return {
            "running": self._running,
            "samplerate": self.samplerate,
            "blocksize": self.blocksize,
            "latency": self.latency_report(),
            "headphone_device": self.headphone_device,
            "virtual_device": self.virtual_device,
            "mic_device": self.mic_device,
            "exclusive_mode": self.exclusive_mode,
            "exclusive_active": list(self.exclusive_active),
            "exclusive_notes": list(self.exclusive_notes),
            "format_notes": list(self.format_notes),
            **self.hostapi_report(),
            "xruns": self.xruns,
            "cpu": round(self._cpu_load, 3),
            "mic_ring": self._mic_ring.available(),
            "vc_ring": self._vc_ring.available(),
            "ring_overflows": self._mic_ring.overflows + self._vc_ring.overflows,
            "ring_underflows": self._mic_ring.underflows + self._vc_ring.underflows,
            "vc_drift_ppm": round((self._vc_ring.ratio - 1.0) * 1e6, 1),
            "last_error": self.last_error,
            "separation_mode": self.separator.mode,
            "vc_sync_ms": self._vc_sync_ms,
            "mic_denoise": self.mic_denoise.as_dict(),
            "vc_sync_max_ms": CEILING_DELAY_MS,
            "vc_sync_suggested_ms": self.suggested_vc_sync_ms(),
            "music_pitch": self.music_pitch.as_dict(),
            "mic_echo": self.mic_echo.as_dict(),
            "params": self.params.as_dict(),
            "player": self.player.state(),
        }

    def meter_snapshot(self) -> dict:
        return self.meters.snapshot()
