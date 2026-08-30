"""工作階段控制器 —— 把 IPC 指令接到音訊引擎上。

前端(Avalonia)只跟這一層對話,所有耗時工作(下載、解碼、Demucs)
都在背景執行緒跑,並以 ``progress`` 事件回報。
"""

from __future__ import annotations

import json
import os
import secrets
import threading
import time
import traceback

import numpy as np

from . import BLOCK_SIZE, SAMPLE_RATE, __version__
from .audio import devices as devices_mod
from .audio.engine import AudioEngine, EngineError
from .dsp.eq import DEFAULT_BANDS
from .ipc import IpcServer
from .media import ffmpeg as ffmpeg_mod
from .media import lyrics as lyrics_mod
from .media import onnx_separator as onnx_mod
from .media import separator as separator_mod
from .media import youtube
from .paths import settings_path

MODE_REALTIME = "realtime"
MODE_DEMUCS = "demucs"
# 自訓模型(ONNX)。跟 demucs 一樣是離線分離,差別在模型是使用者自己訓的。
# 實測這個架構在 CPU 上只有 9.7x 實時(tiny),進不了音訊回呼,所以只做離線。
MODE_CUSTOM = "custom"


class Session:
    def __init__(self, host: str = "127.0.0.1", port: int = 0) -> None:
        self.engine = AudioEngine(SAMPLE_RATE, BLOCK_SIZE)
        self.token = secrets.token_hex(16)
        self.server = IpcServer(self._handle, host, port, self.token, log=self._log)

        self.separation_mode = MODE_REALTIME
        self.remove_vocals = False
        self.remove_instrumental = False
        self.demucs_model = separator_mod.DEFAULT_MODEL
        self.demucs_device = "auto"
        self.demucs_shifts = 0
        self.demucs_overlap = 0.25
        self.custom_model = ""          # 空字串 = 用找到的第一個
        self.download_options = youtube.DownloadOptions()

        self.media: youtube.MediaInfo | None = None
        self._load_thread: threading.Thread | None = None
        self._cancel = threading.Event()
        self._busy = False
        self._busy_stage = ""
        self._busy_progress = 0.0

        # ── 歌詞 ────────────────────────────────────────────────────────
        # 來源只有兩個:音檔旁的 .lrc/.srt,以及影片附帶的字幕。
        self.lyrics_enabled = True
        # 空字串 = 自動挑(照 PREFERRED_LANGUAGES)。使用者選過之後就固定
        # 用他選的,換歌也沿用 —— 會看日文字幕的人下一首多半還是要日文。
        self.lyrics_language = ""
        self.lyrics: dict | None = None
        self._lyrics_thread: threading.Thread | None = None
        self._lyrics_cancel = threading.Event()

        self._tuning = False
        self._calibrating = False

        self._stop = threading.Event()
        self._meter_thread: threading.Thread | None = None
        self._meter_hz = 25.0
        self._last_player_state: dict = {}

    # ── 啟動 / 關閉 ─────────────────────────────────────────────────────
    def start(self) -> int:
        port = self.server.start()
        self._meter_thread = threading.Thread(target=self._meter_loop,
                                              name="ktisv-meters", daemon=True)
        self._meter_thread.start()
        return port

    def shutdown(self) -> None:
        self._stop.set()
        self._cancel.set()
        try:
            self.engine.stop()
        except Exception:
            pass
        self.server.stop()

    def wait(self) -> None:
        while not self._stop.is_set():
            time.sleep(0.25)

    # ── 推送迴圈 ────────────────────────────────────────────────────────
    def _meter_loop(self) -> None:
        interval = 1.0 / self._meter_hz
        tick = 0
        while not self._stop.is_set():
            time.sleep(interval)
            if self.server.client_count == 0:
                continue
            try:
                self.server.broadcast("meters", self.engine.meter_snapshot())
                tick += 1
                if tick % 5 == 0:  # 5 Hz:播放位置與引擎統計
                    state = self.engine.player.state()
                    self.server.broadcast("player", state)
                    if state.get("finished") and not self._last_player_state.get("finished"):
                        self.server.broadcast("track_finished", state)
                    self._last_player_state = state
                if tick % 25 == 0:  # 1 Hz:整體狀態
                    self.server.broadcast("status", self.status())
            except Exception:
                self._log(traceback.format_exc())

    def _emit(self, event: str, data: object = None) -> None:
        self.server.broadcast(event, data)

    def _log(self, message: str) -> None:
        self._emit("log", {"message": message, "time": time.time()})

    def _progress(self, stage: str, value: float) -> None:
        self._busy_stage = stage
        self._busy_progress = value
        self._emit("progress", {"stage": stage, "value": round(value, 4),
                                "busy": self._busy})

    # ── 狀態 ────────────────────────────────────────────────────────────
    def status(self) -> dict:
        status = self.engine.status()
        status.update({
            "version": __version__,
            "separation_engine": self.separation_mode,
            "remove_vocals": self.remove_vocals,
            "remove_instrumental": self.remove_instrumental,
            "busy": self._busy,
            "stage": self._busy_stage,
            "progress": round(self._busy_progress, 4),
            "media": self.media.to_dict() if self.media else None,
            "ffmpeg": ffmpeg_mod.have_ffmpeg(),
            "eq_bands": list(DEFAULT_BANDS),
            "music_eq": self._eq_state("music"),
            "mic_eq": self._eq_state("mic"),
            "separator": self.engine.separator_info(),
            "demucs_python": separator_mod.resolve_python(),
            "demucs_model": self.demucs_model,
            "demucs_shifts": self.demucs_shifts,
            "demucs_overlap": self.demucs_overlap,
        })
        return status

    # ── 指令分派 ────────────────────────────────────────────────────────
    def _handle(self, cmd: str, args: dict):
        handler = getattr(self, f"cmd_{cmd}", None)
        if handler is None:
            raise ValueError(f"未知的指令: {cmd}")
        return handler(args)

    # -- 基本 --
    def cmd_ping(self, args: dict) -> dict:
        return {"pong": time.time(), "version": __version__}

    def cmd_get_status(self, args: dict) -> dict:
        return self.status()

    def cmd_shutdown(self, args: dict) -> dict:
        threading.Timer(0.2, self.shutdown).start()
        return {"ok": True}

    # -- 裝置 --
    def cmd_get_devices(self, args: dict) -> dict:
        if args.get("refresh"):
            devices_mod.refresh()
        result = devices_mod.list_devices()
        result["suggested_virtual"] = devices_mod.suggest_virtual_output(result)
        return result

    def cmd_set_devices(self, args: dict) -> dict:
        engine = self.engine
        if "headphone" in args:
            engine.headphone_device = _opt_int(args["headphone"])
        if "virtual" in args:
            engine.virtual_device = _opt_int(args["virtual"])
        if "mic" in args:
            engine.mic_device = _opt_int(args["mic"])
        if "mic_channels" in args:
            engine.mic_channels = max(1, min(2, int(args["mic_channels"])))
        if engine.running and args.get("restart", True):
            engine.restart()
        return self.status()

    def cmd_set_audio_options(self, args: dict) -> dict:
        """低延遲相關設定。兩者都要重開串流才生效。"""
        engine = self.engine
        if "exclusive_mode" in args:
            engine.exclusive_mode = bool(args["exclusive_mode"])
        if "blocksize" in args:
            engine.set_blocksize(int(args["blocksize"]))
        if engine.running and args.get("restart", True):
            engine.restart()
        return self.status()

    def cmd_start_calibration(self, args: dict) -> dict:
        """音遊式對時校準。播放期間會佔用耳機,所以進度另外用事件回報。"""
        if self._calibrating:
            raise RuntimeError("已經在校準中。")
        info = self.engine.start_calibration()
        self._calibrating = True
        threading.Thread(target=self._calibration_worker,
                         name="ktisv-calibrate", daemon=True).start()
        return info

    def cmd_cancel_calibration(self, args: dict) -> dict:
        self.engine.cancel_calibration()
        return {"cancelled": True}

    def _calibration_worker(self) -> None:
        """盯著校準跑完,把進度與結果推給前端。

        校準本身在音訊回呼裡進行,這裡只負責回報 —— 所以輪詢就夠了,
        不必在回呼裡碰任何同步物件。
        """
        try:
            while True:
                state = self.engine.calibration_state()
                if not state.get("active") and not state.get("finished"):
                    return                      # 被取消了
                self._emit("calibration_progress", state)
                if state.get("finished"):
                    break
                time.sleep(0.08)

            self._emit("calibration_done", self.engine.calibration_result())
        except Exception as exc:
            self._log(traceback.format_exc())
            self._emit("calibration_failed", {"error": str(exc)[:200]})
        finally:
            self._calibrating = False

    def cmd_set_mic_denoise(self, args: dict) -> dict:
        """麥克風降噪:電源哼聲陷波 + 嘶聲閘門。即時生效。"""
        return self.engine.set_mic_denoise(**args)

    def cmd_set_vc_sync(self, args: dict) -> dict:
        """送給 Discord 那一路的對時(毫秒,可正可負)。即時生效。"""
        applied = self.engine.set_vc_sync_ms(args.get("ms", 0.0))
        return {"vc_sync_ms": applied}

    def cmd_set_music_pitch(self, args: dict) -> dict:
        """升 key / 降 key。半音為單位,0 = 原調。"""
        return self.engine.set_music_pitch(args.get("semitones", 0.0))

    def cmd_set_mic_echo(self, args: dict) -> dict:
        """麥克風回音。所有欄位都是選填,沒帶的就維持原值。"""
        return self.engine.set_mic_echo(
            enabled=None if args.get("enabled") is None else bool(args["enabled"]),
            delay_ms=None if args.get("delay_ms") is None else float(args["delay_ms"]),
            feedback=None if args.get("feedback") is None else float(args["feedback"]),
            mix=None if args.get("mix") is None else float(args["mix"]),
            damping=None if args.get("damping") is None else float(args["damping"]),
        )

    def cmd_get_latency(self, args: dict) -> dict:
        return self.engine.latency_report()

    def cmd_tune_latency(self, args: dict) -> dict:
        """自動調校延遲。掃描期間會反覆開關串流,所以放到背景執行緒。"""
        if self._tuning:
            raise RuntimeError("已經在調校中。")

        target = float(args.get("target_ms", 30.0))
        allow_exclusive = bool(args.get("allow_exclusive", True))

        self._tuning = True
        threading.Thread(
            target=self._tune_worker, args=(target, allow_exclusive),
            name="ktisv-tune", daemon=True).start()
        return {"started": True, "target_ms": target}

    def _tune_worker(self, target: float, allow_exclusive: bool) -> None:
        try:
            result = self.engine.tune_latency(
                target_ms=target,
                allow_exclusive=allow_exclusive,
                progress=lambda stage, value: self._emit(
                    "tune_progress", {"stage": stage, "value": round(value, 3)}))
            self._emit("tune_done", result)
        except Exception as exc:
            self._log(traceback.format_exc())
            self._emit("tune_failed", {"error": str(exc)[:200]})
        finally:
            self._tuning = False

    def cmd_engine_start(self, args: dict) -> dict:
        return self.engine.start()

    def cmd_engine_stop(self, args: dict) -> dict:
        return self.engine.stop()

    def cmd_engine_restart(self, args: dict) -> dict:
        return self.engine.restart()

    # -- 混音 --
    def cmd_set_gain(self, args: dict) -> dict:
        name = args["name"]
        if "db" in args:
            self.engine.set_gain_db(name, float(args["db"]))
        else:
            self.engine.set_gain_linear(name, float(args["value"]))
        return {"name": name, "value": getattr(self.engine.params, name).target}

    def cmd_set_gains(self, args: dict) -> dict:
        for name, value in (args.get("db") or {}).items():
            self.engine.set_gain_db(name, float(value))
        for name, value in (args.get("linear") or {}).items():
            self.engine.set_gain_linear(name, float(value))
        return self.engine.params.as_dict()

    def cmd_set_flag(self, args: dict) -> dict:
        self.engine.set_flag(args["name"], bool(args["value"]))
        return self.engine.params.as_dict()

    # -- 分離 --
    def cmd_set_separation(self, args: dict) -> dict:
        if "remove_vocals" in args:
            self.remove_vocals = bool(args["remove_vocals"])
        if "remove_instrumental" in args:
            self.remove_instrumental = bool(args["remove_instrumental"])
        self._apply_separation()
        return {"remove_vocals": self.remove_vocals,
                "remove_instrumental": self.remove_instrumental,
                "mode": self.engine.separator.mode}

    def cmd_set_separation_engine(self, args: dict) -> dict:
        mode = args.get("mode", MODE_REALTIME)
        if mode not in (MODE_REALTIME, MODE_DEMUCS, MODE_CUSTOM):
            raise ValueError(f"未知的分離引擎: {mode}")
        changed = mode != self.separation_mode
        self.separation_mode = mode
        if "model" in args:
            self.demucs_model = args["model"] or separator_mod.DEFAULT_MODEL
        if "device" in args:
            self.demucs_device = args["device"] or "auto"
        self._apply_separation()
        return {"mode": self.separation_mode, "reload_required":
                changed and self.media is not None}

    def cmd_set_separator_params(self, args: dict) -> dict:
        sep = self.engine.separator
        if "strength" in args:
            sep.strength = max(0.0, min(1.0, float(args["strength"])))
        if "low_cut" in args:
            sep.low_cut = float(args["low_cut"])
        if "high_cut" in args:
            sep.high_cut = float(args["high_cut"])
        if "sharpness" in args and hasattr(sep, "sharpness"):
            sep.sharpness = max(0.5, min(6.0, float(args["sharpness"])))
        return self.engine.separator_info()

    def cmd_set_separator_quality(self, args: dict) -> dict:
        """切換即時分離演算法(fast = 時域零延遲 / quality = 頻譜域)。"""
        return self.engine.set_separator_quality(args.get("quality", "quality"))

    def cmd_set_demucs_params(self, args: dict) -> dict:
        """Demucs 推論參數(UVR 暴露的也是這兩個)。

        改動後已載入的音源不會自動重跑 —— 參數已進快取鍵,重新載入才會生效。
        """
        if "shifts" in args:
            self.demucs_shifts = max(0, min(10, int(args["shifts"])))
        if "overlap" in args:
            self.demucs_overlap = max(0.05, min(0.9, float(args["overlap"])))
        return {"shifts": self.demucs_shifts, "overlap": self.demucs_overlap}

    def _apply_separation(self) -> None:
        realtime = (self.separation_mode == MODE_REALTIME)
        self.engine.apply_separation_flags(self.remove_vocals,
                                           self.remove_instrumental, realtime)

    # -- EQ --
    def _eq(self, target: str):
        if target == "music":
            return self.engine.music_eq
        if target == "mic":
            return self.engine.mic_eq
        raise ValueError(f"未知的 EQ: {target}")

    def _eq_state(self, target: str) -> dict:
        eq = self._eq(target)
        return {"target": target, "enabled": eq.enabled,
                "gains": eq.gains, "bands": eq.band_info()}

    def cmd_eq_set_band(self, args: dict) -> dict:
        """改單一頻段。gain / freq / q 都是選填,沒帶的就不動。"""
        eq = self._eq(args["target"])
        eq.set_band(
            int(args["index"]),
            gain=None if args.get("gain") is None else float(args["gain"]),
            freq=None if args.get("freq") is None else float(args["freq"]),
            q=None if args.get("q") is None else float(args["q"]),
        )
        return self._eq_state(args["target"])

    def cmd_eq_set_all(self, args: dict) -> dict:
        eq = self._eq(args["target"])
        eq.set_gains(args["gains"])
        return {"target": args["target"], "gains": eq.gains}

    def cmd_eq_set_bands(self, args: dict) -> dict:
        """整組換掉頻段配置。``bands`` 是 {freq, gain, q} 的陣列。

        注意這會重建所有濾波器狀態 —— 是設定動作,不適合拿來當推桿的回呼。
        """
        eq = self._eq(args["target"])
        eq.set_bands(args["bands"])
        return self._eq_state(args["target"])

    def cmd_eq_add_band(self, args: dict) -> dict:
        eq = self._eq(args["target"])
        index = eq.add_band(
            float(args.get("freq", 1000.0)),
            float(args.get("gain", 0.0)),
            None if args.get("q") is None else float(args["q"]),
        )
        return {**self._eq_state(args["target"]), "index": index}

    def cmd_eq_remove_band(self, args: dict) -> dict:
        eq = self._eq(args["target"])
        eq.remove_band(int(args["index"]))
        return self._eq_state(args["target"])

    def cmd_eq_enable(self, args: dict) -> dict:
        eq = self._eq(args["target"])
        eq.enabled = bool(args["enabled"])
        if not eq.enabled:
            eq.clear_state()
        return {"target": args["target"], "enabled": eq.enabled}

    def cmd_eq_reset(self, args: dict) -> dict:
        eq = self._eq(args["target"])
        eq.reset()
        return {"target": args["target"], "gains": eq.gains}

    def cmd_eq_response(self, args: dict) -> dict:
        eq = self._eq(args["target"])
        freqs = np.logspace(np.log10(20.0), np.log10(20000.0),
                            int(args.get("points", 96)))
        return {"freqs": [round(f, 2) for f in freqs.tolist()],
                "db": [round(v, 3) for v in eq.response(freqs).tolist()]}

    def cmd_eq_info(self, args: dict) -> dict:
        return {
            "bands": list(self.engine.music_eq.bands),
            "music": self._eq_state("music"),
            "mic": self._eq_state("mic"),
        }

    # -- 傳輸 --
    def cmd_transport(self, args: dict) -> dict:
        action = args.get("action", "toggle")
        player = self.engine.player
        if action == "play":
            player.play()
        elif action == "pause":
            player.pause()
        elif action == "stop":
            player.stop()
        elif action == "toggle":
            player.toggle()
        else:
            raise ValueError(f"未知的傳輸動作: {action}")
        return player.state()

    def cmd_seek(self, args: dict) -> dict:
        self.engine.player.seek(float(args["position"]))
        return self.engine.player.state()

    def cmd_set_loop(self, args: dict) -> dict:
        self.engine.player.loop = bool(args["value"])
        return self.engine.player.state()

    # -- 載入 --
    def cmd_load(self, args: dict) -> dict:
        if self._busy:
            raise RuntimeError("目前正在處理另一個音源,請先取消。")
        source = (args.get("source") or "").strip()
        if not source:
            raise ValueError("請提供 YouTube 網址或本機音訊檔路徑。")
        autoplay = bool(args.get("autoplay", True))

        self._cancel.clear()
        self._busy = True
        self._load_thread = threading.Thread(
            target=self._load_worker, args=(source, autoplay),
            name="ktisv-load", daemon=True)
        self._load_thread.start()
        return {"started": True, "source": source}

    def cmd_cancel_load(self, args: dict) -> dict:
        self._cancel.set()
        return {"cancelled": True}

    # -- 搜尋 --
    def cmd_search(self, args: dict) -> dict:
        """關鍵字搜尋。網路等待放到背景,結果以 search_results 事件送回。

        若同步等在這裡,這個連線的後續指令(推桿、勾選框)會全部排在後面,
        搜尋的一兩秒內操作會像卡住一樣。
        """
        query = (args.get("query") or "").strip()
        if not query:
            raise ValueError("請輸入要搜尋的關鍵字。")
        limit = int(args.get("limit", 12))

        threading.Thread(
            target=self._search_worker, args=(query, limit),
            name="ktisv-search", daemon=True).start()
        return {"started": True, "query": query}

    def _search_worker(self, query: str, limit: int) -> None:
        try:
            results = youtube.search(query, limit, self.download_options)
            self._emit("search_results", {
                "query": query,
                "results": [r.to_dict() for r in results],
            })
        except Exception as exc:
            self._log(traceback.format_exc())
            self._emit("search_failed", {"query": query, "error": str(exc)})

    def _load_worker(self, source: str, autoplay: bool) -> None:
        try:
            if not ffmpeg_mod.have_ffmpeg():
                raise RuntimeError(
                    "找不到 ffmpeg。請在 engine 目錄執行 "
                    "`pip install imageio-ffmpeg`,或安裝 ffmpeg 並加入 PATH。")

            if youtube.is_url(source):
                self._progress("解析網址", 0.02)
                info = youtube.download_audio(
                    source, self.download_options,
                    progress=lambda stage, v: self._progress(stage, 0.02 + v * 0.38))
            else:
                if not os.path.isfile(source):
                    raise FileNotFoundError(f"找不到檔案: {source}")
                info = youtube.local_file_info(source)
                self._progress("讀取本機檔案", 0.40)

            self._check_cancel()
            self._progress("解碼音訊", 0.45)
            audio = ffmpeg_mod.decode_to_array(info.path, SAMPLE_RATE, 2)
            if not info.duration:
                info.duration = len(audio) / SAMPLE_RATE

            self._check_cancel()

            if self.separation_mode == MODE_DEMUCS:
                stems = self._run_demucs(info)
            elif self.separation_mode == MODE_CUSTOM:
                stems = self._run_custom(info)
            else:
                self._progress("準備即時分離", 0.9)
                stems = {"mix": audio}

            self._check_cancel()
            self.engine.player.load(stems, title=info.title)
            self.media = info
            self._apply_separation()

            self._progress("就緒", 1.0)
            self._emit("loaded", {"media": info.to_dict(),
                                  "player": self.engine.player.state(),
                                  "engine": self.separation_mode})
            if autoplay and self.engine.running:
                self.engine.player.play()

            # 歌詞在背景取得 —— 播放不等它。線上字幕查詢要走網路,
            # 沒有理由讓使用者盯著進度條等。
            if self.lyrics_enabled:
                self._start_lyrics_job(info)
        except _Cancelled:
            self._emit("load_cancelled", {"source": source})
            self._progress("已取消", 0.0)
        except Exception as exc:
            self._log(traceback.format_exc())
            self._emit("load_failed", {"source": source, "error": str(exc)})
            self._progress("失敗", 0.0)
        finally:
            self._busy = False
            self._emit("progress", {"stage": self._busy_stage,
                                    "value": self._busy_progress, "busy": False})

    # ── 歌詞(背景取得,不阻擋播放)──────────────────────────────────
    def _start_lyrics_job(self, info: youtube.MediaInfo) -> None:
        """啟動背景歌詞工作。

        會取消前一首還在跑的工作 —— 使用者已經換歌了,舊的結果沒有意義,
        還可能覆蓋掉新歌的歌詞。
        """
        self._lyrics_cancel.set()
        if self._lyrics_thread and self._lyrics_thread.is_alive():
            self._lyrics_thread.join(timeout=2.0)

        self._lyrics_cancel = threading.Event()
        self.lyrics = None
        self._emit("lyrics_status", {"state": "searching", "title": info.title})

        self._lyrics_thread = threading.Thread(
            target=self._lyrics_worker, args=(info, self._lyrics_cancel),
            name="ktisv-lyrics", daemon=True)
        self._lyrics_thread.start()

    def _lyrics_worker(self, info: youtube.MediaInfo,
                       cancel: threading.Event) -> None:
        def cancelled() -> bool:
            return cancel.is_set() or self._stop.is_set()

        try:
            found = lyrics_mod.load_for_media(
                info.path, info.webpage_url, self.download_options,
                self.lyrics_language)
            if cancelled():
                return

            if not found.is_empty:
                self.lyrics = found.to_dict()
                self._emit("lyrics", self.lyrics)
            else:
                available = list(found.available)
                reason = "找不到歌詞。可以把同名的 .lrc / .srt / .ass 檔放在音檔旁邊。"
                if available:
                    reason = ("這個語言沒有字幕。可以在上面改選其他語言,"
                              "或把同名的 .lrc / .srt / .ass 檔放在音檔旁邊。")
                self._emit("lyrics_status", {
                    "state": "none",
                    "reason": reason,
                    "available": available,
                })
        except Exception as exc:
            if not cancelled():
                self._log(traceback.format_exc())
                self._emit("lyrics_status",
                           {"state": "failed", "reason": str(exc)[:200]})

    def cmd_get_lyrics(self, args: dict) -> dict:
        """取得目前的歌詞。``refresh`` 會重新抓一次。"""
        if args.get("refresh") and self.media:
            self._start_lyrics_job(self.media)
            return {"state": "searching"}
        return self.lyrics or {"lines": [], "count": 0, "source": "", "synced": False}

    def cmd_set_lyrics_options(self, args: dict) -> dict:
        """歌詞設定。改語言會立刻重新抓一次。"""
        refetch = False
        if "enabled" in args:
            self.lyrics_enabled = bool(args["enabled"])
        if "language" in args:
            language = str(args["language"] or "")
            if language != self.lyrics_language:
                self.lyrics_language = language
                refetch = True

        if refetch and self.lyrics_enabled and self.media is not None:
            self._start_lyrics_job(self.media)

        return {"enabled": self.lyrics_enabled,
                "language": self.lyrics_language}

    def _run_demucs(self, info: youtube.MediaInfo) -> dict:
        self._progress("Demucs 分離中(第一次會下載模型)", 0.5)
        stems = separator_mod.separate(
            info.path,
            samplerate=SAMPLE_RATE,
            model=self.demucs_model,
            device=self.demucs_device,
            shifts=self.demucs_shifts,
            overlap=self.demucs_overlap,
            progress=lambda stage, v: self._progress(stage, 0.5 + v * 0.45),
            cancel=self._cancel.is_set,
        )
        return {"vocals": stems.vocals, "instrumental": stems.instrumental}

    def _run_custom(self, info: youtube.MediaInfo) -> dict:
        """用自訓的 ONNX 模型分離。"""
        model = self.custom_model
        if not model:
            found = onnx_mod.list_models()
            if not found:
                raise RuntimeError(
                    "沒有可用的自訓模型。把訓練匯出的 .onnx 放進 "
                    f"{onnx_mod.models_dir()} 就會出現在選單裡。")
            model = found[0]["path"]

        self._progress("自訓模型分離中", 0.5)
        stems = onnx_mod.separate(
            info.path,
            model_path=model,
            samplerate=SAMPLE_RATE,
            progress=lambda stage, v: self._progress(stage, 0.5 + v * 0.45),
            cancel=self._cancel.is_set,
        )
        return {"vocals": stems.vocals, "instrumental": stems.instrumental}

    # -- 自訓模型 --
    def cmd_list_custom_models(self, args: dict) -> dict:
        ok, why = onnx_mod.available()
        return {"available": ok, "reason": why,
                "dir": onnx_mod.models_dir(),
                "models": onnx_mod.list_models(),
                "selected": self.custom_model}

    def cmd_set_custom_model(self, args: dict) -> dict:
        path = (args.get("path") or "").strip()
        if path and not os.path.isfile(path):
            raise FileNotFoundError(f"找不到模型:{path}")
        self.custom_model = path
        return {"selected": self.custom_model}

    def _check_cancel(self) -> None:
        if self._cancel.is_set():
            raise _Cancelled()

    # -- Demucs 設定 --
    def cmd_demucs_probe(self, args: dict) -> dict:
        return separator_mod.probe(args.get("python") or None)

    def cmd_set_demucs_python(self, args: dict) -> dict:
        separator_mod.set_python(args.get("path") or "")
        return separator_mod.probe()

    # -- 下載設定 --
    def cmd_set_download_options(self, args: dict) -> dict:
        if "cookies_from_browser" in args:
            self.download_options.cookies_from_browser = args["cookies_from_browser"] or ""
        if "proxy" in args:
            self.download_options.proxy = args["proxy"] or ""
        return {"cookies_from_browser": self.download_options.cookies_from_browser,
                "proxy": self.download_options.proxy}

    # -- 設定檔 --
    def cmd_save_settings(self, args: dict) -> dict:
        data = args.get("data")
        if data is None:
            data = self._snapshot_settings()
        path = settings_path()
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=2)
        return {"path": path}

    def cmd_load_settings(self, args: dict) -> dict:
        path = settings_path()
        if not os.path.isfile(path):
            return {}
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)

    def _snapshot_settings(self) -> dict:
        engine = self.engine
        return {
            "devices": {
                "headphone": engine.headphone_device,
                "virtual": engine.virtual_device,
                "mic": engine.mic_device,
            },
            "params": engine.params.as_dict(),
            "vc_sync_ms": engine.vc_sync_ms,
            "music_pitch": engine.music_pitch.semitones,
            "mic_echo": engine.mic_echo.as_dict(),
            # 存整組頻段而不只是增益 —— 頻率與 Q 現在也是使用者調的東西,
            # 只存增益的話,下次讀回來會套到完全不同的頻率上。
            "music_eq": engine.music_eq.band_info(),
            "mic_eq": engine.mic_eq.band_info(),
            "separation_engine": self.separation_mode,
            "remove_vocals": self.remove_vocals,
            "remove_instrumental": self.remove_instrumental,
        }


class _Cancelled(Exception):
    pass


def _opt_int(value) -> int | None:
    if value is None or value == "" or (isinstance(value, int) and value < 0):
        return None
    return int(value)
