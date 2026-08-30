"""IPC 端對端測試 —— 啟動真的引擎行程,用 socket 走一輪指令。

    python -m tests.test_ipc        (在 engine 目錄下執行)
"""

from __future__ import annotations

import json
import os
import re
import socket
import subprocess
import sys
import time

FAILURES: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    mark = "ok  " if condition else "FAIL"
    print(f"  [{mark}] {name}{('  — ' + detail) if detail else ''}")
    if not condition:
        FAILURES.append(name)


class Client:
    def __init__(self, port: int, token: str) -> None:
        self.sock = socket.create_connection(("127.0.0.1", port), timeout=10)
        self.sock.settimeout(20)
        self._buf = b""
        self._id = 0
        self.events: list[dict] = []
        self.call("hello", {"token": token})

    def call(self, cmd: str, args: dict | None = None) -> dict:
        self._id += 1
        msg_id = self._id
        payload = json.dumps({"id": msg_id, "cmd": cmd, "args": args or {}})
        self.sock.sendall((payload + "\n").encode("utf-8"))
        while True:
            message = self._read()
            if "event" in message:
                self.events.append(message)
                continue
            if message.get("id") != msg_id:
                continue
            if not message.get("ok"):
                raise RuntimeError(f"{cmd} 失敗: {message.get('error')}")
            return message.get("result")

    def drain(self, seconds: float) -> None:
        deadline = time.time() + seconds
        self.sock.settimeout(0.3)
        try:
            while time.time() < deadline:
                try:
                    self.events.append(self._read())
                except (socket.timeout, TimeoutError):
                    pass
        finally:
            self.sock.settimeout(20)

    def _read(self) -> dict:
        while b"\n" not in self._buf:
            chunk = self.sock.recv(65536)
            if not chunk:
                raise ConnectionError("引擎關閉了連線")
            self._buf += chunk
        line, self._buf = self._buf.split(b"\n", 1)
        return json.loads(line.decode("utf-8"))

    def close(self) -> None:
        try:
            self.sock.close()
        except Exception:
            pass


def main() -> int:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # --packaged 改測 dist 裡打包好的 exe,而不是原始碼模組
    if "--packaged" in sys.argv:
        exe = os.path.join(root, "dist", "ktisv-engine", "ktisv-engine.exe")
        if not os.path.isfile(exe):
            print(f"找不到打包產物:{exe}\n請先執行 pyinstaller ktisv_engine.spec")
            return 1
        cmd = [exe]
        print(f"(測試打包版:{exe})")
    else:
        cmd = [sys.executable, "-m", "ktisv_engine"]

    proc = subprocess.Popen(
        cmd, cwd=root, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, encoding="utf-8", bufsize=1,
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
    )

    print("啟動引擎行程")
    handshake = ""
    deadline = time.time() + 30
    while time.time() < deadline:
        line = proc.stdout.readline()
        if not line:
            break
        if line.startswith("KTISV_ENGINE"):
            handshake = line.strip()
            break

    match = re.match(r"KTISV_ENGINE port=(\d+) token=(\w+)", handshake)
    check("握手訊息格式正確", match is not None, handshake or "(沒有輸出)")
    if match is None:
        proc.kill()
        print(proc.stderr.read())
        return 1

    port, token = int(match.group(1)), match.group(2)
    print(f"  引擎在 127.0.0.1:{port}\n")

    client = None
    try:
        # --- 認證 ---
        print("認證")
        bad = socket.create_connection(("127.0.0.1", port), timeout=5)
        bad.sendall(b'{"id":1,"cmd":"hello","args":{"token":"wrong"}}\n')
        reply = json.loads(bad.recv(4096).decode("utf-8").split("\n")[0])
        check("錯誤的 token 被拒絕", reply.get("ok") is False, reply.get("error", ""))
        bad.close()

        skipped = socket.create_connection(("127.0.0.1", port), timeout=5)
        skipped.sendall(b'{"id":1,"cmd":"get_status"}\n')
        reply = json.loads(skipped.recv(4096).decode("utf-8").split("\n")[0])
        check("未認證就下指令會被擋", reply.get("ok") is False, reply.get("error", ""))
        skipped.close()

        client = Client(port, token)
        check("正確的 token 可連線", True)
        print()

        # --- 基本查詢 ---
        print("基本查詢")
        pong = client.call("ping")
        check("ping 有回應", "pong" in pong)

        status = client.call("get_status")
        check("狀態含必要欄位",
              {"running", "params", "player", "music_eq", "mic_eq"} <= set(status))
        check("ffmpeg 可用", status["ffmpeg"] is True)

        devices = client.call("get_devices")
        check("列出輸入裝置", len(devices["inputs"]) > 0, f"{len(devices['inputs'])} 個")
        check("列出輸出裝置", len(devices["outputs"]) > 0, f"{len(devices['outputs'])} 個")
        print()

        # --- 混音參數 ---
        print("混音參數")
        result = client.call("set_gain", {"name": "master_hp", "db": -6.0})
        check("以 dB 設定增益", abs(result["value"] - 0.5012) < 0.01, f"{result['value']:.4f}")

        result = client.call("set_gains", {"linear": {"send_mic_vc": 0.75,
                                                      "send_music_vc": 0.5}})
        check("批次設定增益", result["send_mic_vc"] == 0.75 and result["send_music_vc"] == 0.5)

        result = client.call("set_flag", {"name": "monitor_self", "value": True})
        check("開關可設定", result["monitor_self"] is True)

        try:
            client.call("set_gain", {"name": "not_a_gain", "db": 0})
            check("未知增益會回報錯誤", False)
        except RuntimeError:
            check("未知增益會回報錯誤", True)
        print()

        # --- EQ ---
        print("等化器")
        info = client.call("eq_info")
        check("EQ 有 10 段", len(info["bands"]) == 10, str(len(info["bands"])))

        result = client.call("eq_set_band", {"target": "music", "index": 4, "gain": 6.0})
        check("設定單一頻段", result["gains"][4] == 6.0)

        gains = [3.0, -3.0, 0, 0, 0, 0, 0, 0, 4.5, -4.5]
        result = client.call("eq_set_all", {"target": "mic", "gains": gains})
        check("批次設定麥克風 EQ", result["gains"] == gains)

        result = client.call("eq_set_band", {"target": "music", "index": 0, "gain": 99.0})
        check("增益被限制在 ±15 dB", result["gains"][0] == 15.0, f"{result['gains'][0]}")

        response = client.call("eq_response", {"target": "mic", "points": 32})
        check("可取得響應曲線", len(response["db"]) == 32)

        client.call("eq_reset", {"target": "music"})
        info = client.call("eq_info")
        check("重設後歸零", all(g == 0.0 for g in info["music"]["gains"]))

        try:
            client.call("eq_set_band", {"target": "nope", "index": 0, "gain": 0})
            check("未知 EQ 目標會回報錯誤", False)
        except RuntimeError:
            check("未知 EQ 目標會回報錯誤", True)
        print()

        # --- EQ:自訂頻段 ---
        print("等化器 —— 自訂頻段")
        result = client.call("eq_add_band",
                             {"target": "music", "freq": 1500.0, "gain": 3.0, "q": 4.0})
        added = result["index"]
        check("可新增頻段", len(result["bands"]) == 11
              and result["bands"][added]["freq"] == 1500.0, str(added))
        check("新頻段帶著自己的 Q", result["bands"][added]["q"] == 4.0)

        result = client.call("eq_set_band",
                             {"target": "music", "index": added, "freq": 2500.0, "q": 1.0})
        check("可單獨改頻率與 Q", result["bands"][added]["freq"] == 2500.0
              and result["bands"][added]["q"] == 1.0)

        result = client.call("eq_remove_band", {"target": "music", "index": added})
        check("可刪除頻段", len(result["bands"]) == 10
              and all(b["freq"] != 2500.0 for b in result["bands"]))

        result = client.call("eq_set_bands", {"target": "mic", "bands": [
            {"freq": 120.0, "gain": -3.0, "q": 0.7},
            {"freq": 900.0, "gain": 2.0, "q": 2.0},
            {"freq": 6000.0, "gain": 4.0, "q": 0.7},
        ]})
        check("可整組換掉頻段", [b["freq"] for b in result["bands"]] == [120.0, 900.0, 6000.0])
        check("兩端自動成為 shelf",
              result["bands"][0]["type"] == "low_shelf"
              and result["bands"][2]["type"] == "high_shelf"
              and result["bands"][1]["type"] == "peaking")

        result = client.call("eq_set_band", {"target": "mic", "index": 1, "q": 999.0})
        check("Q 值被限制在合理範圍", result["bands"][1]["q"] <= 18.0,
              str(result["bands"][1]["q"]))

        client.call("eq_set_bands", {"target": "mic", "bands": [{"freq": 1000.0}]})
        try:
            client.call("eq_remove_band", {"target": "mic", "index": 0})
            check("不能把頻段刪光", False)
        except RuntimeError:
            check("不能把頻段刪光", True)
        print()

        # --- 變調 ---
        print("變調(升 key / 降 key)")
        status = client.call("get_status")
        check("狀態含變調設定", "music_pitch" in status
              and status["music_pitch"]["semitones"] == 0)

        result = client.call("set_music_pitch", {"semitones": 3})
        check("可升 key", result["semitones"] == 3
              and abs(result["ratio"] - 2 ** (3 / 12)) < 0.002,
              f"ratio={result['ratio']}")
        check("升 key 之後會回報延遲", result["latency_ms"] > 0,
              f"{result['latency_ms']} ms")

        result = client.call("set_music_pitch", {"semitones": -4})
        check("可降 key", result["semitones"] == -4 and result["ratio"] < 1.0)

        result = client.call("set_music_pitch", {"semitones": 99})
        check("超出範圍會被夾住", result["semitones"] == 12, str(result["semitones"]))

        result = client.call("set_music_pitch", {"semitones": 0})
        check("回原調時不佔延遲",
              result["semitones"] == 0 and result["latency_ms"] == 0)
        print()

        # --- 麥克風回音 ---
        print("麥克風回音")
        status = client.call("get_status")
        check("狀態含回音設定", "mic_echo" in status
              and status["mic_echo"]["enabled"] is False)

        result = client.call("set_mic_echo", {"enabled": True, "delay_ms": 250.0,
                                              "feedback": 0.4, "mix": 0.3,
                                              "damping": 0.5})
        check("可開啟並設定回音",
              result["enabled"] is True and result["delay_ms"] == 250.0
              and result["feedback"] == 0.4 and result["mix"] == 0.3)

        result = client.call("set_mic_echo", {"mix": 0.6})
        check("沒帶的欄位不會被重設",
              result["mix"] == 0.6 and result["delay_ms"] == 250.0
              and result["enabled"] is True)

        result = client.call("set_mic_echo", {"feedback": 5.0, "delay_ms": 9999.0})
        check("回授與時間被限制在安全範圍",
              result["feedback"] <= 0.9 and result["delay_ms"] <= 1000.0,
              f"fb={result['feedback']} delay={result['delay_ms']}")

        result = client.call("set_mic_echo", {"enabled": False})
        check("可關閉回音", result["enabled"] is False)
        print()

        # --- 分離 ---
        print("分離設定")
        result = client.call("set_separation", {"remove_vocals": True})
        check("勾去人聲 → 即時模式切到 remove_vocals",
              result["mode"] == "remove_vocals", result["mode"])

        result = client.call("set_separation", {"remove_instrumental": True})
        check("兩個都勾 → silence", result["mode"] == "silence", result["mode"])

        result = client.call("set_separation", {"remove_vocals": False,
                                               "remove_instrumental": False})
        check("都不勾 → off", result["mode"] == "off", result["mode"])

        result = client.call("set_separator_params", {"strength": 0.8, "low_cut": 220})
        check("分離參數可調", result["strength"] == 0.8 and result["low_cut"] == 220.0)

        # --- 即時分離的演算法切換 ---
        check("預設用頻譜域演算法", result["quality"] == "quality", result["quality"])
        check("回報延遲", result["latency_ms"] > 0, f"{result['latency_ms']} ms")

        fast = client.call("set_separator_quality", {"quality": "fast"})
        check("可切到時域(零延遲)",
              fast["quality"] == "fast" and fast["latency_ms"] == 0,
              f"{fast['latency_ms']} ms")
        check("切換時參數不會遺失",
              fast["strength"] == 0.8 and fast["low_cut"] == 220.0,
              f"strength={fast['strength']} low_cut={fast['low_cut']}")

        back = client.call("set_separator_quality", {"quality": "quality"})
        check("可切回頻譜域", back["quality"] == "quality")
        check("切回後參數仍在",
              back["strength"] == 0.8 and back["low_cut"] == 220.0)
        check("頻譜域才有銳利度參數", back.get("sharpness") is not None)

        try:
            client.call("set_separator_quality", {"quality": "bogus"})
            check("未知演算法會被拒絕", False)
        except RuntimeError:
            check("未知演算法會被拒絕", True)

        # --- Demucs 推論參數(UVR 也是調這兩個)---
        result = client.call("set_demucs_params", {"shifts": 2, "overlap": 0.5})
        check("Demucs 參數可設定",
              result["shifts"] == 2 and abs(result["overlap"] - 0.5) < 1e-9,
              str(result))
        clamped = client.call("set_demucs_params", {"shifts": 99, "overlap": 5.0})
        check("Demucs 參數會被夾在合理範圍",
              clamped["shifts"] <= 10 and clamped["overlap"] <= 0.9,
              str(clamped))
        client.call("set_demucs_params", {"shifts": 0, "overlap": 0.25})

        result = client.call("set_separation_engine", {"mode": "demucs"})
        check("可切換到 demucs", result["mode"] == "demucs")
        client.call("set_separation_engine", {"mode": "realtime"})

        try:
            client.call("set_separation_engine", {"mode": "magic"})
            check("未知引擎會回報錯誤", False)
        except RuntimeError:
            check("未知引擎會回報錯誤", True)

        probe = client.call("demucs_probe", {})
        check("可偵測 Demucs 環境", "available" in probe,
              "可用" if probe.get("available") else f"未安裝({probe.get('reason', '')[:40]})")
        print()

        # --- 搜尋(非同步,結果走事件)---
        print("搜尋")
        started = client.call("search", {"query": "lofi hip hop", "limit": 5})
        check("搜尋指令立即回應", started.get("started") is True)

        client.events.clear()
        client.drain(20.0)
        search_events = [e for e in client.events
                         if e.get("event") in ("search_results", "search_failed")]
        check("有收到搜尋結果事件", len(search_events) > 0,
              str([e["event"] for e in search_events]))
        if search_events and search_events[-1]["event"] == "search_results":
            results = search_events[-1]["data"]["results"]
            check("搜尋結果非空", len(results) > 0, f"{len(results)} 首")
            if results:
                first = results[0]
                check("結果含必要欄位",
                      {"url", "title", "duration_text"} <= set(first),
                      str(sorted(first)))
                check("結果網址是 YouTube 連結",
                      first["url"].startswith("http"), first["url"][:40])
        else:
            detail = search_events[-1]["data"].get("error", "") if search_events else "無事件"
            check("搜尋結果非空", False, f"搜尋失敗: {detail}")

        try:
            client.call("search", {"query": ""})
            check("空關鍵字會被擋", False)
        except RuntimeError:
            check("空關鍵字會被擋", True)
        print()

        # --- 傳輸(沒載入音源時要安全) ---
        print("傳輸控制")
        state = client.call("transport", {"action": "play"})
        check("未載入時播放不會爆掉", state["playing"] is False)
        state = client.call("seek", {"position": 10.0})
        check("未載入時跳轉不會爆掉", state["position"] == 0.0)
        state = client.call("set_loop", {"value": True})
        check("循環可設定", state["loop"] is True)
        print()

        # --- 事件推送 ---
        print("事件推送")
        client.events.clear()          # 前面的搜尋 drain 累積了大量事件,先清乾淨
        client.drain(1.5)
        kinds = {e.get("event") for e in client.events}
        check("有推送電平表", "meters" in kinds, str(sorted(kinds)))
        check("有推送播放狀態", "player" in kinds)
        meter_events = [e for e in client.events if e.get("event") == "meters"]
        check("電平推送頻率合理", 20 <= len(meter_events) <= 60,
              f"1.5 秒收到 {len(meter_events)} 筆")
        if meter_events:
            sample = meter_events[-1]["data"]
            check("電平資料含所有量測點",
                  {"music_in", "music_out", "mic_in", "mic_out", "hp_out", "vc_out"}
                  <= set(sample))
            check("電平欄位齊全", {"peak", "rms", "hold", "clip"} <= set(sample["hp_out"]))
        print()

        # --- 設定檔 ---
        print("設定檔")
        saved = client.call("save_settings", {})
        check("可存檔", os.path.isfile(saved["path"]), saved["path"])
        loaded = client.call("load_settings", {})
        check("可讀回", "params" in loaded and "music_eq" in loaded)
        print()

        # --- 錯誤處理 ---
        print("錯誤處理")
        try:
            client.call("no_such_command")
            check("未知指令會回報錯誤", False)
        except RuntimeError as exc:
            check("未知指令會回報錯誤", "未知的指令" in str(exc))

        try:
            client.call("load", {"source": ""})
            check("空音源會被擋下", False)
        except RuntimeError:
            check("空音源會被擋下", True)

        client.sock.sendall(b"{ this is not json }\n")
        check("壞掉的 JSON 不會弄死引擎", client.call("ping") is not None)
        print()

    finally:
        if client is not None:
            client.close()
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()

    if FAILURES:
        print(f"{len(FAILURES)} 項未通過: " + ", ".join(FAILURES))
        return 1
    print("全部通過。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
