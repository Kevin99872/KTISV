"""localhost JSON-lines 伺服器。

協定
----
每一行都是一個 UTF-8 的 JSON 物件,以 ``\\n`` 結尾。

前端 → 引擎:  ``{"id": 12, "cmd": "play", "args": {...}}``
引擎 → 前端:  ``{"id": 12, "ok": true, "result": {...}}``
              ``{"id": 12, "ok": false, "error": "訊息"}``
              ``{"event": "meters", "data": {...}}``   (無 id,主動推送)

連線後的第一個指令必須是 ``hello`` 並帶上啟動時印在 stdout 的 token,
否則連線會被關閉 —— 避免同一台機器上的其他程式亂接。
"""

from __future__ import annotations

import json
import queue
import socket
import threading
import traceback
from typing import Callable

Handler = Callable[[str, dict], object]


class _Client:
    def __init__(self, conn: socket.socket, addr) -> None:
        self.conn = conn
        self.addr = addr
        self.out: queue.Queue[str | None] = queue.Queue(maxsize=512)
        self.authenticated = False
        self.alive = True

    def send(self, payload: dict) -> None:
        if not self.alive:
            return
        try:
            self.out.put_nowait(json.dumps(payload, ensure_ascii=False) + "\n")
        except queue.Full:
            # 前端塞車時寧可丟掉電平推送,也不要拖慢引擎
            pass

    def close(self) -> None:
        self.alive = False
        try:
            self.out.put_nowait(None)
        except queue.Full:
            pass


class IpcServer:
    def __init__(self, handler: Handler, host: str = "127.0.0.1", port: int = 0,
                 token: str = "", log: Callable[[str], None] | None = None) -> None:
        self.handler = handler
        self.host = host
        self.port = port
        self.token = token
        self._log = log or (lambda msg: None)
        self._sock: socket.socket | None = None
        self._clients: list[_Client] = []
        self._clients_lock = threading.Lock()
        self._stop = threading.Event()
        self._accept_thread: threading.Thread | None = None
        self.on_client_change: Callable[[int], None] | None = None

    # ── 生命週期 ────────────────────────────────────────────────────────
    def start(self) -> int:
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind((self.host, self.port))
        self._sock.listen(4)
        self.port = self._sock.getsockname()[1]
        self._accept_thread = threading.Thread(target=self._accept_loop,
                                               name="ktisv-ipc-accept", daemon=True)
        self._accept_thread.start()
        return self.port

    def stop(self) -> None:
        self._stop.set()
        if self._sock is not None:
            try:
                self._sock.close()
            except Exception:
                pass
        with self._clients_lock:
            clients = list(self._clients)
        for client in clients:
            client.close()

    @property
    def client_count(self) -> int:
        with self._clients_lock:
            return len(self._clients)

    # ── 推送 ────────────────────────────────────────────────────────────
    def broadcast(self, event: str, data: object = None) -> None:
        payload = {"event": event, "data": data}
        with self._clients_lock:
            clients = [c for c in self._clients if c.authenticated]
        for client in clients:
            client.send(payload)

    # ── 內部 ────────────────────────────────────────────────────────────
    def _accept_loop(self) -> None:
        assert self._sock is not None
        while not self._stop.is_set():
            try:
                conn, addr = self._sock.accept()
            except OSError:
                break
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            client = _Client(conn, addr)
            with self._clients_lock:
                self._clients.append(client)
            threading.Thread(target=self._reader, args=(client,),
                             name="ktisv-ipc-read", daemon=True).start()
            threading.Thread(target=self._writer, args=(client,),
                             name="ktisv-ipc-write", daemon=True).start()
            self._notify_change()

    def _notify_change(self) -> None:
        if self.on_client_change:
            try:
                self.on_client_change(self.client_count)
            except Exception:
                pass

    def _writer(self, client: _Client) -> None:
        try:
            while True:
                item = client.out.get()
                if item is None:
                    break
                client.conn.sendall(item.encode("utf-8"))
        except OSError:
            pass
        finally:
            client.alive = False
            try:
                client.conn.close()
            except Exception:
                pass

    def _reader(self, client: _Client) -> None:
        buffer = b""
        try:
            while client.alive:
                chunk = client.conn.recv(65536)
                if not chunk:
                    break
                buffer += chunk
                while b"\n" in buffer:
                    line, buffer = buffer.split(b"\n", 1)
                    if line.strip():
                        self._dispatch(client, line)
        except OSError:
            pass
        finally:
            client.close()
            with self._clients_lock:
                if client in self._clients:
                    self._clients.remove(client)
            self._notify_change()

    def _dispatch(self, client: _Client, line: bytes) -> None:
        try:
            message = json.loads(line.decode("utf-8"))
        except Exception as exc:
            client.send({"id": None, "ok": False, "error": f"JSON 解析失敗: {exc}"})
            return

        msg_id = message.get("id")
        cmd = message.get("cmd") or ""
        args = message.get("args") or {}

        if not client.authenticated:
            if cmd != "hello":
                client.send({"id": msg_id, "ok": False, "error": "尚未認證"})
                client.close()
                return
            if self.token and args.get("token") != self.token:
                client.send({"id": msg_id, "ok": False, "error": "token 不正確"})
                client.close()
                return
            client.authenticated = True
            client.send({"id": msg_id, "ok": True, "result": {"hello": "ktisv"}})
            return

        try:
            result = self.handler(cmd, args)
            client.send({"id": msg_id, "ok": True, "result": result})
        except Exception as exc:
            self._log(f"指令 {cmd} 失敗: {exc}\n{traceback.format_exc()}")
            client.send({"id": msg_id, "ok": False, "error": str(exc) or type(exc).__name__})
