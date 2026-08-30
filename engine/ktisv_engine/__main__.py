"""引擎進入點。

啟動後會在 stdout 印出一行讓前端解析的握手資訊:

    KTISV_ENGINE port=52341 token=... version=0.1.0

之後 stdout 只會有除錯訊息,正式的通訊都走 TCP。
"""

from __future__ import annotations

import argparse
import signal
import sys

from . import __version__
from .session import Session


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ktisv_engine", description="KTISV 音訊引擎")
    parser.add_argument("--port", type=int, default=0, help="TCP 埠(0 = 自動)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--selftest", action="store_true",
                        help="不啟動伺服器,只檢查環境")
    args = parser.parse_args(argv)

    if args.selftest:
        return _selftest()

    _prepare_realtime()

    session = Session(args.host, args.port)
    port = session.start()

    print(f"KTISV_ENGINE port={port} token={session.token} version={__version__}",
          flush=True)

    def _bye(*_):
        session.shutdown()

    signal.signal(signal.SIGINT, _bye)
    try:
        signal.signal(signal.SIGTERM, _bye)
    except (AttributeError, ValueError):
        pass

    try:
        session.wait()
    except KeyboardInterrupt:
        session.shutdown()
    return 0


def _prepare_realtime() -> None:
    """讓這個行程扛得住小 block。

    block 降到 128 取樣後,音訊回呼每 2.7 ms 就得跑完一次。兩件事會讓它
    來不及,而兩件事都不在音訊程式碼裡:

    * **GIL** —— 直譯器預設讓一條執行緒最多霸佔 5 ms 才換手。那比整個回呼
      預算還長,IPC 或下載執行緒隨便一次都足以讓回呼遲到。改成 1 ms。
    * **行程優先權** —— 一般優先權的行程會被前景視窗搶先。提一級就好;
      再高(HIGH)會反過來卡住介面。

    兩者都是盡力而為,失敗就照舊跑,不值得為此讓引擎起不來。
    """
    sys.setswitchinterval(0.001)

    if sys.platform != "win32":
        return
    try:
        import ctypes

        ABOVE_NORMAL_PRIORITY_CLASS = 0x00008000
        kernel32 = ctypes.windll.kernel32
        kernel32.SetPriorityClass(kernel32.GetCurrentProcess(),
                                  ABOVE_NORMAL_PRIORITY_CLASS)
    except Exception:
        pass


def _selftest() -> int:
    ok = True
    print(f"KTISV 引擎 {__version__}  ·  Python {sys.version.split()[0]}")

    for module in ("numpy", "scipy", "sounddevice"):
        try:
            __import__(module)
            print(f"  [ok]   {module}")
        except Exception as exc:
            ok = False
            print(f"  [缺少] {module}: {exc}")

    try:
        import yt_dlp  # noqa: F401
        print("  [ok]   yt-dlp")
    except Exception:
        ok = False
        print("  [缺少] yt-dlp(無法抓 YouTube 音源)")

    from .media import ffmpeg as ffmpeg_mod
    try:
        print(f"  [ok]   ffmpeg: {ffmpeg_mod.ffmpeg_path()}")
    except Exception as exc:
        ok = False
        print(f"  [缺少] ffmpeg: {exc}")

    try:
        from .audio import devices
        info = devices.list_devices()
        print(f"  [ok]   音訊裝置: 輸入 {len(info['inputs'])} / 輸出 {len(info['outputs'])}")
        virtual = devices.suggest_virtual_output(info)
        if virtual is None:
            print("  [注意] 找不到虛擬音效卡。要送音訊給 Discord 需要安裝 "
                  "VB-CABLE 或 VoiceMeeter。")
        else:
            print(f"  [ok]   建議的虛擬輸出: {devices.describe(virtual)}")
    except Exception as exc:
        ok = False
        print(f"  [錯誤] 音訊裝置列舉失敗: {exc}")

    from .media import separator
    probe = separator.probe()
    if probe.get("available"):
        print(f"  [ok]   Demucs: torch {probe['torch']} / demucs {probe['demucs']}"
              f" / CUDA {probe['cuda']}")
    else:
        print(f"  [選用] Demucs 未就緒({probe.get('reason')}) —— 即時分離仍可使用")

    print("\n環境檢查" + ("通過。" if ok else "有缺少項目,請見上方。"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
