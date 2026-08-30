# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller 打包設定。

    cd engine
    uv run pyinstaller ktisv_engine.spec --noconfirm

產物在 engine/dist/ktisv-engine/,裡面的 ktisv-engine.exe 就是 C# 前端要呼叫的引擎。
採 onedir(資料夾)而非 onefile:onefile 每次啟動都要把整包解壓到暫存目錄,對這種
會被前端反覆冷啟動的常駐程式來說又慢又容易被防毒攔;onedir 啟動快得多。
"""

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

datas = []
binaries = []
hiddenimports = []

# imageio-ffmpeg 內附的 ffmpeg 執行檔 —— 少了它就無法解碼任何音源。
#
# include_py_files=True 不是可有可無的:imageio_ffmpeg 找執行檔的方式是
# importlib.resources.files("imageio_ffmpeg.binaries"),那需要該子套件的
# __init__.py 真的存在。只收資料檔的話,exe 明明打包進去了卻找不到,
# 它會一路退到「系統 ffmpeg」然後失敗 —— 症狀是打包版抓得到 YouTube
# 音源、卻完全放不出聲音,而錯誤訊息只會說「找不到 ffmpeg」。
datas += collect_data_files("imageio_ffmpeg", include_py_files=True)
hiddenimports += ["imageio_ffmpeg.binaries"]

# 打包當下就確認 exe 真的在清單裡。少了它的產物是壞的,但要等到使用者
# 載入第一首歌才會發現 —— 這種錯誤值得讓建置直接失敗。
if not any(str(src).lower().endswith(".exe") and "ffmpeg" in str(src).lower()
           for src, _dst in datas):
    raise SystemExit(
        "打包中止:找不到 imageio-ffmpeg 的 ffmpeg 執行檔。\n"
        "請先在 engine 目錄執行 `uv sync`,再確認\n"
        '  python -c "import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())"\n'
        "可以印出路徑。")

# sounddevice 靠 _sounddevice_data 帶著 PortAudio 的 DLL
datas += collect_data_files("sounddevice")

# yt-dlp 用大量動態 import 載入各網站的 extractor,靜態分析抓不到
hiddenimports += collect_submodules("yt_dlp")

# scipy.signal 的部分後端也是延遲載入
hiddenimports += collect_submodules("scipy.signal")

# demucs 用字串查表決定要建哪個模型(htdemucs / htdemucs_ft / mdx …),
# 靜態分析看不到那些 import。少了它們的症狀是「選了某個模型才壞」。
hiddenimports += collect_submodules("demucs")

# demucs 的預設模型清單是套件內的 .yaml,不是程式碼 —— 只收 .py 的話
# 打包版會在載入模型時說找不到 remote 檔案。
datas += collect_data_files("demucs")

block_cipher = None

a = Analysis(
    ["run_engine.py"],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    # torch / demucs 現在是內建的 —— 使用者不該為了分離人聲還得自己架
    # 一個 Python 環境。用的是 **CPU 版** torch(454 MB;CUDA 版 2753 MB),
    # 理由見 pyproject.toml。
    excludes=["tkinter", "matplotlib", "PIL"],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="ktisv-engine",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,          # 引擎靠 stdout 送握手訊息,必須保留 console
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="ktisv-engine",
)
