"""KTISV 人聲分離的 TensorFlow 訓練管線。

與 ``ktisv_research/`` 的 PyTorch 版並存,不取代它。兩者刻意輸出**同一份
ONNX 契約**(輸入 ``mixture``、輸出 ``vocals`` / ``accompaniment``,固定片段
長度),所以 C# 端不需要知道權重是哪個框架訓出來的。

為什麼要另一個環境
------------------
TensorFlow 2.9 只支援 Python 3.7–3.10,而 ``research/.venv`` 是 3.12。
所以 TF 這條路住在 ``research/.venv-tf``(Python 3.10),兩邊互不干擾。
建立方式見 ``research/TENSORFLOW.md``。

共用的部分
----------
``ktisv_research.mixing`` 與 ``ktisv_research.metrics`` 是純 numpy 的,
兩個環境都能匯入 —— 合成混音的規則與評分指標只該有一份實作。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

__version__ = "0.1.0"

SAMPLE_RATE = 44100
"""與 ktisv_research 一致。即時引擎跑 48 kHz,換算時要重取樣。"""


# ── CUDA 11 的 DLL ──────────────────────────────────────────────────────
def enable_cuda11(verbose: bool = False) -> str | None:
    """讓 TensorFlow 2.9 找得到 CUDA 11.x 的 DLL。回傳用到的目錄。

    TF 2.9 是 Windows 上**最後幾版還有原生 GPU 支援**的 TensorFlow
    (2.11 起 Windows 只剩 CPU)。代價是它寫死要 CUDA 11.2 世代的
    ``cudart64_110.dll`` / ``cublas64_11.dll`` / ``cudnn64_8.dll`` …,
    而這台機器裝的是 CUDA 13.x —— 版號對不上,TF 就默默退回 CPU。

    重裝一套 CUDA 11.2 + cuDNN 8 到系統上是可以,但那會動到整台機器的
    環境,只為了一個虛擬環境裡的套件。這裡改用一個乾淨得多的辦法:
    **PyTorch 的 Windows cu118 wheel 自己就捆了整組 CUDA 11 執行期 DLL**,
    而且就在同一個 venv 裡。把那個目錄加進 PATH,TF 就找得到了。

    (所以 ``.venv-tf`` 裡的 torch 不是拿來訓練的,只是一包 DLL。
    真正跑 demucs 的 torch 在 ``.venv`` 那側。)

    必須在 ``import tensorflow`` **之前**呼叫 —— TF 在載入時就會去 dlopen
    這些函式庫,之後再改 PATH 已經來不及。放在這個 ``__init__`` 裡就是為了
    保證這個順序:任何 ``from ktisv_tf...`` 都會先跑到這裡。
    """
    if os.name != "nt" or os.environ.get("KTISV_TF_NO_CUDA_SHIM"):
        return None
    if "tensorflow" in sys.modules and verbose:
        print("⚠ TensorFlow 已經載入,CUDA DLL 路徑設定不會生效")

    lib = Path(sys.prefix) / "Lib" / "site-packages" / "torch" / "lib"
    if not (lib / "cudnn64_8.dll").exists():
        return None

    path = str(lib)
    if path not in os.environ.get("PATH", ""):
        os.environ["PATH"] = path + os.pathsep + os.environ.get("PATH", "")
    # PATH 是給 TF 自己的 LoadLibrary 用的;add_dll_directory 是給
    # Python 擴充模組用的。兩個都設,因為兩種載入路徑都會走到。
    try:
        os.add_dll_directory(path)
    except (OSError, AttributeError):
        pass
    if verbose:
        print(f"CUDA 11 DLL: {path}")
    return path


enable_cuda11()
