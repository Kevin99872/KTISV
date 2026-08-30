"""分離品質指標。

沒有指標就沒有「優化」可言 —— 這是整套訓練流程的地基。

四個層次的指標,各有用途
--------------------------
``si_sdr``   尺度不變 SDR。對整體音量無感,只看波形形狀對不對。
             最穩健、最快,適合訓練時的 loss 監控。

``usdr``     MDX 挑戰賽用的全曲 SDR(utterance-level)。整首算一個值。
             會被安靜段落主導,但業界排行榜用它,方便跟別人比。

``csdr``     切成固定長度的塊分別算 SDR 再取中位數(chunk-level)。
             中位數對「某一段爆掉」不敏感,比 uSDR 更能反映聽感。

``bss_eval`` 完整的 BSS Eval:SDR / SIR / SAR / ISR 四個分量。
             能區分「殘留其他音源(SIR 低)」和「產生假訊號(SAR 低)」,
             這對判斷模型是「沒分乾淨」還是「分過頭了」很關鍵。
             需要 museval,計算慢很多。

慣例
----
所有函式吃 ``(samples, channels)`` 或 ``(samples,)`` 的 float 陣列。
回傳 dB。數值越高越好。完美重建回傳 ``inf``。
"""

from __future__ import annotations

import warnings

import numpy as np

EPS = 1e-10
"""避免 log(0)。取這個量級是因為 float32 音訊的有效精度大約到 1e-7。"""


def _as_2d(x: np.ndarray) -> np.ndarray:
    """統一成 (samples, channels)。"""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        return x.reshape(-1, 1)
    if x.ndim != 2:
        raise ValueError(f"預期 1D 或 2D 陣列,收到 shape={x.shape}")
    # (channels, samples) 誤放時自動轉置 —— 聲道數不可能比取樣數多
    if x.shape[0] < x.shape[1] and x.shape[0] <= 8:
        return x.T
    return x


def _align(reference: np.ndarray, estimate: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """對齊長度與聲道數。"""
    ref = _as_2d(reference)
    est = _as_2d(estimate)

    n = min(len(ref), len(est))
    ref, est = ref[:n], est[:n]

    if ref.shape[1] != est.shape[1]:
        # 單聲道對立體聲時,把單聲道複製成雙聲道
        if ref.shape[1] == 1:
            ref = np.repeat(ref, est.shape[1], axis=1)
        elif est.shape[1] == 1:
            est = np.repeat(est, ref.shape[1], axis=1)
        else:
            channels = min(ref.shape[1], est.shape[1])
            ref, est = ref[:, :channels], est[:, :channels]
    return ref, est


def si_sdr(reference: np.ndarray, estimate: np.ndarray) -> float:
    """尺度不變 SDR(dB)。

    先把估計值投影到參考訊號上求最佳縮放,再算殘差。因此把整首歌
    放大兩倍不會影響分數 —— 這正是我們要的,音量不是分離品質。
    """
    ref, est = _align(reference, estimate)

    ref_energy = float(np.sum(ref ** 2))
    if ref_energy < EPS:
        # 參考訊號是靜音(例如純伴奏曲的人聲軌),無法定義 SI-SDR
        return float("nan")

    # 最佳縮放係數 alpha = <est, ref> / ||ref||²
    alpha = float(np.sum(est * ref)) / ref_energy
    target = alpha * ref
    noise = est - target

    target_energy = float(np.sum(target ** 2))
    noise_energy = float(np.sum(noise ** 2))

    if noise_energy < EPS:
        return float("inf")
    return 10.0 * np.log10(max(target_energy, EPS) / noise_energy)


def usdr(reference: np.ndarray, estimate: np.ndarray) -> float:
    """全曲 SDR(dB)—— MDX 挑戰賽的 utterance-level 定義。

    不做尺度對齊,直接比波形。所以音量錯了會被扣分,這是刻意的:
    送去混音的音軌音量本來就該對。
    """
    ref, est = _align(reference, estimate)

    ref_energy = float(np.sum(ref ** 2))
    if ref_energy < EPS:
        return float("nan")

    error_energy = float(np.sum((ref - est) ** 2))
    if error_energy < EPS:
        return float("inf")
    return 10.0 * np.log10(ref_energy / error_energy)


def csdr(reference: np.ndarray, estimate: np.ndarray,
         samplerate: int = 44100, chunk_seconds: float = 1.0,
         skip_silent: bool = True) -> float:
    """分塊 SDR 的中位數(dB)。

    整首算一個 SDR 時,幾秒的嚴重失真會被幾分鐘的正常段落稀釋掉。
    切塊取中位數能反映「大部分時間聽起來如何」,和聽感比較接近。

    ``skip_silent`` 會跳過參考訊號幾乎無聲的塊 —— 那些塊的 SDR 沒有意義,
    卻會把中位數整個拉歪(前奏、間奏的人聲軌就是這種)。
    """
    ref, est = _align(reference, estimate)
    chunk = max(1, int(chunk_seconds * samplerate))

    scores: list[float] = []
    for start in range(0, len(ref) - chunk + 1, chunk):
        ref_chunk = ref[start:start + chunk]
        if skip_silent and float(np.sum(ref_chunk ** 2)) < EPS:
            continue
        score = usdr(ref_chunk, est[start:start + chunk])
        # 只排除 NaN(靜音參考,無定義)。inf 代表該區塊完美重建,
        # 那是最好的分數而不是錯誤 —— 濾掉它會把中位數的意義整個反轉。
        if not np.isnan(score):
            scores.append(score)

    if not scores:
        return float("nan")
    return float(np.median(scores))


def bss_eval(references: dict[str, np.ndarray],
             estimates: dict[str, np.ndarray],
             samplerate: int = 44100) -> dict[str, dict]:
    """BSS Eval 分量:每個音源的 SDR / SIR / SAR(dB)。

    **必須同時傳入所有音源**(至少人聲 + 伴奏),因為「干擾」的定義就是
    「其他音源洩漏進來的量」—— 只給一條音軌時 SIR 在數學上沒有定義。

    這三個值能拆開兩種完全不同的失敗模式,是 uSDR 這種單一數字看不出來的:

      * **SIR 低** = 其他音源沒濾乾淨(人聲軌裡還聽得到鼓)
      * **SAR 低** = 模型自己造出了原本不存在的東西(水聲、金屬感的假訊號)

    同樣是 10 dB 的 SDR,SIR 低和 SAR 低要用完全相反的方法修 —— 前者要
    模型更「敢切」,後者要它更保守。所以微調時一定要看這兩個分量。

    用 ``fast_bss_eval`` 實作(純 Python,不需要 ffmpeg)。

    範例::

        bss_eval({"vocals": v_ref, "accompaniment": a_ref},
                 {"vocals": v_est, "accompaniment": a_est})
        # → {"vocals": {"sdr": ..., "sir": ..., "sar": ...}, ...}
    """
    try:
        import fast_bss_eval
    except ImportError as exc:
        raise ImportError(
            "bss_eval 需要 fast-bss-eval。請執行:uv sync --extra eval"
        ) from exc

    names = [n for n in references if n in estimates]
    if len(names) < 2:
        raise ValueError(
            "BSS Eval 需要至少兩個音源才能定義干擾(SIR)。"
            f"目前只有 {names or '無'} —— 請一併傳入伴奏軌。"
            "若只想評估單一音軌,請改用 si_sdr / usdr / csdr。")

    # 統一長度並轉成單聲道:BSS Eval 的濾波器設計是針對單通道來源,
    # 立體聲的左右道分別評估再平均,對「分離品質」的判斷沒有增益。
    def mono(x: np.ndarray) -> np.ndarray:
        arr = _as_2d(x)
        return arr.mean(axis=1)

    length = min(min(len(_as_2d(references[n])), len(_as_2d(estimates[n])))
                 for n in names)
    ref_mat = np.stack([mono(references[n])[:length] for n in names])
    est_mat = np.stack([mono(estimates[n])[:length] for n in names])

    # 全靜音的音源會讓濾波器求解退化
    active = [i for i, n in enumerate(names)
              if float(np.sum(ref_mat[i] ** 2)) > EPS]
    if len(active) < 2:
        return {n: {"sdr": float("nan"), "sir": float("nan"),
                    "sar": float("nan")} for n in names}

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # 注意:fast_bss_eval 0.1.x 的 compute_permutation=False 有 bug
        # (內部矩陣維度對不上,任何輸入都會炸)。只能用 True,再自己確認
        # 它沒有把音源重新配對 —— 我們的估計值本來就已經對齊了。
        sdr, sir, sar, perm = fast_bss_eval.bss_eval_sources(
            ref_mat[active], est_mat[active], compute_permutation=True)

    perm = np.asarray(perm).ravel()
    if not np.array_equal(perm, np.arange(len(active))):
        # 模型把音源對調了(人聲軌其實裝著伴奏)。這本身就是重要資訊,
        # 但分數對應的配對已非我們指定的,直接照原樣回報會誤導。
        warnings.warn(
            f"BSS Eval 重新配對了音源(perm={perm.tolist()}),"
            "代表估計的音軌與參考音軌對不上。分數已依配對後的順序回報。",
            RuntimeWarning, stacklevel=2)

    result = {n: {"sdr": float("nan"), "sir": float("nan"), "sar": float("nan")}
              for n in names}
    for slot, index in enumerate(active):
        result[names[index]] = {
            "sdr": float(sdr[slot]),
            "sir": float(sir[slot]),
            "sar": float(sar[slot]),
        }
    return result


def evaluate_stem(reference: np.ndarray, estimate: np.ndarray,
                  samplerate: int = 44100) -> dict:
    """單一音軌的快速指標。

    這裡**不含** SIR/SAR —— 那需要同時知道其他音源,請用 :func:`evaluate_track`。
    """
    return {
        "si_sdr": si_sdr(reference, estimate),
        "usdr": usdr(reference, estimate),
        "csdr": csdr(reference, estimate, samplerate),
    }


def evaluate_track(references: dict[str, np.ndarray],
                   estimates: dict[str, np.ndarray],
                   samplerate: int = 44100, full: bool = False) -> dict[str, dict]:
    """一首歌的完整評估:每個音源各自的指標。

    ``full=True`` 會額外算 SIR/SAR(較慢,但能區分失敗模式)。
    """
    result = {
        name: evaluate_stem(references[name], estimates[name], samplerate)
        for name in references if name in estimates
    }

    if full and len(result) >= 2:
        try:
            for name, scores in bss_eval(references, estimates, samplerate).items():
                if name in result:
                    result[name].update(scores)
        except (ImportError, ValueError) as exc:
            warnings.warn(f"略過 BSS Eval:{exc}", RuntimeWarning, stacklevel=2)

    return result


def summarize(per_track: list[dict]) -> dict:
    """把多首歌的結果彙總成中位數 / 平均 / 四分位數。

    音源分離的分數分佈通常偏斜(少數幾首特別難),所以**中位數比平均值
    更能代表典型表現**,但兩個都留著 —— 平均值對離群值敏感,反而能提醒
    你「有幾首爛得離譜」。
    """
    if not per_track:
        return {}

    keys = {key for track in per_track for key in track
            if isinstance(track.get(key), (int, float))}
    summary: dict = {}
    for key in sorted(keys):
        values = np.array([t[key] for t in per_track
                           if isinstance(t.get(key), (int, float))],
                          dtype=np.float64)
        values = values[np.isfinite(values)]
        if values.size == 0:
            continue
        summary[key] = {
            "median": float(np.median(values)),
            "mean": float(np.mean(values)),
            "q25": float(np.percentile(values, 25)),
            "q75": float(np.percentile(values, 75)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "n": int(values.size),
        }
    return summary
