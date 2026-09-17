"""反覆調參重訓,直到分數不再進步。

    python -m ktisv_tf.sweep --queue data/sweep/queue.json

流程
----
每一輪從佇列裡拿一組設定:

    訓練(ktisv_tf.train)→ 匯出 best 成 ONNX → 在固定的未見歌曲上整段評估

評估分數比目前的冠軍高(超過 ``--min-gain``)就換冠軍,並把 ONNX 複製到
``engine/models/ktisv-instrumental.onnx``,下次打包就會帶著它。
連續 ``--patience`` 輪沒有進步、或佇列空了,就停止。

為什麼不用訓練時的驗證分數當判準
--------------------------------
訓練時的驗證是 3 秒片段、重新隨機配對過的混音;真正上線的是整首原曲、
由引擎切塊重疊相加。兩者不是同一件事。這裡用 ONNX Runtime 跑整段原曲,
走的是和 KTISV 一樣的推論路徑。

佇列檔每一輪都重新讀取,所以跑到一半可以直接編輯它,加入或調整下一輪的設定。
一輪的設定長這樣::

    {"name": "leak3", "resume": "champion", "args": {"--leak-weight": 3, "--steps": 16000}}

``resume`` 可以是 ``"champion"``(接續目前冠軍,preset 必須相同)或 ``null``(從頭訓練)。

分數
----
``伴奏 SI-SDR + 0.5 × 人聲殘留 SIR``,和 ``train.py --select instrumental`` 同一個取向。
注意:同一批驗證歌被反覆拿來挑選,輪數一多分數會帶一點樂觀偏差 —— 這批資料
只有 5 首兩個模型都沒聽過的歌,沒有餘裕再切出獨立的測試集。
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

if __package__ in (None, ""):  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ktisv_research.metrics import si_sdr

RESEARCH = Path(__file__).resolve().parents[1]
PYTHON = sys.executable
DEFAULT_VAL = ["dl-0fdcc8c99fa9963c", "dl-8a6a72df155e23c3", "dl-d5766c6cc3000724",
               "dl-cc20a296046ccb08", "dl-c182856f08ee0ae9"]
BASE_ARGS = {
    "--preset": "small", "--steps": 20000, "--lr": 1.5e-4, "--warmup": 800,
    "--val-every": 500, "--val-batches": 16, "--log-every": 250,
    "--accompaniment-weight": 1, "--leak-weight": 2, "--select": "instrumental",
}


# ── 評估 ────────────────────────────────────────────────────────────────
def leak_sir(estimate: np.ndarray, accompaniment: np.ndarray,
             vocals: np.ndarray, cap: float = 60.0) -> float:
    basis = np.stack([accompaniment.reshape(-1), vocals.reshape(-1)], 1).astype(np.float64)
    coef, *_ = np.linalg.lstsq(basis, estimate.reshape(-1).astype(np.float64), rcond=None)
    wanted = np.sum((coef[0] * basis[:, 0]) ** 2)
    leak = max(np.sum((coef[1] * basis[:, 1]) ** 2), 1e-12)
    return float(min(cap, 10 * np.log10(wanted / leak)))


def separate(session, mixture: np.ndarray) -> np.ndarray:
    """整段推論:半塊間距、Hann 窗重疊相加。回傳人聲 (n, 2)。"""
    seg = session.get_inputs()[0].shape[-1]
    name = session.get_inputs()[0].name
    hop = seg // 2
    n = len(mixture)
    window = np.hanning(seg).astype(np.float32)[:, None]
    out = np.zeros((n, 2), np.float32)
    weight = np.zeros((n, 1), np.float32)
    for start in range(-hop, n, hop):
        chunk = np.zeros((seg, 2), np.float32)
        a, b = max(0, start), min(n, start + seg)
        chunk[a - start:b - start] = mixture[a:b]
        vocals = session.run(None, {name: chunk.T[None]})[0][0].T
        lo = a - start
        out[a:b] += vocals[lo:lo + b - a] * window[lo:lo + b - a]
        weight[a:b] += window[lo:lo + b - a]
    return out / np.maximum(weight, 1e-6)


def evaluate(onnx_path: Path, data: Path, songs: list[str], seconds: float,
             sir_weight: float = 0.5, sir_cap: float = 60.0) -> dict:
    """``sir_cap``:殘留 SIR 超過某個程度就聽不出差別了,再高只是讓平均值
    被一兩首歌灌水、把挑選推向犧牲伴奏的方向。"""
    import onnxruntime as ort

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    sdr, sir, raw = [], [], []
    for song in songs:
        frames = int(seconds * 44100)
        vocals, _ = sf.read(data / song / "vocals.wav", dtype="float32", frames=frames)
        backing, _ = sf.read(data / song / "accompaniment.wav", dtype="float32", frames=frames)
        n = min(len(vocals), len(backing))
        vocals, backing = vocals[:n], backing[:n]
        mixture = vocals + backing
        estimate = mixture - separate(session, mixture)
        sdr.append(si_sdr(backing, estimate))
        raw.append(leak_sir(estimate, backing, vocals))
        sir.append(min(raw[-1], sir_cap))
    result = {"accompaniment_si_sdr": float(np.mean(sdr)),
              "vocal_leak_sir": float(np.mean(sir)),
              "per_song": {s: [round(float(a), 2), round(b, 2)]
                           for s, a, b in zip(songs, sdr, raw)}}
    result["score"] = result["accompaniment_si_sdr"] + sir_weight * result["vocal_leak_sir"]
    return result


# ── 一輪 ────────────────────────────────────────────────────────────────
def run_round(entry: dict, champion: dict | None, folder: Path,
              args: argparse.Namespace) -> dict:
    name = entry["name"]
    out = folder / "runs" / name
    options = {**BASE_ARGS, **entry.get("args", {})}
    options["--out"] = str(out)
    options["--val-tracks"] = ",".join(args.songs)
    resume = entry.get("resume", "champion")
    if resume == "champion":
        if champion is None:
            raise SystemExit("還沒有冠軍,第一輪要指定 resume")
        if champion["preset"] != options["--preset"]:
            raise SystemExit(f"{name}:preset {options['--preset']} 接不上冠軍的 {champion['preset']}")
        options["--resume"] = champion["weights"]
    elif resume:
        options["--resume"] = resume
    if options.get("--resume") and not Path(options["--resume"]).exists():
        raise SystemExit(f"{name}:找不到要接續的權重 {options['--resume']}")

    command = [PYTHON, "-u", "-m", "ktisv_tf.train"]
    for key, value in options.items():
        if value is True:
            command.append(key)
        elif value not in (None, False):
            command += [key, str(value)]

    out.mkdir(parents=True, exist_ok=True)
    log = out / "train.log"
    print(f"\n=== 第 {entry['_round']} 輪:{name} ===\n{' '.join(command[3:])}", flush=True)
    started = time.time()
    with log.open("w", encoding="utf-8") as handle:
        code = subprocess.run(command, cwd=RESEARCH, stdout=handle,
                              stderr=subprocess.STDOUT,
                              env={**__import__("os").environ, "PYTHONIOENCODING": "utf-8"}).returncode
    if code != 0 or not (out / "best.h5").exists():
        return {"name": name, "status": "train_failed", "code": code,
                "minutes": (time.time() - started) / 60}

    onnx_path = folder / "models" / f"{name}.onnx"
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    code = subprocess.run([PYTHON, "-m", "ktisv_tf.export", str(out / "best"),
                           "--out", str(onnx_path)], cwd=RESEARCH,
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode
    if code != 0 or not onnx_path.exists():
        return {"name": name, "status": "export_failed", "code": code}

    scores = evaluate(onnx_path, Path(args.data), args.songs, args.seconds,
                      args.sir_weight, args.sir_cap)
    best_json = json.loads((out / "best.json").read_text("utf-8"))
    return {"name": name, "status": "ok", **scores,
            "minutes": (time.time() - started) / 60,
            "best_step": best_json["step"], "preset": options["--preset"],
            "weights": str(out / "best.h5"), "onnx": str(onnx_path),
            "options": {k: v for k, v in options.items() if k != "--val-tracks"}}


# ── 主流程 ──────────────────────────────────────────────────────────────
def _keep_awake() -> None:
    """掃參數動輒跑一整晚。在這個行程存活期間要求 Windows 不要進入睡眠
    (不改任何電源設定,行程結束就自動失效)。"""
    if sys.platform != "win32":
        return
    import ctypes
    ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
    ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="反覆調參重訓,直到分數不再進步")
    parser.add_argument("--queue", default="data/sweep/queue.json")
    parser.add_argument("--folder", default="data/sweep")
    parser.add_argument("--data", default="data/ktisv-cache")
    parser.add_argument("--seconds", type=float, default=90.0)
    parser.add_argument("--songs", default=",".join(DEFAULT_VAL))
    parser.add_argument("--sir-weight", type=float, default=0.5,
                        help="分數裡人聲殘留 SIR 的權重")
    parser.add_argument("--sir-cap", type=float, default=60.0,
                        help="每首歌的殘留 SIR 先截到這個上限再平均")
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--min-gain", type=float, default=0.1)
    parser.add_argument("--install", default=str(RESEARCH.parent / "engine" / "models"
                                                 / "ktisv-instrumental.onnx"))
    parser.add_argument("--champion", default=None,
                        help="起始冠軍的權重 .h5(旁邊要有同名 .json)")
    parser.add_argument("--deadline", default=None,
                        help="HH:MM,超過這個時間就不再開新的一輪")
    args = parser.parse_args(argv)
    _keep_awake()
    args.songs = [s for s in args.songs.split(",") if s]

    folder = Path(args.folder)
    folder.mkdir(parents=True, exist_ok=True)
    state_path = folder / "state.json"
    history_path = folder / "history.jsonl"
    state = json.loads(state_path.read_text("utf-8")) if state_path.exists() else {}
    champion = state.get("champion")
    done = set(state.get("done", []))
    stale = int(state.get("stale", 0))

    if champion is None and args.champion:
        weights = Path(args.champion)
        meta = json.loads(weights.with_suffix(".json").read_text("utf-8"))
        onnx_path = folder / "models" / "initial.onnx"
        onnx_path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run([PYTHON, "-m", "ktisv_tf.export", str(weights.with_suffix("")),
                        "--out", str(onnx_path)], cwd=RESEARCH, check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        champion = {"name": "initial", "preset": meta["preset"], "weights": str(weights),
                    "onnx": str(onnx_path),
                    **evaluate(onnx_path, Path(args.data), args.songs, args.seconds,
                               args.sir_weight, args.sir_cap)}
        champion.pop("per_song", None)
        print(f"起始冠軍 {weights}:分數 {champion['score']:.2f}"
              f"(伴奏 {champion['accompaniment_si_sdr']:.2f} / 殘留 SIR {champion['vocal_leak_sir']:.2f})")

    def save() -> None:
        state_path.write_text(json.dumps({"champion": champion, "done": sorted(done),
                                          "stale": stale}, ensure_ascii=False, indent=2), "utf-8")

    save()
    round_no = len(done)
    while stale < args.patience:
        queue = json.loads(Path(args.queue).read_text("utf-8"))
        pending = [e for e in queue if e["name"] not in done]
        if not pending:
            print("佇列已空。")
            break
        if args.deadline:
            hour, minute = (int(x) for x in args.deadline.split(":"))
            now = time.localtime()
            if (now.tm_hour, now.tm_min) >= (hour, minute):
                print(f"已過 {args.deadline},不再開新的一輪。")
                break
        entry = dict(pending[0])
        round_no += 1
        entry["_round"] = round_no
        result = run_round(entry, champion, folder, args)
        done.add(entry["name"])

        improved = (result.get("status") == "ok" and
                    (champion is None or result["score"] > champion["score"] + args.min_gain))
        result["champion_before"] = champion["score"] if champion else None
        result["improved"] = improved
        with history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")

        if result.get("status") != "ok":
            print(f"  ✗ {result['name']} 失敗:{result['status']}", flush=True)
            stale += 1
        else:
            print(f"  {result['name']}:分數 {result['score']:.2f}"
                  f"(伴奏 {result['accompaniment_si_sdr']:.2f} / 殘留 SIR {result['vocal_leak_sir']:.2f})"
                  f"  冠軍 {champion['score'] if champion else float('nan'):.2f}"
                  f"  {'★ 新冠軍' if improved else '沒有進步'}"
                  f"  ({result['minutes']:.0f} 分鐘)", flush=True)
            if improved:
                champion = {k: result[k] for k in ("name", "preset", "weights", "onnx", "score",
                                                   "accompaniment_si_sdr", "vocal_leak_sir")}
                stale = 0
                if args.install:
                    Path(args.install).parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(result["onnx"], args.install)
                    print(f"  已安裝到 {args.install}", flush=True)
            else:
                stale += 1
        save()

    print(f"\n結束。冠軍:{champion['name'] if champion else '無'}"
          f"  分數 {champion['score'] if champion else float('nan'):.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
