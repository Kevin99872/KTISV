"""TensorFlow 訓練迴圈。

    python -m ktisv_tf.train --data data/ktisv-cache --preset small

損失函數
--------
沿用 PyTorch 版的結論(``ktisv_research/train.py`` 有完整推導),這裡只複述
最容易踩的那一點:

模型輸出人聲、伴奏定義為 ``混音 − 人聲``,而資料保證 ``混音 = 人聲 + 伴奏``,
所以在**波形域**上

    |伴奏_估計 − 伴奏_真值| = |人聲_真值 − 人聲_估計|

兩軌的損失是同一個數。把兩個都加進去只是把損失乘二,不會給模型任何額外
訊息。這裡只算人聲那一項。

用「波形 L1 + 幅度譜 L1」:
  * 波形 L1 —— 直接對應最終聽到的東西,而且隱含地要求相位正確
  * 幅度譜 L1 —— 波形 L1 對相位誤差過度敏感,單用它時模型會為了對齊相位
    而犧牲頻譜正確性;加一項幅度損失把注意力拉回「頻譜對不對」

驗證看 SI-SDR 而不是損失值:損失是給優化器看的,SI-SDR 才是這個領域用來
比較的東西,而且對整體音量不敏感 —— 模型把輸出整體放大 1 dB 不該算成變差。

還有一個更重要的數字:``si_sdr_improvement``,也就是「比什麼都不做好了多少」。
把混音原封不動當人聲交出去也有一個 SI-SDR;模型贏不過那條基準線就等於沒用。
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import tensorflow as tf

if __package__ in (None, ""):  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ktisv_research.metrics import si_sdr

from . import SAMPLE_RATE, __version__
from .data import (DataConfig, describe, load_tracks, make_dataset,
                   split_by_track)
from .model import (PRESETS, ConvSTFT, aligned_length, build_separator,
                    build_unet, config_to_dict)

keras = tf.keras


# ── 損失 ────────────────────────────────────────────────────────────────
class Loss:
    """波形 L1 + 幅度譜 L1(+ 伴奏幅度譜 + 人聲殘留懲罰)。

    STFT 用和模型同一組卷積核,兩邊才是同一個變換。

    為什麼伴奏還要再算一次
    ----------------------
    檔頭說「伴奏損失和人聲損失是同一個數」—— 那只在**波形域**成立。
    幅度譜是非線性的:``|混音 − 人聲_估計|`` 和 ``|人聲_估計|`` 在同一個
    時頻點上的誤差不相等,尤其在人聲與樂器重疊的地方。所以伴奏的幅度譜
    L1 是真的有新資訊的一項。

    人聲殘留懲罰
    ------------
    KTISV 的主要用途是「只留樂器」。伴奏軌裡**多出來**的能量(真值沒有、
    估計有)幾乎就是漏進來的人聲;**少掉**的能量是被誤刪的樂器。人耳對前者
    敏感得多 —— 伴奏裡飄著一絲人聲比樂器稍微悶一點明顯。所以對「多出來」
    這一側額外加權,讓模型寧可多刪一點也不要漏。
    權重太高會讓伴奏變悶,``--leak-weight`` 可以調。
    """

    def __init__(self, config, spectral_weight: float,
                 accompaniment_weight: float = 0.0,
                 leak_weight: float = 0.0) -> None:
        self.stft = ConvSTFT(config.n_fft, config.hop_length, name="loss_stft")
        self.spectral_weight = float(spectral_weight)
        self.accompaniment_weight = float(accompaniment_weight)
        self.leak_weight = float(leak_weight)
        self.samples = None

    def magnitude(self, waveform: tf.Tensor) -> tf.Tensor:
        """(B, C, N) → 對數壓縮後的幅度譜。"""
        flat = tf.reshape(waveform, [-1, tf.shape(waveform)[-1]])
        real, imag = self.stft(flat)
        # 對數壓縮:與模型輸入端同樣的理由 —— 不壓縮的話,少數高能量的
        # 時頻點會主導損失,安靜段落的誤差幾乎沒有梯度。
        return tf.math.log(tf.sqrt(real * real + imag * imag + 1e-12) + 1.0)

    def __call__(self, estimate: tf.Tensor, target: tf.Tensor,
                 mixture: tf.Tensor | None = None
                 ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        """回傳 (總損失, 波形, 人聲幅度譜, 伴奏項)。"""
        zero = tf.constant(0.0)
        wave = tf.reduce_mean(tf.abs(estimate - target))
        total = wave
        spectral = zero
        if self.spectral_weight > 0:
            spectral = tf.reduce_mean(
                tf.abs(self.magnitude(estimate) - self.magnitude(target)))
            total = total + self.spectral_weight * spectral

        accompaniment = zero
        if mixture is not None and (self.accompaniment_weight > 0
                                    or self.leak_weight > 0):
            est_a = self.magnitude(mixture - estimate)
            true_a = self.magnitude(mixture - target)
            diff = est_a - true_a
            if self.accompaniment_weight > 0:
                accompaniment = accompaniment + self.accompaniment_weight * \
                    tf.reduce_mean(tf.abs(diff))
            if self.leak_weight > 0:
                # 只罰「伴奏多出來」的那一側 —— 那是漏進來的人聲
                accompaniment = accompaniment + self.leak_weight * \
                    tf.reduce_mean(tf.nn.relu(diff))
            total = total + accompaniment
        return total, wave, spectral, accompaniment


# ── 驗證 ────────────────────────────────────────────────────────────────
def vocal_leak_sir(estimate: np.ndarray, accompaniment: np.ndarray,
                   vocals: np.ndarray, ceiling: float = 40.0) -> float:
    """伴奏估計裡「樂器 vs 殘留人聲」的能量比(dB),越高代表人聲漏得越少。

    SI-SDR 把「漏人聲」與「樂器被削掉、有假象」混成一個數,分不出來。
    這裡用 BSS Eval 的 SIR 概念:把估計投影到 {真伴奏, 真人聲} 張成的空間,
    分別看兩個分量的能量。只有人聲那一份才算干擾。
    上限夾在 ``ceiling``:人聲分量趨近 0 時比值會發散,平均會被單一樣本主導。
    """
    basis = np.stack([accompaniment.reshape(-1), vocals.reshape(-1)], axis=1)
    target = estimate.reshape(-1)
    coef, *_ = np.linalg.lstsq(basis.astype(np.float64),
                               target.astype(np.float64), rcond=None)
    wanted = np.sum((coef[0] * basis[:, 0]) ** 2)
    leak = np.sum((coef[1] * basis[:, 1]) ** 2)
    if wanted <= 1e-12:
        return float("nan")
    return float(min(ceiling, 10.0 * np.log10(wanted / max(leak, 1e-12))))


def validate(model, dataset, batches: int) -> dict[str, float]:
    """在固定的驗證樣本上算 SI-SDR。回傳 dB,越高越好。"""
    vocal_scores: list[float] = []
    accompaniment_scores: list[float] = []
    baseline_scores: list[float] = []
    accompaniment_baseline: list[float] = []
    leak_scores: list[float] = []
    leak_baseline: list[float] = []

    for index, (mixture, vocals) in enumerate(dataset):
        if index >= batches:
            break
        estimate_v, estimate_a = model(mixture, training=False)
        estimate_v = estimate_v.numpy()
        estimate_a = estimate_a.numpy()
        mixture_np = mixture.numpy()
        vocals_np = vocals.numpy()
        accompaniment_np = mixture_np - vocals_np

        for i in range(len(vocals_np)):
            vocal_scores.append(si_sdr(vocals_np[i].T, estimate_v[i].T))
            accompaniment_scores.append(
                si_sdr(accompaniment_np[i].T, estimate_a[i].T))
            # 基準線:完全不分離(直接把混音當人聲)。模型至少要贏過它。
            baseline_scores.append(si_sdr(vocals_np[i].T, mixture_np[i].T))
            accompaniment_baseline.append(
                si_sdr(accompaniment_np[i].T, mixture_np[i].T))
            leak_scores.append(vocal_leak_sir(
                estimate_a[i], accompaniment_np[i], vocals_np[i]))
            leak_baseline.append(vocal_leak_sir(
                mixture_np[i], accompaniment_np[i], vocals_np[i]))

    def mean(values: list[float]) -> float:
        finite = [v for v in values if math.isfinite(v)]
        return float(np.mean(finite)) if finite else float("nan")

    vocals_db = mean(vocal_scores)
    baseline_db = mean(baseline_scores)
    accompaniment_db = mean(accompaniment_scores)
    accompaniment_base_db = mean(accompaniment_baseline)
    return {
        "si_sdr_vocals": vocals_db,
        "si_sdr_accompaniment": accompaniment_db,
        "si_sdr_passthrough": baseline_db,
        "si_sdr_accompaniment_passthrough": accompaniment_base_db,
        # 真正該看的數字:比「什麼都不做」好了多少
        "si_sdr_improvement": vocals_db - baseline_db,
        # KTISV 要的是乾淨的伴奏,所以用 --select accompaniment 時挑這個
        "si_sdr_accompaniment_improvement":
            accompaniment_db - accompaniment_base_db,
        "sir_accompaniment": mean(leak_scores),
        # 人聲殘留比「完全不分離」少了多少 dB
        "sir_accompaniment_improvement": mean(leak_scores) - mean(leak_baseline),
    }


# ── 學習率 ──────────────────────────────────────────────────────────────
class WarmupCosine(keras.optimizers.schedules.LearningRateSchedule):
    """線性暖身後 cosine 衰減到 5%。

    續訓時優化器的動量是全新的,第一步就用滿學習率會把已經收斂的權重
    打散 —— 實測從 +8.97 dB 掉到 +7.8 dB,之後花上千步才爬回來。
    """

    def __init__(self, peak: float, steps: int, warmup: int) -> None:
        self.peak = float(peak)
        self.steps = max(1, int(steps))
        self.warmup = max(0, int(warmup))

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        warm = tf.cast(max(self.warmup, 1), tf.float32)
        span = tf.cast(max(self.steps - self.warmup, 1), tf.float32)
        progress = tf.clip_by_value((step - self.warmup) / span, 0.0, 1.0)
        cosine = 0.05 + 0.95 * 0.5 * (1.0 + tf.cos(math.pi * progress))
        ramp = tf.minimum(1.0, (step + 1.0) / warm) if self.warmup else 1.0
        return self.peak * ramp * cosine

    def get_config(self) -> dict:
        return {"peak": self.peak, "steps": self.steps, "warmup": self.warmup}


# ── 檢查點 ──────────────────────────────────────────────────────────────
def save_checkpoint(folder: Path, unet, config, args, step: int,
                    best: float, tag: str) -> None:
    """只存 U-Net 的權重。

    STFT / iSTFT 是常數層(卷積核由公式算出來,沒有可訓練參數),所以
    重建模型時再算一次就好 —— 而且這樣同一份權重可以套到任何片段長度上,
    訓練用 3 秒、匯出用 6 秒不必重訓。
    """
    folder.mkdir(parents=True, exist_ok=True)
    unet.save_weights(str(folder / f"{tag}.h5"))
    (folder / f"{tag}.json").write_text(json.dumps({
        "version": __version__,
        "framework": "tensorflow",
        "tf_version": tf.__version__,
        "step": step,
        "best_si_sdr": best,
        "preset": args.preset,
        "model_config": config_to_dict(config),
        "segment_samples": args.segment_samples,
        "samplerate": SAMPLE_RATE,
        "args": {k: v for k, v in vars(args).items()
                 if isinstance(v, (int, float, str, bool, type(None)))},
    }, ensure_ascii=False, indent=2), "utf-8")


# ── 主流程 ──────────────────────────────────────────────────────────────
def configure_devices(prefer_gpu: bool) -> str:
    gpus = tf.config.list_physical_devices("GPU")
    if not prefer_gpu or not gpus:
        if prefer_gpu and not gpus:
            print("⚠ 沒偵測到 GPU,改用 CPU。TF 2.9 需要 CUDA 11.x 的 DLL;"
                  "見 research/TENSORFLOW.md")
        tf.config.set_visible_devices([], "GPU")
        return "CPU"
    for gpu in gpus:
        # 不要一開始就吃掉整張卡 —— 這台只有 6 GB,而且 demucs 可能同時在跑
        tf.config.experimental.set_memory_growth(gpu, True)
    return f"GPU ({len(gpus)})"


def run(args: argparse.Namespace) -> int:
    device = configure_devices(not args.cpu)

    config = PRESETS[args.preset]
    args.segment_samples = aligned_length(args.seconds, config, SAMPLE_RATE)

    tracks = load_tracks(Path(args.data), args.limit)
    if args.val_tracks:
        # 續訓時一定要用這個:隨機切分會隨資料量改變,新的驗證集可能正好是
        # 舊權重訓練時聽過的歌 —— 分數漂亮但毫無意義,挑出來的 best 也不可信。
        wanted = {name.strip() for name in args.val_tracks.split(",") if name.strip()}
        val_tracks = [t for t in tracks if t.name in wanted]
        train_tracks = [t for t in tracks if t.name not in wanted]
        missing = wanted - {t.name for t in val_tracks}
        if missing or not val_tracks:
            raise SystemExit(f"--val-tracks 找不到:{', '.join(sorted(missing))}")
    else:
        train_tracks, val_tracks = split_by_track(tracks, args.val_ratio, args.seed)
    print(f"裝置      {device}")
    print(f"資料      {describe(tracks)}")
    print(f"  訓練    {describe(train_tracks)}")
    print(f"  驗證    {describe(val_tracks)}  ({', '.join(t.name for t in val_tracks)})")

    data_config = DataConfig(segment_samples=args.segment_samples,
                             independent_prob=args.independent_prob)
    train_data = make_dataset(train_tracks, data_config, args.batch_size,
                              seed=args.seed)
    val_data = make_dataset(val_tracks, data_config, args.batch_size,
                            seed=args.seed + 1, deterministic=True,
                            length=args.val_batches * args.batch_size)

    unet = build_unet(config)
    model = build_separator(config, args.segment_samples, unet)
    print(f"模型      {args.preset} / {unet.count_params():,} 參數 / "
          f"片段 {args.segment_samples} 取樣 = "
          f"{args.segment_samples / SAMPLE_RATE:.2f} 秒")

    if args.resume:
        unet.load_weights(args.resume)
        print(f"續訓      {args.resume}")

    schedule = WarmupCosine(args.lr, args.steps, args.warmup)
    optimizer = keras.optimizers.Adam(schedule)
    loss_fn = Loss(config, args.spectral_weight,
                   args.accompaniment_weight, args.leak_weight)
    def selection_score(scores: dict) -> float:
        if args.select == "vocals":
            return scores["si_sdr_improvement"]
        if args.select == "accompaniment":
            return scores["si_sdr_accompaniment_improvement"]
        # instrumental:伴奏品質為主,人聲殘留少的額外加分。只看 SIR 的話,
        # 把伴奏整個削薄也能拿高分;只看 SI-SDR 又量不到「漏人聲」。
        return (scores["si_sdr_accompaniment_improvement"]
                + 0.5 * scores["sir_accompaniment_improvement"])

    @tf.function
    def train_step(mixture, vocals):
        with tf.GradientTape() as tape:
            estimate, _ = model(mixture, training=True)
            total, wave, spectral, acc = loss_fn(estimate, vocals, mixture)
        gradients = tape.gradient(total, model.trainable_variables)
        # 梯度裁剪:遮罩逼近 0/1 時 sigmoid 會出現大梯度(TRAINING.md 4.2)
        gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return total, wave, spectral, acc

    run_folder = Path(args.out)
    run_folder.mkdir(parents=True, exist_ok=True)
    history = (run_folder / "history.jsonl").open("a", encoding="utf-8")

    best = -float("inf")
    if args.resume:
        # 續訓的起點分數。資料增加後驗證集的歌可能換了,舊的 best.json
        # 分數不能直接拿來比,要在同一批題目上重量一次。
        scores = validate(model, val_data, args.val_batches)
        print(f"  [續訓起點] 人聲改善 {scores['si_sdr_improvement']:+.2f} dB"
              f"  伴奏改善 {scores['si_sdr_accompaniment_improvement']:+.2f} dB"
              f"  人聲殘留 SIR {scores['sir_accompaniment']:+.2f} dB"
              f"  分數 {selection_score(scores):+.2f}")
        history.write(json.dumps({"step": 0, **scores}) + "\n")
        # 起點本身就是一個候選:整輪續訓都沒超過它的話,best.h5 仍是原本的
        # 權重,不會匯出一個比舊版還差的模型。
        best = selection_score(scores)
        save_checkpoint(run_folder, unet, config, args, 0, best, "best")
    window: list[float] = []
    started = time.time()
    print(f"\n開始訓練 {args.steps} 步,每 {args.val_every} 步驗證一次\n")

    for step, (mixture, vocals) in enumerate(train_data, 1):
        total, wave, spectral, acc = train_step(mixture, vocals)
        window.append(float(total))

        if step % args.log_every == 0:
            elapsed = time.time() - started
            print(f"  {step:>6}/{args.steps}  loss {np.mean(window):.4f}"
                  f"  (wave {float(wave):.4f} spec {float(spectral):.4f}"
                  f" acc {float(acc):.4f})"
                  f"  {step / elapsed:.2f} step/s", flush=True)
            window.clear()

        if step % args.val_every == 0 or step == args.steps:
            scores = validate(model, val_data, args.val_batches)
            improvement = selection_score(scores)
            mark = ""
            if improvement > best:
                best = improvement
                save_checkpoint(run_folder, unet, config, args, step, best, "best")
                mark = "  ← 最佳"
            save_checkpoint(run_folder, unet, config, args, step, best, "last")

            print(f"  [驗證 {step}] 人聲 {scores['si_sdr_vocals']:+.2f} dB"
                  f"  伴奏 {scores['si_sdr_accompaniment']:+.2f} dB"
                  f"  人聲改善 {scores['si_sdr_improvement']:+.2f} dB"
                  f"  伴奏改善 {scores['si_sdr_accompaniment_improvement']:+.2f} dB"
                  f"  人聲殘留 SIR {scores['sir_accompaniment']:+.2f} dB"
                  f"  分數 {improvement:+.2f}{mark}", flush=True)
            history.write(json.dumps({"step": step, **scores}) + "\n")
            history.flush()

        if step >= args.steps:
            break

    history.close()
    minutes = (time.time() - started) / 60
    print(f"\n完成:{args.steps} 步 / {minutes:.1f} 分鐘")
    print(f"最佳改善 {best:+.2f} dB → {run_folder / 'best.h5'}")
    if best <= 0:
        print("⚠ 改善沒有超過 0 dB —— 模型還不如直接把混音交出去。"
              "資料量太少或訓練步數不足。")
    print(f"\n匯出:python -m ktisv_tf.export {run_folder / 'best'} "
          f"--out data/models/vocals-tf.onnx")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="訓練 TensorFlow 版人聲分離模型")
    parser.add_argument("--data", default="data/ktisv-cache")
    parser.add_argument("--out", default="data/runs/tf")
    parser.add_argument("--preset", default="small", choices=list(PRESETS))
    parser.add_argument("--seconds", type=float, default=3.0,
                        help="訓練片段長度(會對齊到 2^depth 的整數幀)")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--spectral-weight", type=float, default=1.0)
    parser.add_argument("--accompaniment-weight", type=float, default=0.0,
                        help="伴奏幅度譜 L1 的權重(0 = 關閉,沿用舊行為)")
    parser.add_argument("--leak-weight", type=float, default=0.0,
                        help="伴奏裡「多出來」能量(漏進來的人聲)的額外懲罰")
    parser.add_argument("--warmup", type=int, default=0,
                        help="學習率線性暖身的步數。續訓時建議 500 左右")
    parser.add_argument("--select", default="vocals",
                        choices=["vocals", "accompaniment", "instrumental"],
                        help="best.h5 依哪一軌的改善量挑選")
    parser.add_argument("--independent-prob", type=float, default=0.5)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--val-tracks", default="",
                        help="逗號分隔的驗證集歌名(資料夾名),指定時取代隨機切分")
    parser.add_argument("--val-every", type=int, default=250)
    parser.add_argument("--val-batches", type=int, default=8)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--limit", type=int, default=None,
                        help="只用前幾首(除錯用)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", default=None, help="接續某個 .h5 權重")
    parser.add_argument("--cpu", action="store_true", help="強制用 CPU")
    return run(parser.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
