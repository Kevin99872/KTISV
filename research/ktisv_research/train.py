"""訓練迴圈。

    python -m ktisv_research.train --dataset mir1k --root data/MIR-1K/Wavfile

損失函數的選擇
--------------
一個容易踩的坑:模型輸出人聲,伴奏定義為 ``混音 − 人聲``,而資料保證
``混音 = 人聲 + 伴奏``。所以

    |伴奏_估計 − 伴奏_真值| = |(混音 − 人聲_估計) − (混音 − 人聲_真值)|
                            = |人聲_真值 − 人聲_估計|

**波形域的伴奏損失和人聲損失是同一個數。** 兩個都加只是把損失乘二,
不會給模型任何額外訊息。這裡只算人聲的那一項。

(頻譜域不同 —— 幅度譜取了絕對值,不再是線性的,所以幅度損失對兩軌
確實是兩個不同的量。但預設仍只算人聲,理由見下。)

預設用「波形 L1 + 幅度譜 L1」:
  * 波形 L1 —— 直接對應最終聽到的東西,而且隱含地要求相位正確
  * 幅度譜 L1 —— 波形 L1 對相位誤差過度敏感,單用它時模型會為了對齊相位
    而犧牲頻譜正確性;加一項幅度損失把注意力拉回「頻譜對不對」

驗證指標用 SI-SDR 而不是損失值:損失是給優化器看的,SI-SDR 才是這個領域
拿來比較的東西,而且對整體音量不敏感 —— 模型把輸出整體放大 1 dB 不該
被算成變差。
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from . import SAMPLE_RATE, __version__
from .data import (DataConfig, MixtureDataset, describe, load_dataset,
                   split_by_group)
from .metrics import si_sdr
from .model import PRESETS, Separator, build


# ── 損失 ────────────────────────────────────────────────────────────────
def spectral_l1(model: Separator, estimate: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
    """幅度譜上的 L1。用模型自己的 STFT 設定,兩邊才是同一個變換。"""
    est_mag = model.stft(estimate).abs()
    ref_mag = model.stft(target).abs()
    # log1p 壓縮:與模型輸入端同樣的理由 —— 不壓縮的話,少數高能量的
    # 時頻點會主導損失,安靜段落的誤差幾乎沒有梯度。
    return F.l1_loss(torch.log1p(est_mag), torch.log1p(ref_mag))


def compute_loss(model: Separator, output: dict[str, torch.Tensor],
                 batch: dict[str, torch.Tensor],
                 spectral_weight: float) -> tuple[torch.Tensor, dict[str, float]]:
    wave = F.l1_loss(output["vocals"], batch["vocals"])
    if spectral_weight <= 0:
        return wave, {"wave": wave.item()}

    spectral = spectral_l1(model, output["vocals"], batch["vocals"])
    total = wave + spectral_weight * spectral
    return total, {"wave": wave.item(), "spectral": spectral.item()}


# ── 驗證 ────────────────────────────────────────────────────────────────
@torch.no_grad()
def validate(model: Separator, loader, device: str,
             limit: int | None = None) -> dict[str, float]:
    """在固定的驗證樣本上算 SI-SDR。回傳 dB,越高越好。"""
    model.eval()
    vocal_scores: list[float] = []
    accompaniment_scores: list[float] = []
    baseline_scores: list[float] = []

    for index, batch in enumerate(loader):
        if limit is not None and index >= limit:
            break
        mixture = batch["mixture"].to(device)
        output = model(mixture)

        for key, target, scores in (
                ("vocals", batch["vocals"], vocal_scores),
                ("accompaniment", batch["accompaniment"], accompaniment_scores)):
            estimate = output[key].cpu().numpy()
            reference = target.numpy()
            for i in range(len(reference)):
                scores.append(si_sdr(reference[i].T, estimate[i].T))

        # 基準線:完全不分離(直接把混音當人聲)。模型至少要贏過它,
        # 否則這個模型不如不用。
        reference = batch["vocals"].numpy()
        passthrough = batch["mixture"].numpy()
        for i in range(len(reference)):
            baseline_scores.append(si_sdr(reference[i].T, passthrough[i].T))

    model.train()

    def mean(values: list[float]) -> float:
        finite = [v for v in values if math.isfinite(v)]
        return float(np.mean(finite)) if finite else float("nan")

    vocals = mean(vocal_scores)
    baseline = mean(baseline_scores)
    return {
        "si_sdr_vocals": vocals,
        "si_sdr_accompaniment": mean(accompaniment_scores),
        "si_sdr_passthrough": baseline,
        # 真正該看的數字:比「什麼都不做」好了多少
        "si_sdr_improvement": vocals - baseline,
    }


# ── 檢查點 ──────────────────────────────────────────────────────────────
def save_checkpoint(path: Path, model: Separator, optimizer, step: int,
                    best: float, args: argparse.Namespace) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "version": __version__,
        "step": step,
        "best_si_sdr": best,
        "preset": args.preset,
        "model_config": asdict(model.config),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "args": vars(args),
    }, path)


def load_checkpoint(path: Path, model: Separator, optimizer,
                    device: str) -> tuple[int, float]:
    state = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    if optimizer is not None and "optimizer" in state:
        optimizer.load_state_dict(state["optimizer"])
    return int(state.get("step", 0)), float(state.get("best_si_sdr", -math.inf))


# ── 主流程 ──────────────────────────────────────────────────────────────
def build_loaders(args, config: DataConfig):
    from torch.utils.data import DataLoader

    pairs = load_dataset(args.dataset, Path(args.root), config, args.limit)
    print(f"載入 {args.dataset}: {describe(pairs, config.samplerate)}")

    train_pairs, val_pairs = split_by_group(pairs, args.val_ratio, args.seed)
    print(f"  訓練 {describe(train_pairs, config.samplerate)}")
    print(f"  驗證 {describe(val_pairs, config.samplerate)}"
          f"  (歌手: {sorted({p.group for p in val_pairs})})")

    train_set = MixtureDataset(train_pairs, config,
                               length=args.steps_per_epoch * args.batch_size,
                               seed=args.seed)
    # 驗證集固定 —— 每次都是同一批混音,分數才可比
    val_set = MixtureDataset(val_pairs, config, length=args.val_samples,
                             seed=args.seed + 1, deterministic=True)

    common = dict(batch_size=args.batch_size, num_workers=args.workers,
                  pin_memory=args.device == "cuda", drop_last=True)
    return (DataLoader(train_set, shuffle=False, **common),
            DataLoader(val_set, shuffle=False, **common))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="訓練人聲分離模型")
    parser.add_argument("--dataset", default="folders", choices=["mir1k", "folders"])
    parser.add_argument("--root", required=True, help="資料集根目錄")
    parser.add_argument("--limit", type=int, default=None, help="只用前 N 個片段(除錯用)")
    parser.add_argument("--preset", default="small", choices=list(PRESETS))
    parser.add_argument("--out", default="data/runs/latest")

    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--steps-per-epoch", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--segment-seconds", type=float, default=6.0)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--spectral-weight", type=float, default=1.0,
                        help="幅度譜損失的權重;0 = 只用波形 L1")
    parser.add_argument("--independent-prob", type=float, default=0.5)

    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--val-samples", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=0,
                        help="Windows 上 spawn 會把資料重新載入一遍,預設 0")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--amp", action="store_true",
                        help="U-Net 用混合精度。6 GB 卡上能換到更大的 batch")
    parser.add_argument("--resume", default=None)
    args = parser.parse_args(argv)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = True

    config = DataConfig(segment_seconds=args.segment_seconds,
                        independent_prob=args.independent_prob)
    train_loader, val_loader = build_loaders(args, config)

    device = args.device
    model = build(args.preset).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, args.steps), eta_min=args.lr * 0.05)

    step, best = 0, -math.inf
    if args.resume:
        step, best = load_checkpoint(Path(args.resume), model, optimizer, device)
        print(f"從 {args.resume} 續訓,step={step}, best={best:.2f} dB")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / "log.jsonl"
    print(f"模型 {args.preset}: {model.num_parameters / 1e6:.2f} M 參數 · 裝置 {device}")
    print(f"輸出 → {out}")

    # 訓練前先量一次:模型還沒學任何東西時的分數。之後的每個數字都要跟
    # 它比,否則「12 dB」是好是壞根本無從判斷。
    initial = validate(model, val_loader, device)
    print(f"訓練前 SI-SDR: 人聲 {initial['si_sdr_vocals']:.2f} dB"
          f" / 不分離基準 {initial['si_sdr_passthrough']:.2f} dB")

    started = time.perf_counter()
    running: list[float] = []
    done = False

    while not done:
        for batch in train_loader:
            mixture = batch["mixture"].to(device, non_blocking=True)
            targets = {k: v.to(device, non_blocking=True)
                       for k, v in batch.items() if k != "mixture"}

            # 只有 U-Net 的卷積會被降到 bf16 —— autocast 不碰 torch.stft,
            # 複數頻譜仍然是 fp32。用 bf16 而非 fp16 是因為它的指數範圍和
            # fp32 一樣,不需要 GradScaler,也就沒有 scale 爆掉的問題。
            with torch.autocast(device_type=device.split(":")[0],
                                dtype=torch.bfloat16, enabled=args.amp):
                output = model(mixture)
            loss, parts = compute_loss(model, output, targets, args.spectral_weight)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step()

            running.append(loss.item())
            step += 1

            if step % 50 == 0:
                elapsed = time.perf_counter() - started
                mean_loss = float(np.mean(running[-50:]))
                peak = (torch.cuda.max_memory_allocated() / 1e9
                        if device == "cuda" else 0.0)
                print(f"  step {step:6d}/{args.steps}  loss {mean_loss:.4f}"
                      f"  {parts}  {elapsed / step:.2f} s/step  峰值 {peak:.2f} GB")

            if step % args.steps_per_epoch == 0 or step >= args.steps:
                scores = validate(model, val_loader, device)
                record = {"step": step, "loss": float(np.mean(running[-200:])),
                          "lr": scheduler.get_last_lr()[0], **scores}
                with log_path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(record, ensure_ascii=False) + "\n")

                improvement = scores["si_sdr_improvement"]
                print(f"  ── 驗證 step {step}: 人聲 SI-SDR"
                      f" {scores['si_sdr_vocals']:.2f} dB"
                      f"(比不分離好 {improvement:+.2f} dB)")

                save_checkpoint(out / "last.pt", model, optimizer, step, best, args)
                if scores["si_sdr_vocals"] > best:
                    best = scores["si_sdr_vocals"]
                    save_checkpoint(out / "best.pt", model, optimizer, step,
                                    best, args)
                    print(f"     ↑ 最佳,已存 best.pt")

            if step >= args.steps:
                done = True
                break

    total = time.perf_counter() - started
    print(f"\n完成 {step} 步,耗時 {total / 3600:.2f} 小時。最佳 SI-SDR {best:.2f} dB")
    print(f"檢查點:{out / 'best.pt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
