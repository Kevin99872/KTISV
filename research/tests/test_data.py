"""資料管線與 ONNX 匯出的測試。

    python -m tests.test_data

這裡守的是幾個「壞了不會報錯,只會安靜地毀掉訓練」的性質:
可加性、切分不重疊、驗證集可重現、ONNX 與 PyTorch 輸出一致。
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from ktisv_research import SAMPLE_RATE
from ktisv_research.data import (DataConfig, MixtureDataset, StemPair,
                                 load_dataset, split_by_group)

FAILURES: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    mark = "ok  " if condition else "FAIL"
    print(f"  [{mark}] {name}{('  — ' + detail) if detail else ''}")
    if not condition:
        FAILURES.append(name)


def fake_pairs(count: int = 6, seconds: float = 4.0) -> list[StemPair]:
    rng = np.random.default_rng(0)
    n = int(seconds * SAMPLE_RATE)
    pairs = []
    for i in range(count):
        vocals = (rng.standard_normal((n, 2)) * 0.1).astype(np.float32)
        accompaniment = (rng.standard_normal((n, 2)) * 0.1).astype(np.float32)
        pairs.append(StemPair(f"clip{i}", f"singer{i // 2}", vocals, accompaniment))
    return pairs


def test_split() -> None:
    print("訓練/驗證切分")
    pairs = fake_pairs(6)               # 3 位歌手,每人 2 段
    train, val = split_by_group(pairs, val_ratio=0.34, seed=0)

    train_groups = {p.group for p in train}
    val_groups = {p.group for p in val}
    check("分組完全不重疊", train_groups.isdisjoint(val_groups),
          f"訓練 {sorted(train_groups)} / 驗證 {sorted(val_groups)}")
    check("兩邊都非空", len(train) > 0 and len(val) > 0)
    check("沒有片段遺失", len(train) + len(val) == len(pairs))

    # 同一位歌手的兩個片段必須待在同一邊 —— 這正是按組切的目的
    ok = all(sum(1 for p in pairs if p.group == g) ==
             sum(1 for p in train if p.group == g) for g in train_groups)
    check("同一歌手不會被拆到兩邊", ok)

    try:
        split_by_group(fake_pairs(2)[:1], 0.5, 0)
        check("只有一組時會拒絕切分", False)
    except ValueError:
        check("只有一組時會拒絕切分", True)


def test_dataset() -> None:
    print("樣本合成")
    config = DataConfig(segment_seconds=2.0)
    dataset = MixtureDataset(fake_pairs(6), config, length=16, seed=0)

    sample = dataset[0]
    expected = (2, config.segment_samples)
    check("形狀是 (聲道, 取樣)",
          all(tuple(v.shape) == expected for v in sample.values()),
          str({k: tuple(v.shape) for k, v in sample.items()}))

    # 可加性是監督式訓練的前提:混音必須嚴格等於兩軌相加,
    # 否則正確答案本身就是錯的,而模型會忠實地學會那個錯誤。
    worst = 0.0
    for i in range(16):
        item = dataset[i]
        residual = (item["mixture"] - item["vocals"]
                    - item["accompaniment"]).abs().max().item()
        worst = max(worst, residual)
    check("可加性殘差極小", worst < 1e-5, f"{worst:.2e}")

    peaks = [dataset[i]["mixture"].abs().max().item() for i in range(16)]
    check("混音不削波", max(peaks) <= 1.0, f"最大峰值 {max(peaks):.3f}")
    check("不是全靜音", min(peaks) > 1e-3, f"最小峰值 {min(peaks):.4f}")


def test_determinism() -> None:
    print("重現性")
    config = DataConfig(segment_seconds=2.0)
    pairs = fake_pairs(6)

    fixed = MixtureDataset(pairs, config, length=8, seed=1, deterministic=True)
    check("驗證集每次都一樣",
          bool(torch.equal(fixed[3]["mixture"], fixed[3]["mixture"])))
    again = MixtureDataset(pairs, config, length=8, seed=1, deterministic=True)
    check("同一 seed 重建後仍一致",
          bool(torch.equal(fixed[3]["mixture"], again[3]["mixture"])))

    other = MixtureDataset(pairs, config, length=8, seed=2, deterministic=True)
    check("換 seed 就不同",
          not torch.equal(fixed[3]["mixture"], other[3]["mixture"]))

    torch.manual_seed(0)
    random_set = MixtureDataset(pairs, config, length=8, seed=1)
    check("訓練集有隨機性",
          not torch.equal(random_set[3]["mixture"], random_set[3]["mixture"]))


def test_mir1k_layout() -> None:
    print("MIR-1K 版面")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        rng = np.random.default_rng(0)
        n = 16000 * 3
        # MIR-1K 是 16 kHz、左伴奏右人聲。用可分辨的訊號才驗得出有沒有接反。
        t = np.arange(n) / 16000
        for singer in ("abjones", "amy"):
            for clip in (1, 2):
                accompaniment = 0.2 * np.sin(2 * np.pi * 200 * t)
                vocals = 0.2 * np.sin(2 * np.pi * 3000 * t)
                sf.write(root / f"{singer}_{clip}_01.wav",
                         np.column_stack([accompaniment, vocals]), 16000)

        config = DataConfig()
        pairs = load_dataset("mir1k", root, config)
        check("讀到全部四個片段", len(pairs) == 4, str(len(pairs)))
        check("歌手從檔名取出", {p.group for p in pairs} == {"abjones", "amy"},
              str(sorted({p.group for p in pairs})))
        check("重取樣到 44.1 kHz",
              abs(pairs[0].samples - 3 * SAMPLE_RATE) < 100,
              f"{pairs[0].samples} 取樣")

        # 左右不能接反:人聲那一軌的主頻應該是 3000 Hz,不是 200 Hz
        def dominant(x: np.ndarray) -> float:
            mono = x[:, 0]
            spec = np.abs(np.fft.rfft(mono * np.hanning(len(mono))))
            return float(np.fft.rfftfreq(len(mono), 1 / SAMPLE_RATE)[np.argmax(spec)])

        check("人聲取自右聲道", abs(dominant(pairs[0].vocals) - 3000) < 30,
              f"{dominant(pairs[0].vocals):.0f} Hz")
        check("伴奏取自左聲道",
              abs(dominant(pairs[0].accompaniment) - 200) < 30,
              f"{dominant(pairs[0].accompaniment):.0f} Hz")


def test_onnx_export() -> None:
    print("ONNX 匯出")
    try:
        import onnxruntime  # noqa: F401
    except ImportError:
        print("  [skip] 未安裝 onnxruntime(uv sync --extra onnx)")
        return

    from ktisv_research.export import (aligned_length, export, verify,
                                       verify_onnx)
    from ktisv_research.model import PRESETS, build
    from ktisv_research.train import save_checkpoint

    import argparse

    config = PRESETS["tiny"]
    length = aligned_length(2.0, config)
    frames = length // config.hop_length + 1
    check("片段長度讓幀數被 2^depth 整除",
          frames % (2 ** config.depth) == 0,
          f"{frames} 幀 / 2^{config.depth}")

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        model = build("tiny")
        optimizer = torch.optim.Adam(model.parameters())
        args = argparse.Namespace(preset="tiny")
        save_checkpoint(root / "m.pt", model, optimizer, 0, 0.0, args)

        errors = verify(root / "m.pt", seconds=2.0)
        check("卷積版 STFT 與 torch.stft 一致",
              errors["stft_max_error"] < 1e-3, f"{errors['stft_max_error']:.2e}")
        check("卷積版 iSTFT 往返無損",
              errors["istft_roundtrip_error"] < 1e-4,
              f"{errors['istft_roundtrip_error']:.2e}")

        path, exported_length = export(root / "m.pt", root / "m.onnx",
                                       seconds=2.0)
        check("匯出的長度與 aligned_length 相同", exported_length == length)

        onnx_errors = verify_onnx(root / "m.pt", path, exported_length)
        check("ONNX Runtime 與 PyTorch 輸出一致",
              onnx_errors["vocals_max_error"] < 1e-3,
              f"{onnx_errors['vocals_max_error']:.2e}")
        check("ONNX 輸出仍然可加",
              onnx_errors["additivity_error"] < 1e-4,
              f"{onnx_errors['additivity_error']:.2e}")

        import onnx as onnx_module

        proto = onnx_module.load(str(path))
        metadata = {p.key: p.value for p in proto.metadata_props}
        check("片段長度寫進了 metadata",
              metadata.get("segment_samples") == str(exported_length),
              str(metadata))


def main() -> int:
    for fn in (test_split, test_dataset, test_determinism,
               test_mir1k_layout, test_onnx_export):
        fn()
        print()
    if FAILURES:
        print(f"{len(FAILURES)} 項未通過: " + ", ".join(FAILURES))
        return 1
    print("全部通過。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
