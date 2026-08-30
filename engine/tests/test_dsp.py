"""離線 DSP 測試 —— 不開任何音訊裝置,直接驅動混音核心。

    python -m tests.test_dsp        (在 engine 目錄下執行)
"""

from __future__ import annotations

import sys

import numpy as np

from ktisv_engine import BLOCK_SIZE, SAMPLE_RATE
from ktisv_engine.audio.engine import AudioEngine
from ktisv_engine.dsp.delay import CEILING_DELAY_MS, INITIAL_DELAY_MS, DelayLine
from ktisv_engine.dsp.eq import GraphicEQ
from ktisv_engine.dsp.ring import RingBuffer
from ktisv_engine.dsp.separation import CenterSeparator

FAILURES: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    mark = "ok  " if condition else "FAIL"
    print(f"  [{mark}] {name}{('  — ' + detail) if detail else ''}")
    if not condition:
        FAILURES.append(name)


def db(x: np.ndarray) -> float:
    rms = float(np.sqrt(np.mean(np.square(x))))
    return 20.0 * np.log10(max(rms, 1e-12))


def make_song(seconds: float = 4.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """做一首假歌:置中的 1 kHz「人聲」+ 左右分開的 300 Hz / 5 kHz「樂器」。"""
    t = np.arange(int(seconds * SAMPLE_RATE)) / SAMPLE_RATE
    vocal = 0.35 * np.sin(2 * np.pi * 1000 * t)          # 完全置中
    bass = 0.30 * np.sin(2 * np.pi * 80 * t)             # 置中低頻(應被保留)
    left_inst = 0.30 * np.sin(2 * np.pi * 300 * t)
    right_inst = 0.30 * np.sin(2 * np.pi * 5000 * t)

    mix = np.column_stack([vocal + bass + left_inst, vocal + bass + right_inst])
    return mix.astype(np.float32), vocal.astype(np.float32), \
        np.column_stack([bass + left_inst, bass + right_inst]).astype(np.float32)


def band_energy(x: np.ndarray, f0: float, width: float = 40.0) -> float:
    """單聲道訊號在 f0 附近的能量(dB)。"""
    mono = x[:, 0] if x.ndim > 1 else x
    n = len(mono)
    spec = np.abs(np.fft.rfft(mono * np.hanning(n)))
    freqs = np.fft.rfftfreq(n, 1.0 / SAMPLE_RATE)
    sel = (freqs > f0 - width) & (freqs < f0 + width)
    return 20.0 * np.log10(max(float(np.sum(spec[sel])), 1e-12))


# ── 測試 ────────────────────────────────────────────────────────────────
def test_ring() -> None:
    print("環形緩衝")
    ring = RingBuffer(1000, 2)
    data = np.arange(600, dtype=np.float32).reshape(300, 2)
    ring.write(data)
    check("寫入後可用量正確", ring.available() == 300, f"{ring.available()}")
    out = ring.read(100)
    check("讀出的資料一致", np.allclose(out, data[:100]))
    check("讀取後可用量遞減", ring.available() == 200)
    ring.read(500)
    check("讀取不足時補零並計數", ring.underflows == 1)
    ring.write(np.ones((1200, 2), dtype=np.float32))
    check("溢位會丟舊資料並計數", ring.overflows == 1 and ring.available() == 1000)


def test_eq() -> None:
    print("等化器")
    eq = GraphicEQ(SAMPLE_RATE, 1)
    t = np.arange(SAMPLE_RATE) / SAMPLE_RATE
    tone_1k = (0.5 * np.sin(2 * np.pi * 1000 * t)).astype(np.float32).reshape(-1, 1)

    flat = eq.process(tone_1k.copy())
    check("全平時旁通", np.allclose(flat, tone_1k))

    idx = list(eq.bands).index(1000.0)
    eq.set_gain(idx, 12.0)
    boosted = eq.process(tone_1k.copy())
    gain = db(boosted[SAMPLE_RATE // 2:]) - db(tone_1k[SAMPLE_RATE // 2:])
    check("1 kHz +12 dB 生效", 9.0 < gain < 13.0, f"實測 {gain:+.1f} dB")

    eq.reset()
    eq.set_gain(idx, -12.0)
    cut = eq.process(tone_1k.copy())
    gain = db(cut[SAMPLE_RATE // 2:]) - db(tone_1k[SAMPLE_RATE // 2:])
    check("1 kHz -12 dB 生效", -13.0 < gain < -9.0, f"實測 {gain:+.1f} dB")

    resp = eq.response(np.array([1000.0]))
    check("響應曲線與實測相符", abs(float(resp[0]) - gain) < 2.0,
          f"曲線 {float(resp[0]):+.1f} dB")


def test_eq_bands() -> None:
    print("等化器 —— 自訂頻段")
    from ktisv_engine.dsp.eq import MAX_BANDS

    eq = GraphicEQ(SAMPLE_RATE, 1)
    before = eq.band_count

    index = eq.add_band(1500.0, gain=6.0, q=4.0)
    check("新增的頻段依頻率插在正確位置", eq.bands[index] == 1500.0
          and eq.band_count == before + 1, f"索引 {index}")
    check("兩端仍然是 shelf",
          eq.band_info()[0]["type"] == "low_shelf"
          and eq.band_info()[-1]["type"] == "high_shelf")
    check("中間的是 peaking", eq.band_info()[index]["type"] == "peaking")

    resp = eq.response(np.array([1500.0]))
    check("新頻段的增益反映在響應上", 4.0 < float(resp[0]) < 7.0,
          f"{float(resp[0]):+.1f} dB")

    eq.remove_band(index)
    check("刪除後段數回復", eq.band_count == before and 1500.0 not in eq.bands)

    # 改頻率:峰值要跟著搬家,而不是留在原處
    target = list(eq.bands).index(1000.0)
    eq.set_band(target, gain=12.0)
    eq.set_band(target, freq=3000.0)
    moved = eq.response(np.array([1000.0, 3000.0]))
    check("改變頻率後峰值跟著移動", moved[1] > moved[0] + 6.0,
          f"1 kHz {moved[0]:+.1f} dB / 3 kHz {moved[1]:+.1f} dB")

    # Q 值:同樣的增益,Q 越大裙襬越窄
    target = list(eq.bands).index(3000.0)
    eq.set_band(target, q=0.5)
    wide = float(eq.response(np.array([1500.0]))[0])
    eq.set_band(target, q=8.0)
    narrow = float(eq.response(np.array([1500.0]))[0])
    check("Q 值影響頻寬", wide > narrow + 3.0,
          f"Q=0.5 時 {wide:+.1f} dB / Q=8 時 {narrow:+.1f} dB")

    # 增刪頻段不能把濾波器狀態洗掉 —— 洗掉就是一聲爆音
    eq2 = GraphicEQ(SAMPLE_RATE, 1)
    eq2.set_gain(list(eq2.bands).index(1000.0), 12.0)
    t = np.arange(BLOCK_SIZE * 200) / SAMPLE_RATE
    tone = (0.5 * np.sin(2 * np.pi * 1000 * t)).astype(np.float32).reshape(-1, 1)
    out = []
    for i in range(200):
        if i == 100:
            eq2.add_band(7000.0)          # 結構在播放中被改動
        out.append(eq2.process(tone[i * BLOCK_SIZE:(i + 1) * BLOCK_SIZE].copy()))
    stream = np.concatenate(out)[:, 0]
    seam = float(np.max(np.abs(np.diff(stream[99 * BLOCK_SIZE:101 * BLOCK_SIZE]))))
    steady = float(np.max(np.abs(np.diff(stream[150 * BLOCK_SIZE:]))))
    check("播放中增刪頻段不爆音", seam < steady * 1.5,
          f"接縫 {seam:.4f} vs 穩態 {steady:.4f}")

    check("段數有上限", eq.band_count <= MAX_BANDS)

    # Cookbook 的 shelf 公式在「Q 大 + 增益深」時根號會變負,沒擋住就是
    # 一整條 NaN、完全沒聲音。兩端各掃一次極端值。
    extreme = GraphicEQ(SAMPLE_RATE, 1)
    finite = True
    for gain in (-15.0, 15.0):
        for q in (0.1, 1.41, 8.0, 18.0):
            extreme.set_band(0, gain=gain, q=q)
            extreme.set_band(extreme.band_count - 1, gain=gain, q=q)
            finite &= bool(np.all(np.isfinite(extreme.response(
                np.array([30.0, 1000.0, 16000.0])))))
    check("極端的 shelf Q 與增益不會算出 NaN", finite)

    try:
        eq.set_bands([1000.0])
        eq.remove_band(0)
        check("不能把頻段刪光", False)
    except ValueError:
        check("不能把頻段刪光", True)


def test_pitch() -> None:
    print("變調(升 key / 降 key)")
    from ktisv_engine.dsp.pitch import PitchShifter

    def run(shifter: PitchShifter, signal: np.ndarray) -> np.ndarray:
        blocks = len(signal) // BLOCK_SIZE
        return np.concatenate([
            shifter.process(signal[i * BLOCK_SIZE:(i + 1) * BLOCK_SIZE].copy())
            for i in range(blocks)])

    def dominant(x: np.ndarray) -> float:
        """訊號裡最強的那個頻率(Hz)。"""
        mono = x[:, 0]
        spec = np.abs(np.fft.rfft(mono * np.hanning(len(mono))))
        return float(np.fft.rfftfreq(len(mono), 1.0 / SAMPLE_RATE)[np.argmax(spec)])

    t = np.arange(BLOCK_SIZE * 1200) / SAMPLE_RATE
    tone = np.repeat((0.4 * np.sin(2 * np.pi * 440 * t))
                     .astype(np.float32)[:, None], 2, axis=1)

    shifter = PitchShifter(SAMPLE_RATE, 2)
    check("原調時完全旁通", np.array_equal(run(shifter, tone), tone))

    # 半音是 2^(1/12);量到的頻率必須落在正確的音上,而不只是「有變高」
    worst = 0.0
    for semitones in (-12, -5, -2, 2, 5, 7, 12):
        shifter = PitchShifter(SAMPLE_RATE, 2)
        shifter.semitones = semitones
        out = run(shifter, tone)
        tail = out[len(out) // 2:]          # 前半段還在暖機
        want = 440.0 * 2.0 ** (semitones / 12.0)
        cents = 1200.0 * np.log2(dominant(tail) / want)
        worst = max(worst, abs(cents))
    check("各個半音的音高都對", worst < 12.0, f"最大誤差 {worst:.1f} 音分")

    # 速度不能跟著變 —— 這正是不能只用重新取樣的原因。
    #
    # 量的是兩段聲音之間的**間隔**,不是它們的絕對位置:整條路徑本來就有
    # 數十毫秒的演算法延遲,拿絕對位置比會把延遲誤判成速度跑掉。間隔則
    # 不受延遲影響,速度一變就立刻現形。
    gap = BLOCK_SIZE * 400
    burst = np.zeros((BLOCK_SIZE * 1400, 2), dtype=np.float32)
    burst[BLOCK_SIZE * 200:BLOCK_SIZE * 260] = tone[:BLOCK_SIZE * 60]
    burst[BLOCK_SIZE * 200 + gap:BLOCK_SIZE * 260 + gap] = tone[:BLOCK_SIZE * 60]

    def onsets(signal: np.ndarray) -> list[int]:
        """訊號由靜音轉為有聲的位置。

        取包絡而不是原始取樣 —— 正弦波每個週期都會經過零點,直接看取樣會把
        每一次過零都當成一次起音。起音那幾毫秒包絡還會在門檻上下抖,所以再
        把靠得很近的合併成同一次。
        """
        step = 64
        count = len(signal) // step
        envelope = np.abs(signal[:count * step, 0]).reshape(count, step).max(axis=1)
        loud = envelope > 0.05
        merged: list[int] = []
        for index in np.flatnonzero(loud[1:] & ~loud[:-1]) + 1:
            position = int(index) * step
            if not merged or position - merged[-1] > SAMPLE_RATE // 20:
                merged.append(position)
        return merged

    # 速度守恆要看的是「偏移會不會累積」,不是單一次的絕對誤差。
    #
    # 相位聲碼器以 hop 為單位吐資料,起音落在哪一格帶有 ±1~2 個 hop 的
    # 量化誤差(87.5% 重疊下一個 hop 是 5.3 ms)。那個誤差是固定的,不會
    # 隨曲子變長而變大;真正的速度漂移才會等比例放大。所以量兩種間隔,
    # 比的是兩者的差 —— 這樣量到的才是速度本身。
    def spacing_drift(shifter_semitones: int, gap_blocks: int) -> float | None:
        total = gap_blocks + 600
        long_tone = np.repeat(
            (0.4 * np.sin(2 * np.pi * 440
                          * np.arange(BLOCK_SIZE * total) / SAMPLE_RATE))
            .astype(np.float32)[:, None], 2, axis=1)
        span = BLOCK_SIZE * gap_blocks
        signal = np.zeros((BLOCK_SIZE * total, 2), dtype=np.float32)
        signal[BLOCK_SIZE * 200:BLOCK_SIZE * 260] = long_tone[:BLOCK_SIZE * 60]
        signal[BLOCK_SIZE * 200 + span:BLOCK_SIZE * 260 + span] =             long_tone[:BLOCK_SIZE * 60]
        shifter = PitchShifter(SAMPLE_RATE, 2)
        shifter.semitones = shifter_semitones
        out = run(shifter, signal)[:, 0]

        # 用能量重心而不是門檻過線。向下變調的拖尾會讓包絡遲遲降不到門檻
        # 以下,threshold 法會漏抓第二次起音;重心對拖尾與門檻都不敏感,
        # 而兩個爆音的拖尾形狀一樣,所以相減之後那部分自然抵消。
        energy = out.astype(np.float64) ** 2
        middle = min(len(energy), BLOCK_SIZE * 200 + span // 2)

        def centroid(seg_start: int, seg_end: int) -> float | None:
            seg = energy[seg_start:seg_end]
            total = float(np.sum(seg))
            if total < 1e-9:
                return None
            return seg_start + float(np.dot(np.arange(len(seg)), seg)) / total

        first = centroid(0, middle)
        second = centroid(middle, len(energy))
        if first is None or second is None:
            return None
        return ((second - first) - span) / SAMPLE_RATE * 1000.0

    for semitones in (-5, 5):
        near = spacing_drift(semitones, 400)      # 約 1.1 秒
        far = spacing_drift(semitones, 1600)      # 約 4.3 秒,四倍長
        if near is None or far is None:
            check(f"{semitones:+d} 半音時速度不變", False, "起音偵測失敗")
            continue
        # 真的變速的話,四倍長度會放大成四倍偏移;固定的量化誤差不會。
        accumulation = abs(far - near)
        check(f"{semitones:+d} 半音時速度不會漂移",
              accumulation < 8.0,
              f"1.1 秒偏 {near:+.1f} ms、4.3 秒偏 {far:+.1f} ms,"
              f"沒有累積(差 {accumulation:.1f} ms)")

    # 音量不能因為變調而改變(重疊相加的正規化算錯就會)
    shifter = PitchShifter(SAMPLE_RATE, 2)
    shifter.semitones = 4
    out = run(shifter, tone)
    level = db(out[len(out) // 2:]) - db(tone[len(tone) // 2:])
    check("變調不改變音量", abs(level) < 2.0, f"{level:+.1f} dB")

    # 立體聲的兩邊要各自處理,不能被混成單聲道
    stereo = np.zeros((BLOCK_SIZE * 1200, 2), dtype=np.float32)
    stereo[:, 0] = 0.4 * np.sin(2 * np.pi * 300 * t)
    stereo[:, 1] = 0.4 * np.sin(2 * np.pi * 900 * t)
    shifter = PitchShifter(SAMPLE_RATE, 2)
    shifter.semitones = 12
    out = run(shifter, stereo)[BLOCK_SIZE * 600:]
    left = dominant(out)
    right = dominant(out[:, ::-1])
    check("左右聲道各自獨立", abs(left - 600) < 20 and abs(right - 1800) < 20,
          f"左 {left:.0f} Hz / 右 {right:.0f} Hz")

    # 演奏中改 key:不能爆音,也不能卡住不出聲
    shifter = PitchShifter(SAMPLE_RATE, 2)
    shifter.semitones = 2
    out = []
    for i in range(1200):
        if i == 800:
            shifter.semitones = -3
        out.append(shifter.process(tone[i * BLOCK_SIZE:(i + 1) * BLOCK_SIZE].copy()))
    stream = np.concatenate(out)[:, 0]
    after = stream[1000 * BLOCK_SIZE:]
    check("播放中改 key 之後還有聲音",
          float(np.max(np.abs(after))) > 0.1, f"峰值 {float(np.max(np.abs(after))):.3f}")
    check("播放中改 key 不爆音",
          float(np.max(np.abs(np.diff(stream[790 * BLOCK_SIZE:830 * BLOCK_SIZE]))))
          < float(np.max(np.abs(np.diff(after)))) * 2.0)

    check("原調時不計入延遲", PitchShifter(SAMPLE_RATE, 2).latency_samples == 0)
    shifter = PitchShifter(SAMPLE_RATE, 2)
    shifter.semitones = 3
    check("有回報變調的延遲", 20.0 < shifter.latency_ms < 60.0,
          f"{shifter.latency_ms:.1f} ms")


def test_echo() -> None:
    print("麥克風回音")
    from ktisv_engine.dsp.echo import Echo

    def run(fx: Echo, signal: np.ndarray) -> np.ndarray:
        blocks = len(signal) // BLOCK_SIZE
        return np.concatenate([
            fx.process(signal[i * BLOCK_SIZE:(i + 1) * BLOCK_SIZE].copy())
            for i in range(blocks)])

    pulse = np.zeros((BLOCK_SIZE * 400, 1), dtype=np.float32)
    pulse[0] = 1.0

    echo = Echo(SAMPLE_RATE, 1)
    check("預設關閉時完全旁通", np.array_equal(run(echo, pulse), pulse))

    echo = Echo(SAMPLE_RATE, 1)
    echo.enabled = True
    echo.delay_ms = 100.0
    echo.feedback = 0.5
    echo.mix = 1.0
    echo.damping = 0.0
    out = run(echo, pulse)[:, 0]

    step = int(0.100 * SAMPLE_RATE)
    taps = [float(out[i * step]) for i in range(4)]
    check("重複出現在正確的時間點,而且逐次衰減",
          abs(taps[0] - 1.0) < 1e-3 and abs(taps[1] - 1.0) < 1e-3
          and abs(taps[2] - 0.5) < 1e-3 and abs(taps[3] - 0.25) < 1e-3,
          " / ".join(f"{v:.3f}" for v in taps))

    echo = Echo(SAMPLE_RATE, 1)
    echo.enabled = True
    echo.delay_ms = 100.0
    echo.feedback = 0.0
    echo.mix = 1.0
    echo.damping = 0.0
    out = run(echo, pulse)[:, 0]
    check("回授為 0 時只重複一次",
          abs(float(out[step]) - 1.0) < 1e-3 and abs(float(out[2 * step])) < 1e-3)

    # 阻尼讓每一次重複都比前一次暗,否則高頻會一路疊到刺耳。
    # 只餵前一小段噪音,後面留白 —— 乾訊號會蓋過尾音,量不到差別。
    burst = np.zeros((BLOCK_SIZE * 400, 1), dtype=np.float32)
    burst[:BLOCK_SIZE * 50] = \
        np.random.RandomState(1).randn(BLOCK_SIZE * 50, 1).astype(np.float32) * 0.2
    bright = Echo(SAMPLE_RATE, 1)
    dark = Echo(SAMPLE_RATE, 1)
    for fx, damping in ((bright, 0.0), (dark, 0.8)):
        fx.enabled = True
        fx.delay_ms = 50.0
        fx.feedback = 0.7
        fx.mix = 1.0
        fx.damping = damping
    tail = slice(BLOCK_SIZE * 100, BLOCK_SIZE * 250)
    high_bright = band_energy(run(bright, burst)[tail], 12000.0, width=2000.0)
    high_dark = band_energy(run(dark, burst)[tail], 12000.0, width=2000.0)
    check("阻尼壓掉尾音的高頻", high_dark < high_bright - 6.0,
          f"{high_bright:.1f} → {high_dark:.1f} dB")

    # 拖動時間旋鈕時不能爆音。基準取同一條輸出的穩態抖動 —— 拿乾訊號比
    # 會誤判,加了尾音的波形本來就比原訊號抖。
    t = np.arange(BLOCK_SIZE * 400) / SAMPLE_RATE
    tone = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32).reshape(-1, 1)
    echo = Echo(SAMPLE_RATE, 1)
    echo.enabled = True
    echo.delay_ms = 200.0
    echo.feedback = 0.4
    echo.mix = 0.5
    out = []
    for i in range(400):
        if i == 300:
            echo.delay_ms = 60.0
        out.append(echo.process(tone[i * BLOCK_SIZE:(i + 1) * BLOCK_SIZE].copy()))
    stream = np.concatenate(out)[:, 0]
    seam = float(np.max(np.abs(np.diff(stream[299 * BLOCK_SIZE:301 * BLOCK_SIZE]))))
    steady = float(np.max(np.abs(np.diff(stream[200 * BLOCK_SIZE:280 * BLOCK_SIZE]))))
    check("改變回音時間不爆音", seam < steady * 1.5,
          f"接縫 {seam:.4f} vs 穩態 {steady:.4f}")

    echo.enabled = False
    echo.reset()
    check("關閉後尾音被清乾淨",
          float(np.max(np.abs(echo.process(np.zeros((BLOCK_SIZE, 1),
                                                    dtype=np.float32))))) == 0.0)


def test_separation() -> None:
    print("即時中央聲道分離")
    mix, _, _ = make_song()

    sep = CenterSeparator(SAMPLE_RATE, low_cut=180.0, high_cut=9000.0)
    sep.mode = "remove_vocals"
    removed = _run_blocks(sep, mix)

    tail = slice(SAMPLE_RATE, None)
    vocal_before = band_energy(mix[tail], 1000.0)
    vocal_after = band_energy(removed[tail], 1000.0)
    check("人聲(1 kHz 置中)被壓下去", vocal_after < vocal_before - 25.0,
          f"{vocal_before:.1f} → {vocal_after:.1f} dB")

    bass_before = band_energy(mix[tail], 80.0)
    bass_after = band_energy(removed[tail], 80.0)
    check("置中低頻(80 Hz 貝斯)被保留", bass_after > bass_before - 4.0,
          f"{bass_before:.1f} → {bass_after:.1f} dB")

    # 硬左右分離的樂器有一半能量落在 mid,mid/side 相消必然會少掉 6 dB。
    # 這是演算法的固有代價,不是缺陷 —— 只要求不低於此。
    inst_before = band_energy(mix[tail], 300.0)
    inst_after = band_energy(removed[tail], 300.0)
    check("非置中樂器(300 Hz)大致保留", inst_after > inst_before - 8.0,
          f"{inst_before:.1f} → {inst_after:.1f} dB")

    sep2 = CenterSeparator(SAMPLE_RATE)
    sep2.mode = "isolate_vocals"
    isolated = _run_blocks(sep2, mix)
    iso_vocal = band_energy(isolated[tail], 1000.0)
    iso_inst = band_energy(isolated[tail], 300.0)
    check("取出人聲時 1 kHz 仍在", iso_vocal > vocal_before - 8.0,
          f"{iso_vocal:.1f} dB")
    # 同理:取出人聲只能把側向樂器壓掉一半(約 6 dB),
    # 要乾淨的人聲軌得用 Demucs。
    ratio_before = vocal_before - inst_before
    ratio_after = iso_vocal - iso_inst
    check("取出人聲時人聲/樂器比改善", ratio_after > ratio_before + 3.0,
          f"{ratio_before:.1f} → {ratio_after:.1f} dB")

    sep3 = CenterSeparator(SAMPLE_RATE)
    sep3.mode = "off"
    check("關閉時完全旁通", np.allclose(_run_blocks(sep3, mix), mix))


def _run_blocks(sep: CenterSeparator, audio: np.ndarray) -> np.ndarray:
    out = np.empty_like(audio)
    for start in range(0, len(audio) - BLOCK_SIZE + 1, BLOCK_SIZE):
        block = audio[start:start + BLOCK_SIZE]
        out[start:start + BLOCK_SIZE] = sep.process(block)
    return out


def test_engine_mix() -> None:
    print("混音引擎(離線)")
    engine = AudioEngine()
    mix, vocals, instrumental = make_song(2.0)

    # --- Demucs 模式:兩個分軌 + 勾選框 ---
    engine.player.load({"vocals": np.column_stack([vocals, vocals]),
                        "instrumental": instrumental}, title="test")
    engine.player.play()
    engine.params.send_music_hp.snap(1.0)
    engine.params.send_music_vc.snap(1.0)
    engine.params.master_hp.snap(1.0)
    engine.params.master_vc.snap(1.0)
    engine.params.music_fader.snap(1.0)

    engine.apply_separation_flags(False, False, realtime=False)
    full = _pump(engine, 1.0)
    check("兩軌都在時有聲音", db(full) > -30.0, f"{db(full):.1f} dBFS")

    engine.player.seek(0.0)
    engine.apply_separation_flags(True, False, realtime=False)
    no_vocal = _pump(engine, 1.0)
    check("勾「去人聲」後 1 kHz 消失",
          band_energy(no_vocal, 1000.0) < band_energy(full, 1000.0) - 30.0)
    check("勾「去人聲」後伴奏還在",
          band_energy(no_vocal, 300.0) > band_energy(full, 300.0) - 4.0)

    engine.player.seek(0.0)
    engine.apply_separation_flags(False, True, realtime=False)
    no_inst = _pump(engine, 1.0)
    check("勾「去伴奏」後只剩人聲",
          band_energy(no_inst, 300.0) < band_energy(full, 300.0) - 30.0
          and band_energy(no_inst, 1000.0) > band_energy(full, 1000.0) - 4.0)

    engine.player.seek(0.0)
    engine.apply_separation_flags(True, True, realtime=False)
    both = _pump(engine, 0.5)
    tail_only = both[len(both) // 2:]   # 前半段是淡出尾巴
    check("兩個都勾 → 靜音", db(tail_only) < -70.0, f"{db(tail_only):.1f} dBFS")

    # --- 即時模式:單軌 + 中央消除 ---
    engine.player.load({"mix": mix}, title="test-rt")
    engine.player.play()
    engine.apply_separation_flags(True, False, realtime=True)
    rt = _pump(engine, 1.0)
    check("即時模式勾「去人聲」有作用", engine.separator.mode == "remove_vocals")
    check("即時模式仍保有伴奏", db(rt) > -30.0, f"{db(rt):.1f} dBFS")

    # --- 麥克風監聽路徑 ---
    engine.player.pause()
    _pump(engine, 0.1)   # 讓傳輸淡出與濾波器暫態走完
    t = np.arange(BLOCK_SIZE * 50) / SAMPLE_RATE
    mic_tone = (0.4 * np.sin(2 * np.pi * 440 * t)).astype(np.float32).reshape(-1, 1)
    engine._mic_ring.write(mic_tone)
    engine._mic_stream = object()          # 讓 _produce 認為麥克風已開
    engine.params.send_mic_vc.snap(1.0)
    engine.params.send_mic_monitor.snap(1.0)
    engine.params.mic_fader.snap(1.0)

    engine.params.monitor_self = False
    _prime(engine, BLOCK_SIZE * 10)
    engine._mic_ring.write(mic_tone)
    hp, vc = engine._produce(BLOCK_SIZE * 10)
    check("未勾監聽時耳機聽不到自己", db(hp) < -70.0, f"{db(hp):.1f} dBFS")
    check("麥克風有進到虛擬音效卡", db(vc) > -30.0, f"{db(vc):.1f} dBFS")

    engine._mic_ring.write(mic_tone)
    engine.params.monitor_self = True
    _prime(engine, BLOCK_SIZE * 10)
    engine._mic_ring.write(mic_tone)
    hp, _ = engine._produce(BLOCK_SIZE * 10)
    check("勾了監聽就聽得到自己", db(hp) > -30.0, f"{db(hp):.1f} dBFS")

    # --- 靜音與限幅 ---
    engine._mic_ring.write(mic_tone)
    engine.params.mic_muted = True
    _prime(engine, BLOCK_SIZE * 10)
    engine._mic_ring.write(mic_tone)
    _, vc = engine._produce(BLOCK_SIZE * 10)
    check("麥克風靜音生效", db(vc) < -70.0, f"{db(vc):.1f} dBFS")
    engine.params.mic_muted = False

    loud = np.ones((BLOCK_SIZE, 2), dtype=np.float32) * 4.0
    engine.player.load({"mix": loud}, title="loud")
    engine.player.play()
    engine.apply_separation_flags(False, False, realtime=True)
    engine.params.music_fader.snap(1.0)
    _prime(engine)
    hp, _ = engine._produce(BLOCK_SIZE)
    check("軟限幅把峰值壓在 0 dBFS 內", float(np.max(np.abs(hp))) <= 1.0,
          f"峰值 {float(np.max(np.abs(hp))):.3f}")

    snapshot = engine.meter_snapshot()
    check("電平表有回報數值", set(snapshot) >= {"music_out", "mic_out", "hp_out", "vc_out"})


def test_engine_pitch() -> None:
    print("變調接進混音引擎")
    engine = AudioEngine()
    for name in ("send_music_hp", "send_music_vc", "master_hp", "master_vc",
                 "music_fader", "mic_fader", "send_mic_vc", "send_mic_monitor"):
        getattr(engine.params, name).snap(1.0)

    t = np.arange(SAMPLE_RATE * 6) / SAMPLE_RATE
    music = np.repeat((0.4 * np.sin(2 * np.pi * 440 * t))
                      .astype(np.float32)[:, None], 2, axis=1)
    engine.player.load({"mix": music}, title="pitch")
    engine.player.play()

    def dominant(x: np.ndarray) -> float:
        mono = x[:, 0]
        spec = np.abs(np.fft.rfft(mono * np.hanning(len(mono))))
        return float(np.fft.rfftfreq(len(mono), 1.0 / SAMPLE_RATE)[np.argmax(spec)])

    engine.set_music_pitch(5)
    hp = _pump(engine, 3.0)
    heard = dominant(hp[len(hp) // 2:])
    want = 440.0 * 2.0 ** (5 / 12.0)
    check("耳機聽到的音樂升了 5 個半音", abs(heard - want) < 8.0,
          f"{heard:.0f} Hz(期望 {want:.0f} Hz)")

    # 送給 Discord 的那一路必須是同一份 —— 兩邊唱的不是同一個調就沒得合了
    engine.player.seek(0.0)
    _pump(engine, 0.5)
    vc_blocks = []
    for _ in range(int(2.0 * SAMPLE_RATE / BLOCK_SIZE)):
        _, vc = engine._produce(BLOCK_SIZE)
        vc_blocks.append(vc.copy())
    vc = np.concatenate(vc_blocks)
    check("虛擬麥克風那一路也是同一個調",
          abs(dominant(vc[len(vc) // 2:]) - want) < 8.0,
          f"{dominant(vc[len(vc) // 2:]):.0f} Hz")

    # 麥克風不能跟著被變調 —— 變調是為了讓歌配合你,不是把你的聲音改掉
    engine.params.music_fader.snap(0.0)
    engine._mic_stream = object()
    mic_tone = (0.4 * np.sin(2 * np.pi * 440
                             * (np.arange(SAMPLE_RATE * 2) / SAMPLE_RATE))
                ).astype(np.float32).reshape(-1, 1)
    engine._mic_ring.write(mic_tone)
    engine.params.monitor_self = True
    hp = _pump(engine, 1.0)
    check("你自己的聲音沒有被變調",
          abs(dominant(hp[len(hp) // 2:]) - 440.0) < 8.0,
          f"{dominant(hp[len(hp) // 2:]):.0f} Hz")

    engine.set_music_pitch(0)
    check("回到原調時整條旁通", engine.music_pitch.latency_samples == 0)


def _pump(engine: AudioEngine, seconds: float) -> np.ndarray:
    _prime(engine)
    blocks = int(seconds * SAMPLE_RATE / BLOCK_SIZE)
    out = []
    for _ in range(blocks):
        hp, _vc = engine._produce(BLOCK_SIZE)
        out.append(hp.copy())
    return np.concatenate(out) if out else np.zeros((0, 2), dtype=np.float32)


def test_smooth_gain() -> None:
    print("增益平滑")
    from ktisv_engine.dsp.gain import SmoothGain, db_to_lin

    g = SmoothGain(0.0, 20.0, SAMPLE_RATE)
    g.target = 1.0
    env = np.concatenate([g.envelope(BLOCK_SIZE) for _ in range(20)]).ravel()
    steps = np.abs(np.diff(env))
    check("沒有瞬間跳變", float(np.max(steps)) < 0.02, f"最大單步 {float(np.max(steps)):.4f}")
    check("最終收斂到目標", abs(float(env[-1]) - 1.0) < 0.05, f"{float(env[-1]):.3f}")
    check("-60 dB 視為靜音", db_to_lin(-60.0) == 0.0)
    check("0 dB = 1.0", abs(db_to_lin(0.0) - 1.0) < 1e-9)


def test_delay() -> None:
    print("延遲線")
    from ktisv_engine.dsp.delay import DelayLine

    def run(line: DelayLine, signal: np.ndarray) -> np.ndarray:
        blocks = len(signal) // BLOCK_SIZE
        return np.concatenate([
            line.process(signal[i * BLOCK_SIZE:(i + 1) * BLOCK_SIZE].copy())
            for i in range(blocks)])

    # 延遲量必須精確到取樣,不然拿來對齊就沒有意義。
    # reset() 是 start() 會做的事:直接就位、不滑行。
    worst = 0
    for ms in (1.0, 10.0, 50.0, 200.0):
        line = DelayLine(SAMPLE_RATE, 2)
        line.delay_ms = ms
        line.reset()
        signal = np.zeros((BLOCK_SIZE * 300, 2), dtype=np.float32)
        signal[BLOCK_SIZE] = 1.0
        out = run(line, signal)
        want = BLOCK_SIZE + int(round(ms * SAMPLE_RATE / 1000.0))
        worst = max(worst, abs(int(np.argmax(np.abs(out[:, 0]))) - want))
    check("延遲量精確到取樣", worst == 0, f"最大誤差 {worst} 取樣")

    # 播放中改延遲會用滑行(避免電子音),但終點必須完全準確 ——
    # 滑到定位之後,量到的延遲要跟設定值一模一樣。
    line = DelayLine(SAMPLE_RATE, 2)
    filler = np.zeros((BLOCK_SIZE, 2), dtype=np.float32)
    line.delay_ms = 80.0
    blocks = 0
    while not line.settled and blocks < int(3 * SAMPLE_RATE / BLOCK_SIZE):
        line.process(filler.copy())
        blocks += 1
    check("滑行會在合理時間內到位", line.settled and blocks > 1,
          f"{blocks * BLOCK_SIZE / SAMPLE_RATE * 1000:.0f} ms")

    signal = np.zeros((BLOCK_SIZE * 300, 2), dtype=np.float32)
    signal[BLOCK_SIZE] = 1.0
    out = run(line, signal)
    want = BLOCK_SIZE + int(round(80.0 * SAMPLE_RATE / 1000.0))
    got = int(np.argmax(np.abs(out[:, 0])))
    check("滑行到位後的延遲精確", got == want,
          f"{(got - BLOCK_SIZE) / SAMPLE_RATE * 1000:.2f} ms(設定 80 ms)")

    # 這是修掉「調整時出現電子音」的核心:滑行期間輸出必須始終是一條
    # 連續波形,不能變成「訊號 + 自己的位移版」那種梳狀濾波。用包絡起伏
    # 量 —— 純音經過乾淨的延遲振幅是平的,梳狀凹陷會讓它忽大忽小。
    from scipy.signal import hilbert

    line = DelayLine(SAMPLE_RATE, 2)
    total = int(2.5 * SAMPLE_RATE / BLOCK_SIZE)
    t = np.arange(total * BLOCK_SIZE) / SAMPLE_RATE
    sweep_tone = np.repeat((0.4 * np.sin(2 * np.pi * 440 * t))
                           .astype(np.float32)[:, None], 2, axis=1)
    every = max(1, int(0.040 * SAMPLE_RATE / BLOCK_SIZE))   # 推桿約 40 ms 一次
    swept = []
    for i in range(total):
        if i % every == 0:
            line.delay_ms = 150.0 * min(1.0, i / (total * 0.5))
        swept.append(line.process(
            sweep_tone[i * BLOCK_SIZE:(i + 1) * BLOCK_SIZE].copy()))
    moving = np.concatenate(swept)[:, 0]
    moving = moving[int(len(moving) * 0.1):int(len(moving) * 0.5)]
    env = np.abs(hilbert(moving.astype(np.float64)))
    env = env[len(env) // 20:-len(env) // 20]
    wobble = float(np.percentile(env, 95) - np.percentile(env, 5))         / float(np.mean(env)) * 100
    check("調整過程中不產生梳狀失真", wobble < 2.0,
          f"包絡起伏 {wobble:.1f}%(舊的淡接做法約 4-5%)")

    line = DelayLine(SAMPLE_RATE, 2)
    block = np.random.RandomState(0).randn(BLOCK_SIZE, 2).astype(np.float32)
    check("延遲 0 時輸出等於輸入", np.array_equal(line.process(block.copy()), block))

    # 改變延遲時不能爆音 —— 讀取位置一跳,波形就接不起來,這是最容易漏的地方
    t = np.arange(BLOCK_SIZE * 400) / SAMPLE_RATE
    tone = np.repeat((0.5 * np.sin(2 * np.pi * 440 * t))
                     .astype(np.float32)[:, None], 2, axis=1)
    reference = float(np.max(np.abs(np.diff(tone[:, 0]))))

    line = DelayLine(SAMPLE_RATE, 2)
    line.delay_ms = 20.0
    out = []
    for i in range(400):
        if i == 200:
            line.delay_ms = 90.0          # 故意拉一個很大的跳變
        out.append(line.process(tone[i * BLOCK_SIZE:(i + 1) * BLOCK_SIZE].copy()))
    jump = float(np.max(np.abs(np.diff(np.concatenate(out)[BLOCK_SIZE:, 0]))))
    check("改變延遲不產生爆音", jump < reference * 1.5,
          f"最大跳變 {jump:.4f} vs 原訊號 {reference:.4f}")

    # 關閉時也要持續餵資料,否則開啟的瞬間會聽到一段空白
    line = DelayLine(SAMPLE_RATE, 2)
    for i in range(200):
        line.process(tone[i * BLOCK_SIZE:(i + 1) * BLOCK_SIZE].copy())
    line.delay_ms = 20.0
    first = line.process(tone[200 * BLOCK_SIZE:201 * BLOCK_SIZE].copy())
    peak = float(np.max(np.abs(first)))
    check("剛開啟時讀得到歷史音訊,不是靜音", peak > 0.1, f"峰值 {peak:.3f}")

    # ── 緩衝隨設定值長大 ───────────────────────────────────────────────
    # 呼叫端不必事先知道會用到多少延遲,所以設定超過初始配置量時緩衝要
    # 自己長 —— 而且長大之後延遲量仍然要準、歷史也不能弄丟。
    line = DelayLine(SAMPLE_RATE, 2, initial_ms=50.0)
    before = line.max_samples
    line.delay_ms = 900.0
    check("超過初始配置量會自己長大", line.max_samples >= int(0.9 * SAMPLE_RATE),
          f"{before} → {line.max_samples} 取樣")
    check("長大後回報的延遲量正確", abs(line.delay_ms - 900.0) < 0.05,
          f"{line.delay_ms:.2f} ms")

    line = DelayLine(SAMPLE_RATE, 2, initial_ms=50.0)
    line.delay_ms = 600.0
    line.reset()
    signal = np.zeros((BLOCK_SIZE * 700, 2), dtype=np.float32)
    signal[BLOCK_SIZE] = 1.0
    out = run(line, signal)
    want = BLOCK_SIZE + int(round(600.0 * SAMPLE_RATE / 1000.0))
    got = int(np.argmax(np.abs(out[:, 0])))
    check("長大後實際延遲正確", got == want,
          f"脈衝在 {(got - BLOCK_SIZE) / SAMPLE_RATE * 1000:.2f} ms")

    line = DelayLine(SAMPLE_RATE, 2, initial_ms=50.0)
    line.delay_ms = 10.0
    for i in range(300):
        line.process(tone[i * BLOCK_SIZE:(i + 1) * BLOCK_SIZE].copy())
    line.delay_ms = 800.0                      # 觸發成長
    kept = float(np.max(np.abs(line._linearized())))
    check("長大時保住既有的歷史音訊", kept > 0.1, f"歷史峰值 {kept:.3f}")


class _Pulse:
    """只在第一個 block 打一個脈衝的假播放器。"""

    def __init__(self) -> None:
        self.calls = 0

    def read(self, frames: int) -> np.ndarray:
        out = np.zeros((frames, 2), dtype=np.float32)
        if self.calls == 0:
            out[0] = 1.0
        self.calls += 1
        return out

    def state(self) -> dict:
        return {}


def _sync_engine(sync_ms: float) -> AudioEngine:
    """裝好脈衝音源、設定好對時、且已收斂的引擎。"""
    engine = AudioEngine(blocksize=BLOCK_SIZE)
    for name in ("send_music_hp", "send_music_vc", "send_mic_vc",
                 "master_hp", "master_vc", "music_fader", "mic_fader"):
        getattr(engine.params, name).snap(1.0)
    engine.player = _Pulse()
    engine.set_vc_sync_ms(sync_ms)
    # start() 會做這件事。少了它,延遲線會把「從 0 淡接到目標」當成一次即時
    # 調整 —— 那個淡接本身是對的(拖滑桿時才不會爆音),但會讓第一個 block
    # 漏出一點未延遲的訊號,不是這裡要量的穩態行為。
    engine._vc_music_delay.reset()
    engine._vc_mic_delay.reset()
    return engine


def _prime(engine: AudioEngine, frames: int = BLOCK_SIZE) -> None:
    """把限幅器的前瞻管線填滿。

    限幅器前瞻一個 block,所以第一次呼叫 _produce() 一定回靜音。實際使用
    時那只是開頭的 1-3 ms,但測試若沒先推一輪,量到的就全是那段靜音。
    """
    engine._produce(frames)


def _pump_both(engine: AudioEngine, blocks: int = 40):
    _prime(engine)
    hp_all, vc_all = [], []
    for _ in range(blocks):
        hp, vc = engine._produce(BLOCK_SIZE)
        hp_all.append(hp.copy())
        vc_all.append(vc.copy())
    return np.concatenate(hp_all), np.concatenate(vc_all)


def test_vc_sync() -> None:
    print("DC 對時(送給 Discord 那一路,可正可負)")

    # 正值:延後音樂。耳機不受影響。
    engine = _sync_engine(20.0)
    hp, vc = _pump_both(engine)
    hp_at = int(np.argmax(np.abs(hp[:, 0])))
    vc_at = int(np.argmax(np.abs(vc[:, 0])))
    check("正值不影響耳機", hp_at == 0, f"脈衝在第 {hp_at} 取樣")
    check("正值延後送出的音樂", vc_at == int(0.020 * SAMPLE_RATE),
          f"脈衝在 {vc_at / SAMPLE_RATE * 1000:.2f} ms")

    # 負值:音樂不動,改成延後麥克風。
    engine = _sync_engine(-20.0)
    tone = np.full((BLOCK_SIZE * 40, 1), 0.5, dtype=np.float32)
    engine._mic_ring.write(tone)
    engine._mic_stream = object()          # 讓 _produce 認為麥克風已開
    _, vc = _pump_both(engine)
    check("負值不延後音樂", int(np.argmax(np.abs(vc[:, 0]))) == 0,
          "音樂脈衝仍在第 0 取樣")

    # 麥克風那一路要真的晚 20 ms 才出現
    engine = _sync_engine(-20.0)
    engine._mic_ring.write(tone)
    engine._mic_stream = object()
    engine.params.send_music_vc.snap(0.0)  # 只留麥克風,才看得出它的起點
    _, vc = _pump_both(engine)
    onset = int(np.argmax(np.abs(vc[:, 0]) > 0.05))
    check("負值延後送出的歌聲", abs(onset - int(0.020 * SAMPLE_RATE)) <= BLOCK_SIZE,
          f"歌聲在 {onset / SAMPLE_RATE * 1000:.2f} ms 才出現")

    # 號誌切換時只有一條線在作用,絕對延遲就是 |值|
    engine = AudioEngine(blocksize=BLOCK_SIZE)
    engine.set_vc_sync_ms(30.0)
    check("正值只動音樂線",
          engine._vc_music_delay.delay_ms == 30.0
          and engine._vc_mic_delay.delay_ms == 0.0)
    engine.set_vc_sync_ms(-30.0)
    check("負值只動麥克風線",
          engine._vc_music_delay.delay_ms == 0.0
          and engine._vc_mic_delay.delay_ms == 30.0)
    # 沒有功能上的上限:遠超過預設配置量的值也要照單全收
    big = INITIAL_DELAY_MS * 6
    check("遠超過預設配置量的值照樣吃下",
          engine.set_vc_sync_ms(big) == big
          and engine._vc_music_delay.delay_ms == big,
          f"{big:.0f} ms")
    check("負方向同樣沒有上限",
          engine.set_vc_sync_ms(-big) == -big
          and engine._vc_mic_delay.delay_ms == big)
    check("只剩記憶體保險絲擋住手誤",
          engine.set_vc_sync_ms(9e9) == CEILING_DELAY_MS
          and engine.set_vc_sync_ms(-9e9) == -CEILING_DELAY_MS,
          f"夾在 ±{CEILING_DELAY_MS:.0f} ms")
    check("NaN 當成 0", engine.set_vc_sync_ms(float("nan")) == 0.0)


def test_monitor_send() -> None:
    print("耳機監聽送出訊號")

    # 對時 20 ms;開了監聽之後,耳機聽到的應該跟送出的一樣被延後
    engine = _sync_engine(20.0)
    engine.params.monitor_send = True
    hp, vc = _pump_both(engine)
    hp_at = int(np.argmax(np.abs(hp[:, 0])))
    vc_at = int(np.argmax(np.abs(vc[:, 0])))
    check("耳機跟著送出訊號一起被延後", hp_at == vc_at,
          f"耳機 {hp_at / SAMPLE_RATE * 1000:.2f} ms / 送出 "
          f"{vc_at / SAMPLE_RATE * 1000:.2f} ms")

    # 兩端內容相同,只差耳機主音量
    engine = _sync_engine(0.0)
    engine.params.monitor_send = True
    engine.params.master_hp.snap(1.0)
    hp, vc = _pump_both(engine)
    check("兩端是同一份訊號", np.allclose(hp, vc, atol=1e-6),
          f"最大差 {float(np.max(np.abs(hp - vc))):.2e}")

    # 關掉就回到本地混音(不含對時)
    engine = _sync_engine(20.0)
    engine.params.monitor_send = False
    hp, _ = _pump_both(engine)
    check("關掉之後耳機回到未對時的本地混音",
          int(np.argmax(np.abs(hp[:, 0]))) == 0)


def test_calibration() -> None:
    print("音遊式校準")
    from ktisv_engine.audio import calibrate

    clips, voiced = calibrate.count_clips(SAMPLE_RATE)
    check("節拍聲備得出來", len(clips) == 4 and all(len(c) for c in clips),
          "語音" if voiced else "合成音(沒有語音合成可用)")

    def fake_singer(cal, offset_ms, level=0.45, bleed=0.0, jitter_ms=0.0,
                    seed=0):
        """依 offset 造一段麥克風訊號,並實際推過校準器。"""
        rng = np.random.RandomState(seed)
        burst = int(SAMPLE_RATE * 0.15)
        env = np.exp(-np.arange(burst) / SAMPLE_RATE * 18.0)
        tone = np.sin(2 * np.pi * 220 * np.arange(burst) / SAMPLE_RATE)
        mic = np.zeros(cal.total + SAMPLE_RATE, dtype=np.float32)
        for position in cal.beat_positions:
            shift = rng.uniform(-jitter_ms, jitter_ms) if jitter_ms else 0.0
            at = position + int(round((offset_ms + shift) * SAMPLE_RATE / 1000.0))
            if at >= 0:
                mic[at:at + burst] += (tone * env * level).astype(np.float32)
        mic += rng.randn(len(mic)).astype(np.float32) * 0.002

        cursor = 0
        while not cal.finished:
            cue = cal.render(BLOCK_SIZE)
            block = mic[cursor:cursor + BLOCK_SIZE].copy()
            if len(block) < BLOCK_SIZE:
                block = np.pad(block, (0, BLOCK_SIZE - len(block)))
            if bleed:
                block = block + cue[:, 0] * bleed
            cal.observe(block.reshape(-1, 1), BLOCK_SIZE)
            cursor += BLOCK_SIZE
        return cal.result()

    # 正負兩個方向都要量得準 —— 會搶拍的人量出來就是負的
    worst = 0.0
    for offset in (0.0, 45.0, 150.0, -70.0, -200.0):
        cal = calibrate.BeatCalibrator(SAMPLE_RATE, clips, voiced)
        result = fake_singer(cal, offset)
        if not result.get("ok"):
            check(f"量得到 {offset:+.0f} ms", False, str(result.get("reason")))
            continue
        worst = max(worst, abs(result["offset_ms"] - offset))
    check("正負偏移都量得準", worst <= 5.0, f"最大誤差 {worst:.1f} ms")

    # 每一拍念的位置會有落差,取中位數才不會被個別一拍帶走
    cal = calibrate.BeatCalibrator(SAMPLE_RATE, clips, voiced)
    jittered = fake_singer(cal, 60.0, jitter_ms=30.0)
    check("人為抖動下仍然可用",
          jittered.get("ok") and abs(jittered["offset_ms"] - 60.0) <= 15.0,
          f"{jittered.get('offset_ms')} ms")

    # 耳機漏音:節拍聲自己漏回麥克風,不能被當成人聲起點
    cal = calibrate.BeatCalibrator(SAMPLE_RATE, clips, voiced)
    leaked = fake_singer(cal, 60.0, bleed=0.2)
    check("耳機漏音下仍量得準",
          leaked.get("ok") and abs(leaked["offset_ms"] - 60.0) <= 10.0,
          f"{leaked.get('offset_ms')} ms")

    # 漏音大到跟人聲同一個量級時,寧可標成不可信也不要給錯的數字
    cal = calibrate.BeatCalibrator(SAMPLE_RATE, clips, voiced)
    swamped = fake_singer(cal, 60.0, level=0.12, bleed=0.10)
    check("漏音蓋過人聲時會標成不可信",
          swamped.get("ok") and not swamped.get("reliable"),
          f"{swamped.get('offset_ms')} ms · bleed_risky={swamped.get('bleed_risky')}")

    # 完全沒出聲要講清楚,而不是回一個看起來正常的 0
    cal = calibrate.BeatCalibrator(SAMPLE_RATE, clips, voiced)
    silent = fake_singer(cal, 0.0, level=0.0)
    check("沒出聲時會回報失敗", not silent.get("ok"), str(silent.get("reason"))[:28])


def test_calibration_in_engine() -> None:
    print("校準接進混音引擎")
    from ktisv_engine.audio import calibrate

    engine = AudioEngine(blocksize=BLOCK_SIZE)
    engine.mic_device = engine.headphone_device = 0
    engine.params.master_hp.snap(1.0)
    engine._mic_stream = object()
    engine._running = True
    engine.start_calibration()
    cal = engine._calibrator

    offset_ms = 75.0
    burst = int(SAMPLE_RATE * 0.15)
    env = np.exp(-np.arange(burst) / SAMPLE_RATE * 18.0)
    tone = np.sin(2 * np.pi * 220 * np.arange(burst) / SAMPLE_RATE)
    mic = np.zeros((cal.total + SAMPLE_RATE, 1), dtype=np.float32)
    for position in cal.beat_positions:
        at = position + int(round(offset_ms * SAMPLE_RATE / 1000.0))
        mic[at:at + burst, 0] += (tone * env * 0.45).astype(np.float32)

    cursor = 0
    hp_peak = vc_peak = 0.0
    while not cal.finished:
        engine._mic_ring.write(np.ascontiguousarray(mic[cursor:cursor + BLOCK_SIZE]))
        hp, vc = engine._produce(BLOCK_SIZE)
        hp_peak = max(hp_peak, float(np.max(np.abs(hp))))
        vc_peak = max(vc_peak, float(np.max(np.abs(vc))))
        cursor += BLOCK_SIZE

    result = engine.calibration_result()
    check("節拍有送進耳機", hp_peak > 0.05, f"耳機峰值 {hp_peak:.3f}")
    check("校準期間不會吵到對方", vc_peak == 0.0, f"送出峰值 {vc_peak:.3f}")
    check("走完整條 _produce 仍量得準",
          result.get("ok") and abs(result["offset_ms"] - offset_ms) <= 10.0,
          f"{result.get('offset_ms')} ms(真實 {offset_ms:.0f} ms)")
    check("取出結果後校準器會收掉", engine._calibrator is None)


def test_denoise() -> None:
    print("麥克風降噪(電流聲)")
    from ktisv_engine.dsp.denoise import MicDenoiser

    sr = SAMPLE_RATE
    t = np.arange(sr * 4) / sr
    rng = np.random.RandomState(0)
    hiss = (rng.randn(len(t)) * 0.004).astype(np.float32)
    hum = (0.02 * np.sin(2 * np.pi * 60 * t)
           + 0.008 * np.sin(2 * np.pi * 120 * t)).astype(np.float32)
    voice = (0.25 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    voice[:int(sr * 1.5)] = 0
    voice[int(sr * 2.5):] = 0
    noisy = (hiss + hum + voice).reshape(-1, 1)

    def run(d):
        return np.concatenate([
            d.process(noisy[i * BLOCK_SIZE:(i + 1) * BLOCK_SIZE].copy())
            for i in range(len(noisy) // BLOCK_SIZE)])

    quiet = slice(int(sr * 0.5), int(sr * 1.4))
    spoken = slice(int(sr * 1.8), int(sr * 2.3))

    d = MicDenoiser(sr, 1)
    check("預設關閉時完全旁通",
          np.array_equal(d.process(noisy[:BLOCK_SIZE].copy()),
                         noisy[:BLOCK_SIZE]))

    d = MicDenoiser(sr, 1)
    d.enabled = True
    d.hum_hz = 60.0
    d.gate_enabled = False
    out = run(d)
    spec = np.abs(np.fft.rfft(out[quiet, 0] * np.hanning(out[quiet].shape[0])))
    ref = np.abs(np.fft.rfft(noisy[quiet, 0] * np.hanning(out[quiet].shape[0])))
    freqs = np.fft.rfftfreq(out[quiet].shape[0], 1.0 / sr)
    i60 = int(np.argmin(np.abs(freqs - 60.0)))
    cut = 20 * np.log10((ref[i60] + 1e-12) / (spec[i60] + 1e-12))
    check("60 Hz 哼聲被切掉", cut > 30.0, f"衰減 {cut:.0f} dB")

    d = MicDenoiser(sr, 1)
    d.enabled = True
    d.hum_hz = 60.0
    out = run(d)
    drop = db(noisy[quiet]) - db(out[quiet])
    keep = db(out[spoken]) - db(noisy[spoken])
    check("安靜段的底噪被壓下去", drop > 10.0, f"降 {drop:.1f} dB")
    check("出聲時幾乎不動人聲", abs(keep) < 1.5, f"變化 {keep:+.1f} dB")

    # 濾波器狀態要從靜止開始。用 sosfilt_zi() 的話等於灌一個大暫態進去,
    # 安靜段落反而會比處理前更大聲 —— 這個檢查專門釘住那個回歸。
    d = MicDenoiser(sr, 1)
    d.enabled = True
    d.hum_hz = 60.0
    d.gate_enabled = False
    first = d.process(noisy[:BLOCK_SIZE * 4].copy())
    check("開頭不會有暫態爆音",
          float(np.max(np.abs(first))) <= float(np.max(np.abs(noisy[:BLOCK_SIZE * 4]))) * 1.2,
          f"峰值 {float(np.max(np.abs(first))):.4f} vs 輸入 "
          f"{float(np.max(np.abs(noisy[:BLOCK_SIZE * 4]))):.4f}")

    # 底噪追蹤要自己找到門檻,不必使用者填數字
    check("底噪追蹤到合理的值", -70.0 < d.floor_db < -20.0,
          f"{d.floor_db:.1f} dB")


def test_limiter() -> None:
    print("前瞻限幅器(HiFi)")
    from scipy.signal.windows import blackmanharris

    from ktisv_engine.dsp.limiter import Limiter

    freq = 997.0            # 避開 FFT 格點,才看得到真實裙襬

    def thd_n(sig):
        n = len(sig) // 2 * 2
        spec = np.abs(np.fft.rfft(sig[:n] * blackmanharris(n))) ** 2
        freqs = np.fft.rfftfreq(n, 1.0 / SAMPLE_RATE)
        total = float(np.sum(spec))
        tone = float(np.sum(spec[np.abs(freqs - freq) < 25]))
        return 10 * np.log10(max(total - tone, 1e-30) / max(tone, 1e-30))

    def run(amp):
        t = np.arange(SAMPLE_RATE) / SAMPLE_RATE
        x = np.repeat((amp * np.sin(2 * np.pi * freq * t))
                      .astype(np.float32)[:, None], 2, axis=1)
        lim = Limiter(SAMPLE_RATE, 2)
        out = np.concatenate([
            lim.process(x[i * BLOCK_SIZE:(i + 1) * BLOCK_SIZE].copy())
            for i in range(len(x) // BLOCK_SIZE)])
        return out

    # 沒超過天花板就要位元透明 —— 這是「HiFi」最基本的要求
    lim = Limiter(SAMPLE_RATE, 2)
    quiet = (np.random.RandomState(1).randn(BLOCK_SIZE, 2) * 0.1).astype(np.float32)
    lim.process(quiet.copy())                    # 填管線
    passed = lim.process(np.zeros((BLOCK_SIZE, 2), dtype=np.float32))
    check("不作用時位元透明", np.array_equal(passed, quiet))

    # 大訊號:失真要遠低於原本的波形整形做法
    for amp, floor in ((0.8, -90.0), (1.0, -85.0)):
        out = run(amp)
        tail = out[len(out) // 3:, 0]
        value = thd_n(tail)
        check(f"{20 * np.log10(amp):+.0f} dBFS 時的 THD+N", value < floor,
              f"{value:.1f} dB(舊的軟限幅在 0 dBFS 是 -30.9 dB)")

    # 峰值必須守在天花板內,而且要真的有壓
    for amp in (1.0, 1.5, 3.0):
        peak = float(np.max(np.abs(run(amp))))
        check(f"輸入 {amp:.1f} 時峰值守得住", peak <= 1.0,
              f"峰值 {peak:.3f}")

    # 前瞻的代價恰好是一個 block,不能更多
    lim = Limiter(SAMPLE_RATE, 2)
    pulse = np.zeros((BLOCK_SIZE * 8, 2), dtype=np.float32)
    pulse[BLOCK_SIZE * 2] = 0.5
    out = np.concatenate([
        lim.process(pulse[i * BLOCK_SIZE:(i + 1) * BLOCK_SIZE].copy())
        for i in range(8)])
    at = int(np.argmax(np.abs(out[:, 0])))
    check("延遲恰好一個 block", at == BLOCK_SIZE * 3,
          f"脈衝在第 {at} 取樣(預期 {BLOCK_SIZE * 3})")


def test_drift() -> None:
    print("跨時脈漂移補償")
    from scipy.signal.windows import blackmanharris

    from ktisv_engine.dsp.drift import DriftCorrector

    freq = 440.0
    target = BLOCK_SIZE * 4

    def run(ppm: float, seconds: float = 30.0):
        """模擬寫入端比讀取端快/慢 ppm 的情形。"""
        corrector = DriftCorrector(SAMPLE_RATE, 2, target_fill=target)
        corrector.write(np.zeros((target, 2), dtype=np.float32))
        corrector._filled = float(target)

        phase = 0.0
        carry = 0.0
        out = []
        for _ in range(int(seconds * SAMPLE_RATE / BLOCK_SIZE)):
            carry += BLOCK_SIZE * (1.0 + ppm * 1e-6)
            count = int(round(carry))
            carry -= count
            index = np.arange(phase, phase + count)
            phase += count
            corrector.write(np.repeat(
                (0.4 * np.sin(2 * np.pi * freq * index / SAMPLE_RATE))
                .astype(np.float32)[:, None], 2, axis=1))
            out.append(corrector.read(BLOCK_SIZE))
        return corrector, np.concatenate(out)[:, 0]

    def junk_db(signal, width=400.0):
        body = signal[len(signal) // 2:]
        n = len(body) // 2 * 2
        spec = np.abs(np.fft.rfft(body[:n] * blackmanharris(n))) ** 2
        freqs = np.fft.rfftfreq(n, 1.0 / SAMPLE_RATE)
        inside = float(np.sum(spec[np.abs(freqs - freq) < width]))
        return 10 * np.log10(max(float(np.sum(spec)) - inside, 1e-30)
                             / max(inside, 1e-30))

    # 這是整件事的重點:時脈有差時,輸出不能因此變髒。
    # 舊做法是丟樣本,每次丟都是一道波形斷點 —— 實機上量到 −40.5 dB 的
    # 寬頻雜訊,聽起來就是斷續與電流聲。
    # 起始值要用 -inf:量的是 dB,全部都是負數,從 0 起算的話 max() 永遠
    # 回 0,測試就變成永遠失敗(而且失敗訊息看起來像「訊號全是雜訊」)。
    worst = float("-inf")
    for ppm in (0.0, 50.0, -50.0, 200.0, -200.0):
        corrector, signal = run(ppm)
        value = junk_db(signal)
        worst = max(worst, value)
        if corrector.underflows or corrector.overflows:
            check(f"{ppm:+.0f} ppm 時不會欠載或溢位", False,
                  f"under={corrector.underflows} over={corrector.overflows}")
    check("各種時脈差下都保持乾淨", worst < -90.0,
          f"最差 {worst:.1f} dB(舊做法實機量到 −40.5 dB)")

    # 速率要真的追上去,否則存量會一路漂到見底
    for ppm in (80.0, -80.0):
        corrector, _ = run(ppm)
        tracked = (corrector.ratio - 1.0) * 1e6
        check(f"追得上 {ppm:+.0f} ppm 的時脈差",
              abs(tracked - ppm) < 40.0, f"追到 {tracked:+.1f} ppm")
        fill = corrector.available()
        check(f"{ppm:+.0f} ppm 時存量守在目標附近",
              target * 0.3 < fill < target * 2.0,
              f"{fill} / 目標 {target}")

    # 時脈完全一致時應該幾乎不動
    corrector, signal = run(0.0)
    check("沒有時脈差時 ratio 幾乎不動",
          abs(corrector.ratio - 1.0) < 5e-5,
          f"{(corrector.ratio - 1.0) * 1e6:+.1f} ppm")


def main() -> int:
    for fn in (test_ring, test_smooth_gain, test_eq, test_eq_bands, test_pitch,
               test_echo, test_separation, test_delay, test_vc_sync,
               test_monitor_send, test_denoise, test_limiter, test_drift,
               test_calibration,
               test_calibration_in_engine, test_engine_mix,
               test_engine_pitch):
        fn()
        print()
    if FAILURES:
        print(f"{len(FAILURES)} 項未通過: " + ", ".join(FAILURES))
        return 1
    print("全部通過。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
