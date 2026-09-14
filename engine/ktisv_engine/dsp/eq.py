"""參數式等化器。

頻段數量、每一段的頻率與 Q 值都可以即時改。最低的那一段用 low-shelf、
最高的用 high-shelf、其餘用 peaking,係數採 RBJ Audio EQ Cookbook 的公式,
以 second-order sections 串接後由 ``scipy.signal.sosfilt`` 逐 block 濾波
(保留 zi 狀態以維持連續性)。

為什麼頻段是物件而不是兩條平行的 list
------------------------------------
使用者可以在播放中新增或刪除頻段,而每一段都帶著自己的濾波器狀態(zi)。
用索引對應的話,刪掉中間一段會讓後面所有段的狀態全部錯位,聽起來就是一聲
爆音加上一段亂掉的殘響。改成一段一個物件,重建係數矩陣時就能靠物件本身
認出「這是同一段」,把狀態原封不動搬過去 —— 只有真正被刪掉的那一段會歸零。

參數平滑
--------
前端每 40 ms 送一次推桿值。直接把係數換掉的話,拖推桿時每一步都是一次
係數跳變,聽起來是一串細碎的「滋滋」聲(zipper noise),增益越大越明顯。

所以每一段分成**目標值**(使用者設的)與**目前值**(真正拿去算係數的),
目前值以時間常數 ``SMOOTH_MS`` 指數逼近目標值。逼近期間把 block 切成
``SMOOTH_CHUNK`` 取樣的小段、每段重算一次係數 —— 係數變化被攤成上百個
察覺不到的小步。收斂之後就回到整塊一次濾波,平常不付任何額外成本。

頻率與 Q 在對數軸上逼近:從 100 Hz 移到 10 kHz,線性內插會在前一瞬間就
衝過好幾個八度,對數內插才是人耳聽到的「等速滑過去」。

啟用 / 停用也走同一條路:停用等於把目標增益全部設成 0,等目前值真的歸零
才進入旁通。不必再像以前那樣在停用瞬間清空濾波器狀態(那本身就是一聲爆音)。

自動防削波
----------
EQ 往上推的量會直接疊在訊號峰值上,微笑曲線兩端 +4 dB,整首歌就多 4 dB,
原本就接近滿格的 YouTube 音源會直接削波。``auto_headroom`` 打開時,
依目標響應曲線的最高點自動把整體壓低同樣的量 —— 聽起來音色照樣改變,
但峰值不會因為 EQ 而變高。壓低量同樣平滑,不會隨推桿一格一格跳。
"""

from __future__ import annotations

import math

import numpy as np
from scipy.signal import sosfilt, sosfreqz

DEFAULT_BANDS: tuple[float, ...] = (
    31.25, 62.5, 125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0, 16000.0,
)
DEFAULT_Q = 1.41
SHELF_Q = 0.7
"""兩端 shelf 的預設 Q。控制的是轉折的陡峭程度,不是頻寬。"""

GAIN_LIMIT_DB = 15.0
MIN_FREQ = 20.0
MIN_Q = 0.1
MAX_Q = 18.0
MAX_BANDS = 24
"""頻段數上限。每一段都是一顆 biquad,毫無節制地加下去只會吃掉音訊回呼的預算。"""

SMOOTH_MS = 25.0
"""參數平滑的時間常數。約 3 倍時間常數(75 ms)走完 95% —— 比推桿送出的
間隔(40 ms)長,所以相鄰兩次更新之間是連續的;又短到拖動時不覺得遲鈍。"""

SMOOTH_CHUNK = 64
"""平滑期間每幾個取樣重算一次係數。48 kHz 下約 1.3 ms。"""

_GAIN_EPS_DB = 0.005
_LOG_EPS = 1e-4
_MAKEUP_EPS = 1e-4

_HEADROOM_GRID = np.logspace(np.log10(20.0), np.log10(20000.0), 160)


def _peaking(f0: float, gain_db: float, q: float, sr: int) -> np.ndarray:
    a = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * math.pi * f0 / sr
    alpha = math.sin(w0) / (2.0 * q)
    cos_w0 = math.cos(w0)

    b0 = 1.0 + alpha * a
    b1 = -2.0 * cos_w0
    b2 = 1.0 - alpha * a
    a0 = 1.0 + alpha / a
    a1 = -2.0 * cos_w0
    a2 = 1.0 - alpha / a
    return np.array([b0 / a0, b1 / a0, b2 / a0, 1.0, a1 / a0, a2 / a0], dtype=np.float64)


def _shelf(f0: float, gain_db: float, q: float, sr: int, high: bool) -> np.ndarray:
    a = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * math.pi * f0 / sr
    cos_w0 = math.cos(w0)
    sin_w0 = math.sin(w0)

    # Cookbook 的 shelf 用 q 當「斜率 S」。S 一大、增益又深時,根號裡會變成
    # 負的 —— 那不是「更陡」而是無解,算出來是 NaN,整條濾波鏈會一次污染成
    # NaN 然後徹底沒聲音。既然 Q 現在是使用者可以拉到 18 的旋鈕,這裡就得
    # 自己把根號夾在正數,超過物理上限的部分就停在最陡。
    radicand = (a + 1.0 / a) * (1.0 / q - 1.0) + 2.0
    alpha = sin_w0 / 2.0 * math.sqrt(max(radicand, 0.05))
    two_sqrt_a_alpha = 2.0 * math.sqrt(a) * alpha

    if high:
        b0 = a * ((a + 1.0) + (a - 1.0) * cos_w0 + two_sqrt_a_alpha)
        b1 = -2.0 * a * ((a - 1.0) + (a + 1.0) * cos_w0)
        b2 = a * ((a + 1.0) + (a - 1.0) * cos_w0 - two_sqrt_a_alpha)
        a0 = (a + 1.0) - (a - 1.0) * cos_w0 + two_sqrt_a_alpha
        a1 = 2.0 * ((a - 1.0) - (a + 1.0) * cos_w0)
        a2 = (a + 1.0) - (a - 1.0) * cos_w0 - two_sqrt_a_alpha
    else:
        b0 = a * ((a + 1.0) - (a - 1.0) * cos_w0 + two_sqrt_a_alpha)
        b1 = 2.0 * a * ((a - 1.0) - (a + 1.0) * cos_w0)
        b2 = a * ((a + 1.0) - (a - 1.0) * cos_w0 - two_sqrt_a_alpha)
        a0 = (a + 1.0) + (a - 1.0) * cos_w0 + two_sqrt_a_alpha
        a1 = -2.0 * ((a - 1.0) + (a + 1.0) * cos_w0)
        a2 = (a + 1.0) + (a - 1.0) * cos_w0 - two_sqrt_a_alpha

    return np.array([b0 / a0, b1 / a0, b2 / a0, 1.0, a1 / a0, a2 / a0], dtype=np.float64)


def _section(shelf: str, freq: float, gain: float, q: float, sr: int) -> np.ndarray:
    if shelf:
        return _shelf(freq, gain, q, sr, high=shelf == "high")
    return _peaking(freq, gain, q, sr)


class EqBand:
    """一個頻段。``shelf`` 由所在位置決定,不是使用者設定的(見 GraphicEQ)。

    ``freq`` / ``gain`` / ``q`` 是目標值;``cur_*`` 是音訊執行緒目前實際
    套用的值,由 ``GraphicEQ.process`` 平滑地往目標值推。
    """

    __slots__ = ("freq", "gain", "q", "shelf", "cur_freq", "cur_gain", "cur_q")

    def __init__(self, freq: float, gain: float = 0.0, q: float = DEFAULT_Q) -> None:
        self.freq = float(freq)
        self.gain = float(gain)
        self.q = float(q)
        self.shelf = ""      # "low" / "high" / ""(peaking)
        # 新的頻段從「平的」開始,增益再平滑地長出來 —— 播放中套用預設集或
        # 整組換掉頻段時才不會一口氣跳上去。頻率與 Q 直接就位:增益為 0 時
        # 它們對聲音沒有影響,從哪裡開始都一樣。
        self.cur_freq = self.freq
        self.cur_gain = 0.0
        self.cur_q = self.q

    def to_dict(self) -> dict:
        return {"freq": round(self.freq, 2), "gain": round(self.gain, 2),
                "q": round(self.q, 3),
                "type": self.shelf + "_shelf" if self.shelf else "peaking"}


class GraphicEQ:
    """可增刪頻段的參數式 EQ。所有調整都能在播放中即時生效,而且是平滑的。"""

    def __init__(self, samplerate: int, channels: int,
                 bands=DEFAULT_BANDS, q: float = DEFAULT_Q,
                 auto_headroom: bool = False) -> None:
        self.samplerate = samplerate
        self.channels = channels
        self.default_q = float(q)
        self._enabled = True
        self._auto_headroom = bool(auto_headroom)

        self._bands: list[EqBand] = []
        # (頻段, 濾波器狀態) 綁成一個 tuple 整個替換。控制執行緒增刪頻段時
        # 音訊執行緒可能正在濾波,拆成兩個屬性分開寫的話,中間那一瞬間兩者
        # 的列數會對不上。
        self._layout: tuple[list[EqBand], np.ndarray] = (
            [], np.zeros((0, 2, channels), dtype=np.float64))
        self._sos = np.zeros((0, 6), dtype=np.float64)
        self._target_sos = np.zeros((0, 6), dtype=np.float64)
        self._settled = False
        self._flat = True           # 目前值是否完全平坦(可以旁通)
        self._was_bypassed = True
        self._makeup = 1.0          # 目前套用的整體增益(線性)
        self._makeup_target = 1.0
        self._headroom_db = 0.0
        self.set_bands(bands)
        # 建構時就位,不要從平的慢慢長出來 —— 還沒有聲音流過,沒有爆音可言。
        self.snap()

    # ── 查詢 ────────────────────────────────────────────────────────────
    @property
    def bands(self) -> tuple[float, ...]:
        """各段的中心頻率。"""
        return tuple(b.freq for b in self._bands)

    @property
    def gains(self) -> list[float]:
        return [b.gain for b in self._bands]

    @property
    def band_count(self) -> int:
        return len(self._bands)

    def band_info(self) -> list[dict]:
        return [b.to_dict() for b in self._bands]

    @property
    def enabled(self) -> bool:
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        value = bool(value)
        if value != self._enabled:
            self._enabled = value
            self._retarget()

    @property
    def auto_headroom(self) -> bool:
        return self._auto_headroom

    @auto_headroom.setter
    def auto_headroom(self, value: bool) -> None:
        value = bool(value)
        if value != self._auto_headroom:
            self._auto_headroom = value
            self._retarget()

    @property
    def headroom_db(self) -> float:
        """自動防削波目前(目標)壓低了多少 dB。關閉時為 0。"""
        return self._headroom_db

    # ── 值的調整(不改變結構,濾波器狀態原封不動)──────────────────────
    def _clamp_freq(self, freq: float) -> float:
        # 超過 Nyquist 的頻段沒有意義:雙線性轉換會把它折回來,
        # 使用者看到的是一條完全不對應標籤的曲線。
        return max(MIN_FREQ, min(self.samplerate * 0.45, float(freq)))

    @staticmethod
    def _clamp_gain(gain_db: float) -> float:
        return max(-GAIN_LIMIT_DB, min(GAIN_LIMIT_DB, float(gain_db)))

    @staticmethod
    def _clamp_q(q: float) -> float:
        return max(MIN_Q, min(MAX_Q, float(q)))

    def set_gain(self, index: int, gain_db: float) -> None:
        self.set_band(index, gain=gain_db)

    def set_band(self, index: int, gain: float | None = None,
                 freq: float | None = None, q: float | None = None) -> None:
        """改單一頻段的任一項參數。傳 None 表示該項不動。"""
        if not 0 <= index < len(self._bands):
            return
        band = self._bands[index]
        changed = False

        if gain is not None:
            value = self._clamp_gain(gain)
            if abs(band.gain - value) >= 1e-6:
                band.gain = value
                changed = True
        if freq is not None:
            value = self._clamp_freq(freq)
            if abs(band.freq - value) >= 1e-6:
                band.freq = value
                changed = True
        if q is not None:
            value = self._clamp_q(q)
            if abs(band.q - value) >= 1e-6:
                band.q = value
                changed = True

        if changed:
            self._retarget()

    def set_gains(self, gains_db) -> None:
        changed = False
        for band, g in zip(self._bands, gains_db):
            value = self._clamp_gain(g)
            if abs(band.gain - value) >= 1e-6:
                band.gain = value
                changed = True
        if changed:
            self._retarget()

    def reset(self) -> None:
        """把所有增益歸零。頻率與 Q 是使用者的配置,不動。

        不清濾波器狀態:增益會平滑地回到 0,清狀態反而會爆一聲。
        """
        for band in self._bands:
            band.gain = 0.0
        self._retarget()

    # ── 結構的調整(會影響濾波器狀態的配置)────────────────────────────
    def set_bands(self, specs) -> None:
        """整組換掉。``specs`` 可以是頻率數列,或 {freq, gain, q} 的字典數列。"""
        raw = list(specs)
        bands: list[EqBand] = []
        for i, spec in enumerate(raw):
            if isinstance(spec, EqBand):
                band = spec
            elif isinstance(spec, dict):
                band = EqBand(spec.get("freq", 1000.0),
                              spec.get("gain", 0.0),
                              spec.get("q", self.default_q))
            else:
                # 只給頻率的寫法(例如 DEFAULT_BANDS):兩端會變成 shelf,
                # 而 shelf 的 q 是斜率而非頻寬,用 peaking 的預設值太陡。
                shelf = i == 0 or i == len(raw) - 1
                band = EqBand(spec, 0.0, SHELF_Q if shelf else self.default_q)
            band.freq = self._clamp_freq(band.freq)
            band.gain = self._clamp_gain(band.gain)
            band.q = self._clamp_q(band.q)
            band.cur_freq = self._clamp_freq(band.cur_freq)
            band.cur_q = self._clamp_q(band.cur_q)
            bands.append(band)
            if len(bands) >= MAX_BANDS:
                break

        if not bands:
            raise ValueError("EQ 至少要有一個頻段。")

        bands.sort(key=lambda b: b.freq)
        self._restructure(bands)

    def add_band(self, freq: float, gain: float = 0.0,
                 q: float | None = None) -> int:
        """新增一段,回傳它排序後的索引。"""
        if len(self._bands) >= MAX_BANDS:
            raise ValueError(f"最多只能有 {MAX_BANDS} 個頻段。")

        band = EqBand(self._clamp_freq(freq), self._clamp_gain(gain),
                      self._clamp_q(self.default_q if q is None else q))
        # 依頻率插入。shelf 是照位置給的(頭尾各一),所以維持頻率順序,
        # 兩端才會落在真正最低與最高的那兩段上。
        bands = list(self._bands)
        position = len(bands)
        for i, existing in enumerate(bands):
            if band.freq < existing.freq:
                position = i
                break
        bands.insert(position, band)
        self._restructure(bands)
        return position

    def remove_band(self, index: int) -> None:
        if not 0 <= index < len(self._bands):
            raise ValueError(f"沒有第 {index} 個頻段。")
        if len(self._bands) <= 1:
            raise ValueError("至少要保留一個頻段。")
        bands = list(self._bands)
        bands.pop(index)
        self._restructure(bands)

    def clear_state(self) -> None:
        bands, zi = self._layout
        self._layout = (bands, np.zeros_like(zi))

    def snap(self) -> None:
        """跳過平滑,讓目前值立刻等於目標值。給還沒有聲音流過的場合用。"""
        for band in self._bands:
            band.cur_freq = band.freq
            band.cur_gain = self._effective_gain(band)
            band.cur_q = band.q
        self._makeup = self._makeup_target
        self._sos = self._current_sos(self._bands)
        self._flat = self._is_flat(self._bands)
        self._settled = True

    # ── 係數 ────────────────────────────────────────────────────────────
    def _effective_gain(self, band: EqBand) -> float:
        return band.gain if self._enabled else 0.0

    def _restructure(self, bands: list[EqBand]) -> None:
        # 重新配置 zi。整個歸零會在增刪頻段的瞬間爆一聲,所以逐段認人:
        # 留下來的那幾段把自己的狀態帶到新位置,只有新來的從零開始。
        old_bands, old_zi = self._layout
        previous = {id(band): row for row, band in enumerate(old_bands)}
        zi = np.zeros((len(bands), 2, self.channels), dtype=np.float64)
        for i, band in enumerate(bands):
            row = previous.get(id(band))
            if row is not None:
                zi[i] = old_zi[row]

        last = len(bands) - 1
        for i, band in enumerate(bands):
            band.shelf = "low" if i == 0 and last > 0 else \
                         "high" if i == last and last > 0 else ""

        # 先放倒 _settled 再換 layout,音訊執行緒才不會拿舊係數配新狀態
        self._settled = False
        self._bands = bands
        self._layout = (bands, zi)
        self._retarget()

    def _retarget(self) -> None:
        """目標值變了:重算目標響應(給 UI 與防削波用),並喚醒平滑。"""
        sr = self.samplerate
        self._target_sos = np.array(
            [_section(b.shelf, b.freq, self._effective_gain(b), b.q, sr)
             for b in self._bands], dtype=np.float64).reshape(-1, 6)

        headroom = 0.0
        if self._auto_headroom and self._enabled and \
                any(abs(b.gain) >= 1e-3 for b in self._bands):
            peak = float(np.max(self._response_db(self._target_sos,
                                                  _HEADROOM_GRID)))
            headroom = max(0.0, peak)
        self._headroom_db = headroom
        self._makeup_target = 10.0 ** (-headroom / 20.0)
        self._settled = False

    def _current_sos(self, bands: list[EqBand]) -> np.ndarray:
        sr = self.samplerate
        return np.array([_section(b.shelf, b.cur_freq, b.cur_gain, b.cur_q, sr)
                         for b in bands], dtype=np.float64).reshape(-1, 6)

    @staticmethod
    def _is_flat(bands: list[EqBand]) -> bool:
        return all(abs(b.cur_gain) < 1e-3 for b in bands)

    def _advance(self, bands: list[EqBand], frames: int) -> bool:
        """把目前值往目標推 ``frames`` 個取樣的量。回傳是否全部收斂。"""
        alpha = 1.0 - math.exp(-frames / (SMOOTH_MS * 1e-3 * self.samplerate))
        done = True
        for b in bands:
            target_gain = self._effective_gain(b)
            diff = target_gain - b.cur_gain
            if abs(diff) > _GAIN_EPS_DB:
                b.cur_gain += diff * alpha
                done = False
            else:
                b.cur_gain = target_gain

            log_diff = math.log(b.freq / b.cur_freq)
            if abs(log_diff) > _LOG_EPS:
                b.cur_freq *= math.exp(log_diff * alpha)
                done = False
            else:
                b.cur_freq = b.freq

            log_diff = math.log(b.q / b.cur_q)
            if abs(log_diff) > _LOG_EPS:
                b.cur_q *= math.exp(log_diff * alpha)
                done = False
            else:
                b.cur_q = b.q

        diff = self._makeup_target - self._makeup
        if abs(diff) > _MAKEUP_EPS:
            self._makeup += diff * alpha
            done = False
        else:
            self._makeup = self._makeup_target
        return done

    # ── 處理 ────────────────────────────────────────────────────────────
    def process(self, x: np.ndarray) -> np.ndarray:
        """x: (frames, channels) float32 → 濾波後的陣列(旁通時是原陣列)。"""
        if self._settled and self._flat and self._makeup == 1.0:
            self._was_bypassed = True
            return x

        bands, zi = self._layout
        if self._was_bypassed:
            # 從旁通回來:舊狀態是很久以前留下的,和現在的訊號毫無關係,
            # 帶著它濾波就是一聲爆音。增益此刻一定是 0(旁通的前提),
            # 而「係數為恆等、狀態為零」的 biquad 輸出與輸入逐點相同 ——
            # 所以歸零狀態是無縫的。
            zi = np.zeros_like(zi)
            self._was_bypassed = False

        data = x.astype(np.float64, copy=False)
        sos = self._sos

        # 列數比對:控制執行緒剛換了結構、還沒來得及把 _settled 放倒的那一瞬間,
        # 這裡可能拿到新的 zi 配舊的係數。對不上就走平滑路徑,它會從頻段重算。
        if self._settled and len(sos) == len(bands):
            y, zi = sosfilt(sos, data, axis=0, zi=zi)
            if self._makeup != 1.0:
                y *= self._makeup
        else:
            y = np.empty_like(data)
            n = len(data)
            start = 0
            while start < n:
                if self._settled and len(self._sos) == len(bands):
                    # 中途收斂了,剩下的整段一次濾完
                    y[start:], zi = sosfilt(self._sos, data[start:], axis=0, zi=zi)
                    if self._makeup != 1.0:
                        y[start:] *= self._makeup
                    break
                stop = min(n, start + SMOOTH_CHUNK)
                before = self._makeup
                done = self._advance(bands, stop - start)
                self._sos = self._current_sos(bands)
                y[start:stop], zi = sosfilt(self._sos, data[start:stop],
                                            axis=0, zi=zi)
                # 整體增益在小段內線性內插,連一個小段的階梯都不留
                if before != 1.0 or self._makeup != 1.0:
                    y[start:stop] *= np.linspace(
                        before, self._makeup, stop - start,
                        endpoint=False)[:, None]
                if done:
                    self._flat = self._is_flat(bands)
                    self._settled = True
                start = stop

        # 控制執行緒在這段期間改了結構的話,它已經換上一份新的 layout,
        # 這一塊的狀態就丟掉(只影響一個 block),不要蓋掉對方的配置。
        if self._layout[0] is bands:
            self._layout = (bands, zi)
        return y.astype(np.float32, copy=False)

    def response(self, freqs: np.ndarray) -> np.ndarray:
        """回傳指定頻率上的目標響應(dB,含防削波的整體壓低),給 UI 畫曲線用。"""
        db = self._response_db(self._target_sos, freqs)
        return db - self._headroom_db

    def _response_db(self, sos: np.ndarray, freqs: np.ndarray) -> np.ndarray:
        freqs = np.asarray(freqs, dtype=np.float64)
        if len(sos) == 0:
            return np.zeros_like(freqs)
        w = 2.0 * np.pi * freqs / self.samplerate
        _, h = sosfreqz(sos, worN=w)
        return 20.0 * np.log10(np.maximum(np.abs(h), 1e-9))
