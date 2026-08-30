"""歌詞取得與解析。

來源(依優先順序)
------------------
1. **同名 .lrc 檔** —— 使用者自己準備的歌詞,放在音檔旁邊。最準確,而且
   時間軸通常是為卡拉 OK 對過的。
2. **YouTube 字幕** —— 影片本身附的字幕。上傳者提供的(``subtitles``)優先於
   自動生成的(``automatic_captions``),因為自動生成的斷句與用字常有錯。

刻意**不**從歌詞網站抓取 —— 那是未經授權轉載受著作權保護的內容。這裡取得的
都是使用者本來就有存取權的東西:自己的檔案,或正在播放的那支影片附帶的字幕。

實務上的命中率
--------------
音樂影片很少附字幕(自動字幕對唱歌的辨識效果差,YouTube 多半不會生成),
所以純線上播放時常常會是「找不到歌詞」。要穩定有歌詞,最實際的做法是
自己準備 .lrc 檔。

時間軸
------
所有格式都解析成 ``LyricLine(time, text)`` 的序列,時間是相對於歌曲開頭的秒數。
沒有時間軸的純文字歌詞也支援,但不會有逐行高亮。
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

# 偏好的字幕語言,依序嘗試。中文放前面是因為這個專案主要面向華語使用者。
PREFERRED_LANGUAGES = ("zh-Hant", "zh-TW", "zh-HK", "zh", "zh-Hans", "zh-CN",
                       "ja", "ko", "en")


@dataclass
class LyricLine:
    time: float          # 秒
    text: str

    def to_dict(self) -> dict:
        return {"time": round(self.time, 3), "text": self.text}


@dataclass
class Lyrics:
    lines: list[LyricLine]
    source: str = ""      # "lrc" / "vtt" / "ass" / "ttml" / "json3" / "plain" …
    language: str = ""
    synced: bool = True   # 有沒有可用的時間軸
    # 這支影片還有哪些語言可挑。介面要靠它列下拉選單,所以就算這次
    # 沒抓到歌詞也要填 —— 使用者才知道可以換一種語言再試。
    available: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "lines": [line.to_dict() for line in self.lines],
            "source": self.source,
            "language": self.language,
            "synced": self.synced,
            "count": len(self.lines),
            "available": list(self.available),
        }

    @property
    def is_empty(self) -> bool:
        return not self.lines


# ── 時間字串解析 ────────────────────────────────────────────────────────
_LRC_TAG = re.compile(r"\[(\d+):(\d{1,2})(?:[.:](\d{1,3}))?\]")
_VTT_TIME = re.compile(
    r"(\d{1,2}):(\d{2}):(\d{2})[.,](\d{1,3})\s*-->\s*"
    r"(\d{1,2}):(\d{2}):(\d{2})[.,](\d{1,3})")
_TAG_STRIP = re.compile(r"<[^>]+>")


def _hms_to_seconds(hours: str, minutes: str, seconds: str, millis: str) -> float:
    return (int(hours) * 3600 + int(minutes) * 60 + int(seconds)
            + int(millis.ljust(3, "0")) / 1000.0)


# ── LRC ─────────────────────────────────────────────────────────────────
def parse_lrc(text: str) -> Lyrics:
    """解析 .lrc。一行可以有多個時間標籤(副歌重複時常見)。"""
    lines: list[LyricLine] = []
    for raw in text.splitlines():
        tags = list(_LRC_TAG.finditer(raw))
        if not tags:
            continue
        content = _LRC_TAG.sub("", raw).strip()
        if not content:
            continue
        for tag in tags:
            minutes, seconds, fraction = tag.group(1), tag.group(2), tag.group(3) or "0"
            # LRC 的小數位通常是百分之一秒
            millis = (fraction.ljust(3, "0") if len(fraction) == 3
                      else fraction.ljust(2, "0") + "0")
            lines.append(LyricLine(
                time=int(minutes) * 60 + int(seconds) + int(millis) / 1000.0,
                text=content))

    lines.sort(key=lambda item: item.time)
    return Lyrics(lines=lines, source="lrc", synced=bool(lines))


# ── VTT / SRT ───────────────────────────────────────────────────────────
def parse_vtt(text: str) -> Lyrics:
    """解析 WebVTT 或 SRT。兩者的時間軸格式夠接近,可以共用。"""
    lines: list[LyricLine] = []
    current_time: float | None = None
    buffer: list[str] = []

    def flush() -> None:
        if current_time is None or not buffer:
            return
        content = " ".join(part.strip() for part in buffer if part.strip())
        content = _TAG_STRIP.sub("", content).strip()
        if content:
            lines.append(LyricLine(time=current_time, text=content))

    for raw in text.splitlines():
        stripped = raw.strip()
        match = _VTT_TIME.search(stripped)
        if match:
            flush()
            buffer = []
            current_time = _hms_to_seconds(*match.groups()[:4])
            continue
        if not stripped:
            flush()
            buffer = []
            current_time = None
            continue
        # 跳過 WEBVTT 檔頭、NOTE、以及 SRT 的序號行
        if stripped.upper().startswith(("WEBVTT", "NOTE", "STYLE", "REGION")):
            continue
        if stripped.isdigit() and current_time is None:
            continue
        if current_time is not None:
            buffer.append(stripped)

    flush()
    return Lyrics(lines=_dedupe(lines), source="vtt", synced=bool(lines))


def _dedupe(lines: list[LyricLine]) -> list[LyricLine]:
    """自動字幕常把同一句重複送好幾次(滾動效果),去掉連續重複。"""
    result: list[LyricLine] = []
    for line in lines:
        if result and result[-1].text == line.text:
            continue
        result.append(line)
    return result


# ── ASS / SSA ───────────────────────────────────────────────────────────
# 覆寫標籤 {\pos(...)}、{\an8}、{\c&HFFFFFF&};卡拉 OK 字幕還會有逐字計時
# 的 {\k50}。這些都不是歌詞內容,要拿掉。
_ASS_OVERRIDE = re.compile(r"\{[^}]*\}")
_ASS_DRAWING = re.compile(r"\\p[1-9].*?\\p0", re.DOTALL)
_ASS_TIME = re.compile(r"\s*(\d+):(\d{1,2}):(\d{1,2})[.:](\d{1,2})\s*$")


def _ass_time(value: str) -> float | None:
    """ASS 的時間是 h:mm:ss.cc —— 最後是百分之一秒,不是千分之一。"""
    match = _ASS_TIME.match(value)
    if not match:
        return None
    hours, minutes, seconds, centis = match.groups()
    return (int(hours) * 3600 + int(minutes) * 60 + int(seconds)
            + int(centis) / 100.0)


def parse_ass(text: str) -> Lyrics:
    """解析 ASS / SSA。

    只取 [Events] 段落裡的 Dialogue。欄位順序由該段的 ``Format:`` 那一行
    決定 —— 不同工具產出的順序不一樣,寫死索引會抓到錯的欄位。
    """
    lines: list[LyricLine] = []
    fields: list[str] = []
    in_events = False

    for raw in text.splitlines():
        stripped = raw.strip()
        if stripped.startswith("["):
            in_events = stripped.lower().startswith("[events")
            continue
        if not in_events:
            continue

        head, _, rest = stripped.partition(":")
        head = head.strip().lower()
        if head == "format":
            fields = [part.strip().lower() for part in rest.split(",")]
            continue
        # Comment 是被註解掉的字幕,不該顯示
        if head != "dialogue" or not fields:
            continue

        # 最後一欄是 Text,它本身可能含逗號,所以只切前面那些
        parts = rest.split(",", len(fields) - 1)
        if len(parts) != len(fields):
            continue
        row = dict(zip(fields, parts))

        start = _ass_time(row.get("start", ""))
        if start is None:
            continue

        content = row.get("text", "")
        content = _ASS_DRAWING.sub("", content)
        content = _ASS_OVERRIDE.sub("", content)
        for token, replacement in (("\\N", " "), ("\\n", " "), ("\\h", " ")):
            content = content.replace(token, replacement)
        content = content.strip()
        if content:
            lines.append(LyricLine(time=start, text=content))

    lines.sort(key=lambda line: line.time)
    return Lyrics(lines=_dedupe(lines), source="ass", synced=bool(lines))


# ── TTML / DFXP ─────────────────────────────────────────────────────────
_TTML_CLOCK = re.compile(r"^(\d+):(\d{1,2}):(\d{1,2})(?:[.,](\d{1,3}))?$")
_TTML_OFFSET = re.compile(r"^([\d.]+)(ms|s|m|h)?$")


def _ttml_time(value: str) -> float | None:
    """TTML 的時間可以是 00:01:02.345,也可以是 62.5s / 62500ms。"""
    value = (value or "").strip()
    if not value:
        return None

    match = _TTML_CLOCK.match(value)
    if match:
        hours, minutes, seconds, millis = match.groups()
        return (int(hours) * 3600 + int(minutes) * 60 + int(seconds)
                + int((millis or "0").ljust(3, "0")) / 1000.0)

    match = _TTML_OFFSET.match(value)
    if match:
        number = float(match.group(1))
        unit = match.group(2) or "s"
        return number * {"ms": 0.001, "s": 1.0, "m": 60.0, "h": 3600.0}[unit]
    return None


def parse_ttml(text: str) -> Lyrics:
    """解析 TTML / DFXP。

    用 XML 解析而不是正則:TTML 的 <p> 裡常有巢狀的 <span>(逐字計時、
    樣式),正則抓文字會漏掉巢狀內容或整段重複。itertext() 一次收齊。
    """
    import xml.etree.ElementTree as ElementTree

    try:
        root = ElementTree.fromstring(text)
    except Exception:
        return Lyrics(lines=[], source="ttml", synced=False)

    lines: list[LyricLine] = []
    for node in root.iter():
        # 標籤帶命名空間,例如 {http://www.w3.org/ns/ttml}p
        if node.tag.rsplit("}", 1)[-1].lower() != "p":
            continue
        start = _ttml_time(node.attrib.get("begin", ""))
        if start is None:
            continue
        content = " ".join(part.strip() for part in node.itertext() if part.strip())
        if content:
            lines.append(LyricLine(time=start, text=content))

    lines.sort(key=lambda line: line.time)
    return Lyrics(lines=_dedupe(lines), source="ttml", synced=bool(lines))


# ── YouTube json3 / srv3 ────────────────────────────────────────────────
def parse_json3(text: str) -> Lyrics:
    """解析 YouTube 自家的 json3 字幕。

    自動字幕在 json3 底下是逐字送的:一個 event 裡有好幾個 segs,各自帶
    tOffsetMs。這裡把同一個 event 的 segs 併成一句、時間取 event 的起點 ——
    逐字時間要做卡拉 OK 才有意義,當成歌詞行反而太碎。
    """
    import json as json_mod

    try:
        data = json_mod.loads(text)
    except Exception:
        return Lyrics(lines=[], source="json3", synced=False)

    lines: list[LyricLine] = []
    for event in data.get("events") or []:
        start_ms = event.get("tStartMs")
        segments = event.get("segs")
        if start_ms is None or not segments:
            continue
        content = "".join(seg.get("utf8", "") for seg in segments)
        content = " ".join(content.split())          # 換行與連續空白收成一個
        if content:
            lines.append(LyricLine(time=float(start_ms) / 1000.0, text=content))

    lines.sort(key=lambda line: line.time)
    return Lyrics(lines=_dedupe(lines), source="json3", synced=bool(lines))


def parse_plain(text: str) -> Lyrics:
    """純文字歌詞:沒有時間軸,只能整段顯示。"""
    lines = [LyricLine(time=0.0, text=raw.strip())
             for raw in text.splitlines() if raw.strip()]
    return Lyrics(lines=lines, source="plain", synced=False)


def parse_auto(text: str) -> Lyrics:
    """依內容猜格式。

    順序是有意義的:先認「一定是這個格式」的特徵,再認比較鬆的。
    json3 與 TTML 先判,因為它們的內容裡也可能出現看起來像時間軸的字串。
    """
    head = text.lstrip()[:400]

    if head.startswith("{") and '"events"' in text[:4000]:
        parsed = parse_json3(text)
        if not parsed.is_empty:
            return parsed

    if head.startswith("<") and ("<tt" in head or "ttml" in head.lower()):
        parsed = parse_ttml(text)
        if not parsed.is_empty:
            return parsed

    if "[script info]" in text[:400].lower() or "[events]" in text.lower():
        parsed = parse_ass(text)
        if not parsed.is_empty:
            return parsed

    if _LRC_TAG.search(text):
        return parse_lrc(text)
    if _VTT_TIME.search(text):
        return parse_vtt(text)
    return parse_plain(text)


# ── 本機檔案 ────────────────────────────────────────────────────────────
SIDECAR_SUFFIXES = (".lrc", ".ass", ".ssa", ".vtt", ".srt",
                    ".ttml", ".dfxp", ".json", ".txt")


def find_sidecar(audio_path: str, language: str = "") -> str | None:
    """找音檔旁邊的同名歌詞檔。

    也認得帶語言的命名(``歌名.zh-Hant.srt``)—— 字幕工具與 yt-dlp 都是
    這樣落檔的,同一首歌常常好幾個語言並存。有指定語言就先找那一個,
    找不到再退回沒有語言標記的檔案。
    """
    stem = os.path.splitext(audio_path)[0]

    prefixes = []
    if language:
        # 逐段縮短:zh-Hant-TW → zh-Hant → zh。指定得比檔名細的時候
        # 才找得到(檔名常只寫到 zh-Hant)。
        parts = language.split("-")
        for depth in range(len(parts), 0, -1):
            prefixes.append(f"{stem}.{'-'.join(parts[:depth])}")
    prefixes.append(stem)

    for prefix in prefixes:
        for suffix in SIDECAR_SUFFIXES:
            candidate = prefix + suffix
            if os.path.isfile(candidate):
                return candidate
    return None


def load_file(path: str) -> Lyrics:
    """讀取歌詞檔。編碼常是 UTF-8 或 Big5(中文歌詞檔的老習慣)。"""
    with open(path, "rb") as handle:
        raw = handle.read()

    for encoding in ("utf-8-sig", "utf-8", "big5", "gbk", "shift_jis", "cp1252"):
        try:
            text = raw.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    else:
        text = raw.decode("utf-8", errors="replace")

    lyrics = parse_auto(text)
    lyrics.source = os.path.splitext(path)[1].lstrip(".") or "file"
    return lyrics


# ── YouTube 字幕 ────────────────────────────────────────────────────────
# ── 語言標示 ────────────────────────────────────────────────────────────
# 只放常見的;查不到就直接顯示語言代碼,那比硬猜一個錯的名字好。
_LANGUAGE_NAMES = {
    "zh": "中文", "zh-Hant": "中文(繁體)", "zh-TW": "中文(台灣)",
    "zh-HK": "中文(香港)", "zh-Hans": "中文(簡體)", "zh-CN": "中文(中國)",
    "en": "英文", "ja": "日文", "ko": "韓文", "es": "西班牙文",
    "fr": "法文", "de": "德文", "it": "義大利文", "pt": "葡萄牙文",
    "ru": "俄文", "th": "泰文", "vi": "越南文", "id": "印尼文",
    "ms": "馬來文", "ar": "阿拉伯文", "hi": "印地文",
}


def language_label(code: str) -> str:
    """把語言代碼變成看得懂的名字。"""
    if not code:
        return ""
    if code in _LANGUAGE_NAMES:
        return _LANGUAGE_NAMES[code]
    # zh-Hant-TW 這種:退回去找母語言
    base = code.split("-")[0]
    if base in _LANGUAGE_NAMES:
        return f"{_LANGUAGE_NAMES[base]}({code})"
    return code


def describe_languages(info: dict) -> list[dict]:
    """列出一支影片有哪些字幕語言可選。

    上傳者提供的與自動生成的分開標記 —— 自動字幕的斷句與用字常有明顯
    錯誤,使用者有權知道自己選到的是哪一種。同一個語言兩者都有時只留
    上傳者版本,因為那一定比較好。
    """
    manual = info.get("subtitles") or {}
    automatic = info.get("automatic_captions") or {}

    seen: dict[str, dict] = {}
    for code in manual:
        seen[code] = {"code": code, "label": language_label(code),
                      "auto": False}
    for code in automatic:
        if code in seen:
            continue
        seen[code] = {"code": code, "label": language_label(code),
                      "auto": True}

    def sort_key(entry: dict) -> tuple:
        # 偏好語言排前面,其次是人工字幕,最後照代碼排
        try:
            rank = PREFERRED_LANGUAGES.index(entry["code"])
        except ValueError:
            rank = len(PREFERRED_LANGUAGES)
        return (rank, entry["auto"], entry["code"])

    return sorted(seen.values(), key=sort_key)


def pick_language(available: dict, wanted: str = "") -> str | None:
    """從可用的字幕語言中挑一個。"""
    if not available:
        return None

    if wanted:
        # 使用者明講要哪個語言時,只在「同一個語言」的範圍內找。
        #
        # 絕對不能退回偏好清單:清單上是 de-DE 而使用者輸入 de 的時候,
        # 舊的寫法會一路掉到偏好清單、挑出日文字幕還裝作成功 —— 要德文
        # 卻拿到日文,比直接說「沒有」糟糕得多。
        if wanted in available:
            return wanted
        # de → de-DE、zh-Hant → zh-Hant-TW
        for key in available:
            if key.startswith(wanted + "-"):
                return key
        # 反過來:要 zh-Hant-TW 但清單只到 zh-Hant
        parts = wanted.split("-")
        for depth in range(len(parts) - 1, 0, -1):
            shorter = "-".join(parts[:depth])
            if shorter in available:
                return shorter
        return None

    for language in PREFERRED_LANGUAGES:
        if language in available:
            return language
        # 也接受 zh-Hant-TW 這類帶地區後綴的
        for key in available:
            if key.startswith(language):
                return key
    return next(iter(available), None)


def fetch_youtube(url: str, options=None, language: str = "") -> Lyrics:
    """取得 YouTube 影片附帶的字幕。

    上傳者提供的字幕優先於自動生成的 —— 後者的斷句與用字常有明顯錯誤,
    對照著唱會很痛苦。
    """
    try:
        from yt_dlp import YoutubeDL
    except ImportError:
        return Lyrics(lines=[], source="", synced=False)

    ydl_opts: dict = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        "writesubtitles": False,
    }
    if options is not None:
        if getattr(options, "proxy", ""):
            ydl_opts["proxy"] = options.proxy
        if getattr(options, "cookies_from_browser", ""):
            ydl_opts["cookiesfrombrowser"] = (options.cookies_from_browser,)

    try:
        with YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
    except Exception:
        return Lyrics(lines=[], source="", synced=False)

    if not info:
        return Lyrics(lines=[], source="", synced=False)

    catalogue = describe_languages(info)

    for key, label in (("subtitles", "youtube-manual"),
                       ("automatic_captions", "youtube-auto")):
        tracks = info.get(key) or {}
        chosen = pick_language(tracks, language)
        if not chosen:
            continue
        text = _download_track(tracks[chosen])
        if not text:
            continue
        lyrics = parse_auto(text)
        if lyrics.is_empty:
            continue
        lyrics.source = label
        lyrics.language = chosen
        lyrics.available = catalogue
        return lyrics

    # 一句都沒抓到也要把語言清單帶回去,使用者才能換一種再試
    return Lyrics(lines=[], source="", synced=False, available=catalogue)


def _download_track(formats: list) -> str:
    """從字幕格式清單裡挑一個下載。偏好 vtt / srt 這類純文字格式。"""
    import urllib.request

    def rank(entry: dict) -> int:
        ext = (entry.get("ext") or "").lower()
        # 偏好斷句乾淨、解析穩定的格式。json3 排在 ttml 之前是因為
        # YouTube 的自動字幕在 ttml 下常把同一句拆成很多個 <p>。
        return {"vtt": 0, "srt": 1, "json3": 2, "srv3": 3,
                "ttml": 4, "srv1": 5, "srv2": 6}.get(ext, 9)

    for entry in sorted(formats or [], key=rank):
        url = entry.get("url")
        if not url:
            continue
        try:
            with urllib.request.urlopen(url, timeout=20) as response:
                return response.read().decode("utf-8", errors="replace")
        except Exception:
            continue
    return ""


# ── 統一入口 ────────────────────────────────────────────────────────────
def load_for_media(audio_path: str, source_url: str = "",
                   options=None, language: str = "") -> Lyrics:
    """替一個音源找歌詞。本機檔案優先,再試線上字幕。"""
    sidecar = find_sidecar(audio_path, language) if audio_path else None
    if sidecar:
        lyrics = load_file(sidecar)
        if not lyrics.is_empty:
            return lyrics

    if source_url and source_url.startswith("http"):
        return fetch_youtube(source_url, options, language)

    return Lyrics(lines=[], source="", synced=False)


def current_index(lines: list[LyricLine], position: float) -> int:
    """目前播放位置對應到第幾行。二分搜尋,-1 表示還沒開始。"""
    low, high = 0, len(lines) - 1
    result = -1
    while low <= high:
        mid = (low + high) // 2
        if lines[mid].time <= position:
            result = mid
            low = mid + 1
        else:
            high = mid - 1
    return result
