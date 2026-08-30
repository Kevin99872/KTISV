"""用 yt-dlp 取得影片的音訊軌。

只抓 bestaudio 的單一串流,不做 muxing,所以下載階段本身不需要 ffmpeg;
解碼交給 ``media.ffmpeg``。
"""

from __future__ import annotations

import hashlib
import os
import time
import re
from dataclasses import dataclass, field
from typing import Callable

from ..paths import cache_dir

_SAFE = re.compile(r"[^\w\-. ]+", re.UNICODE)


@dataclass
class MediaInfo:
    path: str
    title: str = ""
    duration: float = 0.0
    uploader: str = ""
    webpage_url: str = ""
    thumbnail: str = ""

    def to_dict(self) -> dict:
        return {
            "path": self.path,
            "title": self.title,
            "duration": self.duration,
            "uploader": self.uploader,
            "url": self.webpage_url,
            "thumbnail": self.thumbnail,
        }


@dataclass
class SearchResult:
    """搜尋結果的一筆。只做 flat 解析,不碰實際串流,所以很快。"""

    url: str
    title: str
    duration: float = 0.0
    uploader: str = ""
    video_id: str = ""
    thumbnail: str = ""
    view_count: int = 0

    @property
    def duration_text(self) -> str:
        if not self.duration:
            return "—"
        total = int(self.duration)
        hours, rest = divmod(total, 3600)
        minutes, seconds = divmod(rest, 60)
        if hours:
            return f"{hours}:{minutes:02d}:{seconds:02d}"
        return f"{minutes}:{seconds:02d}"

    def to_dict(self) -> dict:
        return {
            "url": self.url,
            "title": self.title,
            "duration": self.duration,
            "duration_text": self.duration_text,
            "uploader": self.uploader,
            "video_id": self.video_id,
            "thumbnail": self.thumbnail,
            "views": self.view_count,
        }


@dataclass
class DownloadOptions:
    cookies_from_browser: str = ""
    proxy: str = ""
    extra_args: list[str] = field(default_factory=list)


class YoutubeError(RuntimeError):
    pass


def is_url(text: str) -> bool:
    return bool(re.match(r"^https?://", text.strip(), re.IGNORECASE))


def safe_name(text: str, limit: int = 60) -> str:
    cleaned = _SAFE.sub("_", text).strip().strip(".")
    return (cleaned or "audio")[:limit]



# 暫時性的失敗:重試通常就會過。YouTube 對「短時間抓很多首」會回 403 或
# 429,那不是影片有問題 —— 載一整個歌單很容易踩到。
_TRANSIENT = ("403", "429", "too many requests", "timed out", "timeout",
              "connection reset", "temporarily", "unable to download video data")

# 一看就知道重試也沒用的,直接翻成看得懂的話,不要讓使用者盯著英文堆疊。
_FATAL_HINTS = (
    ("sign in to confirm", "YouTube 要求登入驗證。請在設定裡指定要借用 Cookie 的瀏覽器。"),
    ("age-restricted", "這是年齡限制影片,需要登入才能取得。"),
    ("private video", "這是私人影片。"),
    ("members-only", "這是會員限定影片。"),
    # yt-dlp 的措辭不只一種:「Video unavailable」「This video is unavailable」
    # 「is not available in your country」都出現過,所以比對放寬到 unavailable。
    ("unavailable", "這支影片已被移除、設為私人,或在你的地區無法播放。"),
    ("is not available", "這支影片在你的地區無法播放。"),
    ("live event", "這是直播,無法當成音源下載。"),
)


def _describe_failure(message: str) -> tuple[str, bool]:
    """把 yt-dlp 的錯誤翻成人話。回傳 (訊息, 是否值得重試)。"""
    low = message.lower()
    for needle, friendly in _FATAL_HINTS:
        if needle in low:
            return friendly, False
    if any(needle in low for needle in _TRANSIENT):
        return ("YouTube 暫時擋下了這次下載(通常是短時間內抓太多首)。"
                "已自動重試數次仍失敗,請等一兩分鐘再試。"), True
    return f"下載失敗: {message}", False


def download_audio(url: str,
                   options: DownloadOptions | None = None,
                   progress: Callable[[str, float], None] | None = None) -> MediaInfo:
    """下載並回傳本機音訊檔資訊。相同網址會重用快取。"""
    try:
        from yt_dlp import YoutubeDL
    except ImportError as exc:  # pragma: no cover - 由 UI 顯示
        raise YoutubeError(
            "尚未安裝 yt-dlp。請執行 `pip install -r engine/requirements.txt`。"
        ) from exc

    options = options or DownloadOptions()
    out_dir = cache_dir("downloads")
    stem = hashlib.sha1(url.encode("utf-8")).hexdigest()[:16]

    def hook(status: dict) -> None:
        if progress is None:
            return
        if status.get("status") == "downloading":
            total = status.get("total_bytes") or status.get("total_bytes_estimate") or 0
            done = status.get("downloaded_bytes") or 0
            pct = (done / total) if total else 0.0
            progress("下載中", min(max(pct, 0.0), 1.0))
        elif status.get("status") == "finished":
            progress("下載完成", 1.0)

    ydl_opts: dict = {
        # 優先挑 48 kHz 的純音訊串流。
        #
        # 引擎內部固定跑 48 kHz,所以 48 kHz 的來源可以一路直通、完全不用
        # 重取樣。YouTube 的 opus 一律是 48 kHz,而同一支影片的 m4a 通常是
        # 44.1 kHz —— 兩者位元率差不多(實測 128.9 vs 129.5 kbps),但選到
        # m4a 就會多一次重取樣。
        #
        # (試過把 ffmpeg 的重取樣器調好一點,實測只從 −89.0 dB 進步到
        #  −91.1 dB,不值得為此加參數;直接不重取樣才是對的解法。)
        "format": "bestaudio[asr=48000]/bestaudio/best",
        "outtmpl": os.path.join(out_dir, f"{stem}.%(ext)s"),
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "nocheckcertificate": False,
        "retries": 3,
        "progress_hooks": [hook],
        "overwrites": False,
        "continuedl": True,
    }
    if options.proxy:
        ydl_opts["proxy"] = options.proxy
    if options.cookies_from_browser:
        ydl_opts["cookiesfrombrowser"] = (options.cookies_from_browser,)

    if progress:
        progress("解析網址", 0.0)

    # 暫時性失敗要自己退避重試。限流是「許多歌曲抓不下來」最常見的原因,
    # 而它幾乎都會在幾秒到一分鐘內自己好 —— 直接把錯誤丟給使用者,看起來
    # 就像那首歌壞掉,其實只是抓太快。
    attempts = 3
    last_message = ""
    for attempt in range(attempts):
        try:
            with YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                if info.get("_type") == "playlist":
                    entries = info.get("entries") or []
                    if not entries:
                        raise YoutubeError("這個網址沒有可用的影片。")
                    info = entries[0]
                path = ydl.prepare_filename(info)
            break
        except YoutubeError:
            raise
        except Exception as exc:
            last_message = str(exc)
            friendly, retryable = _describe_failure(last_message)
            if not retryable or attempt == attempts - 1:
                raise YoutubeError(friendly) from exc
            wait = 3.0 * (attempt + 1)          # 3 秒、6 秒
            if progress:
                progress(f"被暫時擋下,{wait:.0f} 秒後重試", 0.0)
            time.sleep(wait)

    if not os.path.isfile(path):
        # 極少數情況副檔名會被改寫,退而求其次掃描快取目錄
        for name in os.listdir(out_dir):
            if name.startswith(stem):
                path = os.path.join(out_dir, name)
                break
    if not os.path.isfile(path):
        raise YoutubeError("下載完成但找不到輸出檔案。")

    return MediaInfo(
        path=path,
        title=info.get("title") or os.path.basename(path),
        duration=float(info.get("duration") or 0.0),
        uploader=info.get("uploader") or "",
        webpage_url=info.get("webpage_url") or url,
        thumbnail=info.get("thumbnail") or "",
    )


def local_file_info(path: str) -> MediaInfo:
    return MediaInfo(path=path, title=os.path.splitext(os.path.basename(path))[0])


def search(query: str, limit: int = 12,
           options: DownloadOptions | None = None) -> list[SearchResult]:
    """用關鍵字搜尋 YouTube,回傳候選清單。

    只做 flat 解析(不解析各影片的串流網址),一次查詢大約一兩秒。
    真正要播的那一首才會走 ``download_audio`` 拿完整資訊。
    """
    query = (query or "").strip()
    if not query:
        return []

    try:
        from yt_dlp import YoutubeDL
    except ImportError as exc:
        raise YoutubeError(
            "尚未安裝 yt-dlp。請在 engine 目錄執行 `uv sync`。"
        ) from exc

    options = options or DownloadOptions()
    limit = max(1, min(30, int(limit)))

    ydl_opts: dict = {
        "quiet": True,
        "no_warnings": True,
        "extract_flat": True,
        "skip_download": True,
        "default_search": "ytsearch",
    }
    if options.proxy:
        ydl_opts["proxy"] = options.proxy
    if options.cookies_from_browser:
        ydl_opts["cookiesfrombrowser"] = (options.cookies_from_browser,)

    try:
        with YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(f"ytsearch{limit}:{query}", download=False)
    except Exception as exc:
        raise YoutubeError(f"搜尋失敗: {exc}") from exc

    results: list[SearchResult] = []
    for entry in (info or {}).get("entries") or []:
        if not entry:
            continue
        video_id = entry.get("id") or ""
        url = entry.get("url") or entry.get("webpage_url") or ""
        if url and not url.startswith("http") and video_id:
            url = f"https://www.youtube.com/watch?v={video_id}"
        if not url:
            continue
        results.append(SearchResult(
            url=url,
            title=entry.get("title") or "(無標題)",
            duration=float(entry.get("duration") or 0.0),
            uploader=entry.get("uploader") or entry.get("channel") or "",
            video_id=video_id,
            thumbnail=_first_thumbnail(entry),
            view_count=int(entry.get("view_count") or 0),
        ))
    return results


def _first_thumbnail(entry: dict) -> str:
    thumbnails = entry.get("thumbnails") or []
    if thumbnails:
        return thumbnails[0].get("url") or ""
    return entry.get("thumbnail") or ""
