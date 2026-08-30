"""歌詞解析測試。

樣本一律使用自製的佔位文字,不含任何真實歌詞內容。

    python -m tests.test_lyrics
"""

from __future__ import annotations

import os
import sys
import tempfile

from ktisv_engine.media import lyrics as L

FAILURES: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    mark = "ok  " if condition else "FAIL"
    print(f"  [{mark}] {name}{('  — ' + detail) if detail else ''}")
    if not condition:
        FAILURES.append(name)


SAMPLE_LRC = """[ti:Sample Song]
[ar:Test Artist]

[00:12.50]line one
[00:18.20]line two
[00:24.00]line three
[01:05.75]line four
[00:30.10][01:30.10]repeated chorus
"""

SAMPLE_VTT = """WEBVTT
Kind: captions
Language: zh-Hant

NOTE this is a comment

00:00:05.000 --> 00:00:09.000
first caption

00:00:09.500 --> 00:00:13.000
second caption
continued on next row

00:00:14.000 --> 00:00:18.000
<c.colorE5E5E5>third with tags</c>
"""

SAMPLE_SRT = """1
00:00:03,000 --> 00:00:06,500
srt line one

2
00:00:07,000 --> 00:00:10,000
srt line two
"""

SAMPLE_PLAIN = """plain line A
plain line B

plain line C
"""


def test_lrc() -> None:
    print("LRC 解析")
    result = L.parse_lrc(SAMPLE_LRC)
    check("有解析到行", len(result.lines) == 6, f"{len(result.lines)} 行")
    check("標記為有時間軸", result.synced)
    check("時間正確", abs(result.lines[0].time - 12.5) < 0.01,
          f"{result.lines[0].time}")
    check("依時間排序",
          all(a.time <= b.time for a, b in zip(result.lines, result.lines[1:])))
    check("中繼標籤不會變成歌詞",
          all("ti:" not in line.text for line in result.lines))

    repeated = [line for line in result.lines if line.text == "repeated chorus"]
    check("重複的副歌時間標籤都展開", len(repeated) == 2,
          f"{[round(r.time, 2) for r in repeated]}")


def test_vtt() -> None:
    print("VTT 解析")
    result = L.parse_vtt(SAMPLE_VTT)
    check("有解析到行", len(result.lines) == 3, f"{len(result.lines)} 行")
    check("時間正確", abs(result.lines[0].time - 5.0) < 0.01,
          f"{result.lines[0].time}")
    check("多行字幕會合併",
          "continued" in result.lines[1].text, result.lines[1].text)
    check("HTML 標籤被移除",
          "<" not in result.lines[2].text, result.lines[2].text)
    check("WEBVTT 檔頭與 NOTE 被跳過",
          all("WEBVTT" not in line.text and "comment" not in line.text
              for line in result.lines))


def test_srt() -> None:
    print("SRT 解析")
    result = L.parse_vtt(SAMPLE_SRT)
    check("有解析到行", len(result.lines) == 2, f"{len(result.lines)} 行")
    check("逗號小數點也支援", abs(result.lines[0].time - 3.0) < 0.01,
          f"{result.lines[0].time}")
    check("序號行不會變成歌詞",
          all(not line.text.strip().isdigit() for line in result.lines))


def test_plain() -> None:
    print("純文字")
    result = L.parse_plain(SAMPLE_PLAIN)
    check("空行被略過", len(result.lines) == 3, f"{len(result.lines)} 行")
    check("標記為無時間軸", not result.synced)


def test_auto_detect() -> None:
    print("格式自動判斷")
    check("認出 LRC", L.parse_auto(SAMPLE_LRC).source == "lrc")
    check("認出 VTT", L.parse_auto(SAMPLE_VTT).source == "vtt")
    check("認出 SRT", L.parse_auto(SAMPLE_SRT).source == "vtt")
    check("其餘當純文字", L.parse_auto(SAMPLE_PLAIN).source == "plain")


def test_dedupe() -> None:
    """自動字幕會把同一句重送多次(滾動效果)。"""
    print("重複行去除")
    rolling = """WEBVTT

00:00:01.000 --> 00:00:02.000
hello

00:00:02.000 --> 00:00:03.000
hello

00:00:03.000 --> 00:00:04.000
hello world
"""
    result = L.parse_vtt(rolling)
    check("連續重複被合併", len(result.lines) == 2,
          f"{[line.text for line in result.lines]}")


def test_sidecar() -> None:
    print("本機歌詞檔")
    with tempfile.TemporaryDirectory() as folder:
        audio = os.path.join(folder, "song.mp3")
        with open(audio, "wb") as handle:
            handle.write(b"fake")
        check("沒有歌詞檔時回傳 None", L.find_sidecar(audio) is None)

        lrc = os.path.join(folder, "song.lrc")
        with open(lrc, "w", encoding="utf-8") as handle:
            handle.write(SAMPLE_LRC)
        check("找得到同名 .lrc", L.find_sidecar(audio) == lrc)

        loaded = L.load_file(lrc)
        check("讀得出內容", len(loaded.lines) == 6, f"{len(loaded.lines)} 行")

        # Big5 編碼(中文歌詞檔的老習慣)
        big5 = os.path.join(folder, "big5.lrc")
        with open(big5, "wb") as handle:
            handle.write("[00:01.00]測試中文\n".encode("big5"))
        result = L.load_file(big5)
        check("Big5 編碼可讀", bool(result.lines) and "測試" in result.lines[0].text,
              result.lines[0].text if result.lines else "(空)")


def test_language_pick() -> None:
    print("字幕語言選擇")
    check("優先繁中",
          L.pick_language({"en": [], "zh-Hant": [], "ja": []}) == "zh-Hant")
    check("沒有繁中時退而求其次",
          L.pick_language({"en": [], "ja": []}) == "ja")
    check("接受帶地區後綴",
          L.pick_language({"zh-Hant-TW": [], "en": []}) == "zh-Hant-TW")
    check("指定語言優先",
          L.pick_language({"en": [], "zh-Hant": []}, wanted="en") == "en")
    check("沒有任何字幕回傳 None", L.pick_language({}) is None)


def test_current_index() -> None:
    print("播放位置對應行號")
    lines = [L.LyricLine(t, f"line {i}")
             for i, t in enumerate([0.0, 10.0, 20.0, 30.0])]
    check("開頭前回傳 -1", L.current_index(lines, -1.0) == -1)
    check("剛好在第一行", L.current_index(lines, 0.0) == 0)
    check("兩行之間取前一行", L.current_index(lines, 15.0) == 1)
    check("剛好在某行", L.current_index(lines, 20.0) == 2)
    check("超過最後一行", L.current_index(lines, 999.0) == 3)
    check("空歌詞不會爆", L.current_index([], 5.0) == -1)


SAMPLE_ASS = """[Script Info]
Title: 測試
ScriptType: v4.00+

[V4+ Styles]
Format: Name, Fontname, Fontsize
Style: Default,Arial,48

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
Dialogue: 0,0:00:12.34,0:00:15.00,Default,,0,0,0,,{\\k50}第一句,含逗號
Comment: 0,0:00:13.00,0:00:14.00,Default,,0,0,0,,這行是註解不該出現
Dialogue: 0,0:00:15.00,0:00:18.00,Default,,0,0,0,,{\\pos(190,220)}第二句\\N續行
"""

SAMPLE_SSA_REORDERED = """[Events]
Format: Marked, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
Dialogue: Marked=0,0:00:05.00,0:00:07.00,Default,,0,0,0,,欄位順序不同
"""

SAMPLE_TTML = """<?xml version="1.0" encoding="utf-8"?>
<tt xmlns="http://www.w3.org/ns/ttml">
  <body><div>
    <p begin="00:00:12.340" end="00:00:15.000">第一句</p>
    <p begin="15.0s" end="18s"><span>第二</span><span>句</span></p>
    <p end="20s">沒有 begin 的要略過</p>
  </div></body>
</tt>
"""

SAMPLE_JSON3 = ('{"events":[{"tStartMs":12340,"segs":[{"utf8":"第一"},'
                '{"utf8":"句"}]},{"tStartMs":15000,"segs":[{"utf8":"第二句"}]},'
                '{"tStartMs":16000},{"tStartMs":17000,"segs":[{"utf8":"\\n"}]}]}')


def test_ass() -> None:
    print("ASS / SSA")
    result = L.parse_ass(SAMPLE_ASS)
    check("有解析到行", len(result.lines) == 2, f"{len(result.lines)} 行")
    check("時間是百分之一秒", abs(result.lines[0].time - 12.34) < 0.005,
          f"{result.lines[0].time:.3f}s")
    check("覆寫標籤被移除", "{" not in result.lines[0].text
          and "\\k" not in result.lines[0].text, result.lines[0].text)
    check("文字裡的逗號不會被切斷",
          result.lines[0].text == "第一句,含逗號", result.lines[0].text)
    check("換行標記變成空白",
          result.lines[1].text == "第二句 續行", result.lines[1].text)
    check("Comment 不會變成歌詞",
          all("註解" not in line.text for line in result.lines))

    reordered = L.parse_ass(SAMPLE_SSA_REORDERED)
    check("欄位順序照 Format 那行走",
          len(reordered.lines) == 1
          and abs(reordered.lines[0].time - 5.0) < 0.01,
          f"{reordered.lines[0].time if reordered.lines else '無'}")


def test_ttml() -> None:
    print("TTML / DFXP")
    result = L.parse_ttml(SAMPLE_TTML)
    check("有解析到行", len(result.lines) == 2, f"{len(result.lines)} 行")
    check("時鐘格式的時間正確", abs(result.lines[0].time - 12.34) < 0.005,
          f"{result.lines[0].time:.3f}s")
    check("秒數偏移格式也支援", abs(result.lines[1].time - 15.0) < 0.01,
          f"{result.lines[1].time:.3f}s")
    check("巢狀 span 會被收齊",
          result.lines[1].text == "第二 句", result.lines[1].text)
    check("沒有 begin 的段落被略過",
          all("略過" not in line.text for line in result.lines))
    check("壞掉的 XML 不會拋例外", L.parse_ttml("<tt><unclosed").is_empty)


def test_json3() -> None:
    print("YouTube json3")
    result = L.parse_json3(SAMPLE_JSON3)
    check("有解析到行", len(result.lines) == 2, f"{len(result.lines)} 行")
    check("同一個 event 的逐字片段會併成一句",
          result.lines[0].text == "第一句", result.lines[0].text)
    check("時間由毫秒換算", abs(result.lines[0].time - 12.34) < 0.005,
          f"{result.lines[0].time:.3f}s")
    check("沒有 segs 或只有換行的 event 被略過", len(result.lines) == 2)
    check("壞掉的 JSON 不會拋例外", L.parse_json3("{壞掉").is_empty)


def test_auto_detect_new() -> None:
    print("自動辨識 —— 新格式")
    check("認出 ASS", L.parse_auto(SAMPLE_ASS).source == "ass")
    check("認出 TTML", L.parse_auto(SAMPLE_TTML).source == "ttml")
    check("認出 json3", L.parse_auto(SAMPLE_JSON3).source == "json3")
    # 舊格式不能因為新增判斷而被搶走
    check("LRC 仍然認得出", L.parse_auto(SAMPLE_LRC).source == "lrc")
    check("VTT 仍然認得出", L.parse_auto(SAMPLE_VTT).source == "vtt")
    check("SRT 仍然認得出", L.parse_auto(SAMPLE_SRT).source == "vtt")
    check("純文字仍然認得出", L.parse_auto("只是一些字\n沒有時間").source == "plain")


def test_language_pick() -> None:
    print("字幕語言挑選")
    available = {"en": [], "ja": [], "zh-Hant": [], "zh-Hans": []}
    check("預設偏好繁體中文", L.pick_language(available) == "zh-Hant")
    check("指定語言優先", L.pick_language(available, "ja") == "ja")
    # 明確指定的語言不存在時要回 None,不能悄悄換成別的語言 ——
    # 清單上是 de-DE 而使用者輸入 de 的舊寫法會掉進偏好清單挑出日文,
    # 「要德文卻拿到日文」比誠實地說沒有糟得多。
    check("指定的語言不存在時回 None,不會換成別的語言",
          L.pick_language(available, "ko") is None)
    check("地區變體算同一個語言",
          L.pick_language({"de-DE": [], "ja": []}, "de") == "de-DE")
    check("指定得比清單細也對得上",
          L.pick_language({"zh-Hant": [], "ja": []}, "zh-Hant-TW") == "zh-Hant")
    check("帶地區後綴也算數",
          L.pick_language({"zh-Hant-TW": []}) == "zh-Hant-TW")
    check("完全沒有時回 None", L.pick_language({}) is None)
    check("只有一種語言就用它", L.pick_language({"ko": []}) == "ko")


def main() -> int:
    tests = (test_lrc, test_vtt, test_srt, test_plain, test_auto_detect,
             test_dedupe, test_sidecar, test_language_pick, test_current_index,
             test_ass, test_ttml, test_json3,
             test_auto_detect_new, test_language_pick)
    for fn in tests:
        fn()
        print()
    if FAILURES:
        print(f"{len(FAILURES)} 項未通過: " + ", ".join(FAILURES))
        return 1
    print("全部通過。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
