using System;
using System.Collections.Generic;

namespace KTISV.Services
{
    /// <summary>
    /// 介面字串的本地化。
    ///
    /// 刻意做得很簡單:一個以 key 查表的字典,查不到就回傳 key 本身。
    /// 這讓字串可以逐步搬進來 —— 還沒搬的地方仍然顯示原本的中文,
    /// 不會因為缺翻譯就變成空白或崩潰。
    /// </summary>
    public static class Localization
    {
        public const string TraditionalChinese = "zh-Hant";
        public const string English = "en";

        private static string _current = TraditionalChinese;

        /// <summary>語言變更時觸發,讓 ViewModel 重新通知綁定。</summary>
        public static event Action? LanguageChanged;

        public static string Current
        {
            get => _current;
            set
            {
                var code = Normalize(value);
                if (_current == code) return;
                _current = code;
                LanguageChanged?.Invoke();
            }
        }

        public static bool IsEnglish => _current == English;

        private static string Normalize(string? code) =>
            string.IsNullOrWhiteSpace(code) ? TraditionalChinese
            : code.StartsWith("en", StringComparison.OrdinalIgnoreCase) ? English
            : TraditionalChinese;

        public static int IndexOf(string? code) => Normalize(code) == English ? 1 : 0;

        /// <summary>查表。找不到就回傳 key,避免缺翻譯造成空白。</summary>
        public static string T(string key)
        {
            var table = IsEnglish ? EnglishTable : ChineseTable;
            return table.TryGetValue(key, out var value) ? value : key;
        }

        /// <summary>依目前語言選一個字串。字串少的地方直接用這個比查表方便。</summary>
        public static string Pick(string chinese, string english)
            => IsEnglish ? english : chinese;

        private static readonly Dictionary<string, string> ChineseTable = new()
        {
            ["tab.youtube"] = "YouTube 搜尋",
            ["tab.karaoke"] = "卡拉 OK",
            ["tab.eq"] = "EQ 調整",
            ["tab.settings"] = "其他設定",
            ["transport.play"] = "播放",
            ["transport.pause"] = "暫停",
            ["transport.stop"] = "停止",
            ["audio.start"] = "▶  啟動音訊",
            ["audio.stop"] = "■  停止音訊",
        };

        private static readonly Dictionary<string, string> EnglishTable = new()
        {
            ["tab.youtube"] = "YouTube",
            ["tab.karaoke"] = "Karaoke",
            ["tab.eq"] = "Equalizer",
            ["tab.settings"] = "Settings",
            ["transport.play"] = "Play",
            ["transport.pause"] = "Pause",
            ["transport.stop"] = "Stop",
            ["audio.start"] = "▶  Start audio",
            ["audio.stop"] = "■  Stop audio",
        };
    }
}
