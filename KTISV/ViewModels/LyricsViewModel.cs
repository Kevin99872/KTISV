using System;
using System.Collections.ObjectModel;
using System.Linq;
using System.Text.Json;
using CommunityToolkit.Mvvm.ComponentModel;

namespace KTISV.ViewModels
{
    /// <summary>歌詞的一行。</summary>
    public sealed partial class LyricLineViewModel(double time, string text) : ViewModelBase
    {
        public double Time { get; } = time;
        public string Text { get; } = text;

        /// <summary>正在唱的那一行。用來做高亮。</summary>
        [ObservableProperty] private bool _isCurrent;

        public string TimeText => TimeSpan.FromSeconds(Time).ToString(@"m\:ss");
    }

    /// <summary>
    /// 歌詞面板的狀態。
    ///
    /// 歌詞來源有兩層(本機 .lrc → 線上字幕),取得是非同步的,所以這裡要能
    /// 表達「還在找」「找不到」等中間狀態 —— 而不是只有「有」跟「沒有」。
    /// </summary>
    public sealed partial class LyricsViewModel : ViewModelBase
    {
        public ObservableCollection<LyricLineViewModel> Lines { get; } = [];

        [ObservableProperty] private int _currentIndex = -1;
        [ObservableProperty] private string _status = "";
        [ObservableProperty] private bool _isWorking;
        [ObservableProperty] private string _sourceLabel = "";
        [ObservableProperty] private bool _isSynced = true;

        /// <summary>字級。使用者可調 —— 投影或站遠一點時需要放大。</summary>
        [ObservableProperty] private double _fontSize = 20;

        [ObservableProperty] private bool _autoScroll = true;

        /// <summary>
        /// 這支影片有哪些字幕語言可選。第一項固定是「自動」——
        /// 讓引擎照偏好順序挑,那是多數人要的。
        /// </summary>
        public ObservableCollection<LyricLanguage> Languages { get; } = [];

        [ObservableProperty] private LyricLanguage? _selectedLanguage;

        public bool HasLanguages => Languages.Count > 1;

        /// <summary>切換語言時要通知外面重新抓 —— VM 自己不會講 IPC。</summary>
        public Action<string>? LanguageChanged { get; set; }

        /// <summary>回填語言清單時抑制「選取變更 → 重新抓」的回授。</summary>
        private bool _suppressLanguagePush;

        partial void OnSelectedLanguageChanged(LyricLanguage? value)
        {
            if (_suppressLanguagePush || value is null) return;
            LanguageChanged?.Invoke(value.Code);
        }

        /// <summary>
        /// 更新可選語言。保留使用者目前選的那一個 —— 換歌時清單會重建,
        /// 但「我要看日文」這個意圖不該被換歌洗掉。
        /// </summary>
        public void SetLanguages(JsonElement data, string activeCode)
        {
            var wanted = SelectedLanguage?.Code ?? "";
            _suppressLanguagePush = true;
            try
            {
                Languages.Clear();
                Languages.Add(LyricLanguage.Auto);

                if (data.ValueKind == JsonValueKind.Object
                    && data.TryGetProperty("available", out var list)
                    && list.ValueKind == JsonValueKind.Array)
                {
                    foreach (var item in list.EnumerateArray())
                    {
                        var code = item.TryGetProperty("code", out var c)
                            ? c.GetString() ?? "" : "";
                        if (string.IsNullOrEmpty(code)) continue;
                        var label = item.TryGetProperty("label", out var l)
                            ? l.GetString() ?? code : code;
                        var auto = item.TryGetProperty("auto", out var a) && a.GetBoolean();
                        Languages.Add(new LyricLanguage(code, label, auto));
                    }
                }

                SelectedLanguage =
                    Languages.FirstOrDefault(x => x.Code == wanted)
                    ?? Languages.FirstOrDefault(x => x.Code == activeCode)
                    ?? Languages[0];
            }
            finally
            {
                _suppressLanguagePush = false;
            }
            OnPropertyChanged(nameof(HasLanguages));
        }

        public bool HasLines => Lines.Count > 0;

        /// <summary>目前這一行,給 View 做自動捲動。</summary>
        public LyricLineViewModel? CurrentLine =>
            CurrentIndex >= 0 && CurrentIndex < Lines.Count ? Lines[CurrentIndex] : null;

        partial void OnCurrentIndexChanged(int value)
        {
            for (var i = 0; i < Lines.Count; i++)
                Lines[i].IsCurrent = i == value;
            OnPropertyChanged(nameof(CurrentLine));
        }

        public void Clear()
        {
            Lines.Clear();
            CurrentIndex = -1;
            SourceLabel = "";
            OnPropertyChanged(nameof(HasLines));
            OnPropertyChanged(nameof(CurrentLine));
        }

        /// <summary>從引擎送來的歌詞資料填入。</summary>
        public void Load(JsonElement data)
        {
            Lines.Clear();
            CurrentIndex = -1;

            if (data.ValueKind == JsonValueKind.Object
                && data.TryGetProperty("lines", out var lines)
                && lines.ValueKind == JsonValueKind.Array)
            {
                foreach (var item in lines.EnumerateArray())
                {
                    var time = item.TryGetProperty("time", out var t) ? t.GetDouble() : 0;
                    var text = item.TryGetProperty("text", out var s) ? s.GetString() ?? "" : "";
                    if (!string.IsNullOrWhiteSpace(text))
                        Lines.Add(new LyricLineViewModel(time, text));
                }
            }

            IsSynced = !data.TryGetProperty("synced", out var synced) || synced.GetBoolean();
            SourceLabel = DescribeSource(
                data.TryGetProperty("source", out var src) ? src.GetString() : "",
                data.TryGetProperty("language", out var lang) ? lang.GetString() : "");

            SetLanguages(data,
                data.TryGetProperty("language", out var active)
                    ? active.GetString() ?? "" : "");

            IsWorking = false;
            Status = Lines.Count > 0 ? "" : "這首歌找不到歌詞";
            OnPropertyChanged(nameof(HasLines));
            OnPropertyChanged(nameof(CurrentLine));
        }

        private static string DescribeSource(string? source, string? language)
        {
            var name = source switch
            {
                "lrc" => "本機歌詞檔",
                "vtt" or "srt" => "本機字幕檔",
                "youtube-manual" => "YouTube 字幕",
                "youtube-auto" => "YouTube 自動字幕",
                "plain" => "純文字",
                _ => "",
            };
            return string.IsNullOrEmpty(language) ? name : $"{name} · {language}";
        }

        /// <summary>依播放位置更新目前行。二分搜尋,每秒呼叫多次也不費力。</summary>
        public void UpdatePosition(double seconds)
        {
            if (Lines.Count == 0 || !IsSynced) return;

            int low = 0, high = Lines.Count - 1, found = -1;
            while (low <= high)
            {
                var mid = (low + high) / 2;
                if (Lines[mid].Time <= seconds)
                {
                    found = mid;
                    low = mid + 1;
                }
                else
                {
                    high = mid - 1;
                }
            }

            if (found != CurrentIndex) CurrentIndex = found;
        }
    }

    /// <summary>一個可選的字幕語言。</summary>
    public sealed record LyricLanguage(string Code, string Label, bool IsAuto)
    {
        /// <summary>「自動」= 交給引擎照偏好順序挑。</summary>
        public static readonly LyricLanguage Auto = new("", "自動選擇", false);

        /// <summary>自動生成的字幕要標出來 —— 品質差很多,使用者有權知道。</summary>
        public string Display => IsAuto ? $"{Label}(自動生成)" : Label;
    }
}
