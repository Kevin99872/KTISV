using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Threading.Tasks;
using Avalonia.Threading;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using KTISV.Models;
using KTISV.Services;

namespace KTISV.ViewModels
{
    public sealed partial class MainWindowViewModel : ViewModelBase, IAsyncDisposable, IEqBridge
    {
        private readonly EngineProcess _process = new();
        private readonly EngineClient _client = new();
        private ParameterSender? _sender;

        /// <summary>從引擎回填狀態時抑制「屬性變更 → 送指令」的回授。</summary>
        private bool _suppressPush;
        private bool _userIsSeeking;

        public MainWindowViewModel()
        {
            // 歌詞語言換了就重新抓一次。VM 自己不講 IPC,所以由這裡轉手。
            Lyrics.LanguageChanged = code =>
            {
                if (_suppressPush) return;
                Lyrics.IsWorking = true;
                Lyrics.Status = "換語言,重新尋找歌詞…";
                _client.Post("set_lyrics_options", new { language = code });
            };

            MusicEq = new EqViewModel("music", "影片 / 音樂 EQ", DefaultEqBands(), this);
            MicEq = new EqViewModel("mic", "麥克風 EQ", DefaultEqBands(), this);
            _process.DiagnosticReceived += line => Dispatcher.UIThread.Post(() => AppendLog(line));
        }

        private static readonly double[] EqFrequencies =
            [31.25, 62.5, 125, 250, 500, 1000, 2000, 4000, 8000, 16000];

        /// <summary>
        /// 連上引擎之前先擺出來的預設配置,連上之後會被 eq_info 的真實內容取代。
        /// 兩端是 shelf,Q 在那裡代表轉折斜率而不是頻寬,所以用比較平緩的值。
        /// </summary>
        private static EqBandSpec[] DefaultEqBands() =>
            [.. EqFrequencies.Select((hz, i) => new EqBandSpec(
                hz, 0, i == 0 || i == EqFrequencies.Length - 1 ? 0.7 : 1.41))];

        // ── 電平表 ──────────────────────────────────────────────────────
        public MeterViewModel MusicInMeter { get; } = new("音樂輸入");
        public MeterViewModel MusicOutMeter { get; } = new("音樂");
        public MeterViewModel MicInMeter { get; } = new("麥克風輸入");
        public MeterViewModel MicOutMeter { get; } = new("麥克風");
        public MeterViewModel HeadphoneMeter { get; } = new("耳機輸出");
        public MeterViewModel VirtualMeter { get; } = new("虛擬麥克風輸出");

        public EqViewModel MusicEq { get; }
        public EqViewModel MicEq { get; }

        // ── 連線 / 狀態 ─────────────────────────────────────────────────
        [ObservableProperty] private bool _isConnected;
        [ObservableProperty] private string _statusText = "尚未連線";
        [ObservableProperty] private string _engineVersion = "";
        [ObservableProperty] private bool _isAudioRunning;
        [ObservableProperty] private bool _hasFfmpeg = true;
        [ObservableProperty] private string _diagnosticsText = "";

        public ObservableCollection<string> LogLines { get; } = [];

        public string AudioButtonText => IsAudioRunning ? "■  停止音訊" : "▶  啟動音訊";

        partial void OnIsAudioRunningChanged(bool value) => OnPropertyChanged(nameof(AudioButtonText));

        // ── 音源 ────────────────────────────────────────────────────────
        [ObservableProperty] private string _sourceInput = "";
        [ObservableProperty] private bool _isBusy;
        [ObservableProperty] private double _progress;
        [ObservableProperty] private string _progressStage = "";
        [ObservableProperty] private string _trackTitle = "尚未載入音源";

        /// <summary>由 View 注入的檔案選擇器(需要視窗才能開對話框)。</summary>
        public Func<Task<string?>>? PickAudioFileAsync { get; set; }

        // ── 搜尋 ────────────────────────────────────────────────────────
        public ObservableCollection<SearchResult> SearchResults { get; } = [];

        [ObservableProperty] private bool _isSearching;
        [ObservableProperty] private string _searchHint = "";
        [ObservableProperty] private SearchResult? _selectedResult;

        public bool HasSearchResults => SearchResults.Count > 0;

        /// <summary>點選搜尋結果就直接載入播放。</summary>
        partial void OnSelectedResultChanged(SearchResult? value)
        {
            if (_suppressPush || value is null) return;
            PlayItem(value);
        }

        [RelayCommand]
        private void ClearSearch()
        {
            SearchResults.Clear();
            SelectedResult = null;
            SearchHint = "";
            OnPropertyChanged(nameof(HasSearchResults));
        }

        // ── 歌詞 ────────────────────────────────────────────────────────
        public LyricsViewModel Lyrics { get; } = new();

        [RelayCommand]
        private async Task RefreshLyricsAsync()
        {
            Lyrics.Clear();
            Lyrics.IsWorking = true;
            Lyrics.Status = "重新尋找歌詞…";
            if (!await SafeCallAsync("get_lyrics", new { refresh = true }))
            {
                Lyrics.IsWorking = false;
                Lyrics.Status = "重新尋找失敗";
            }
        }

        // ── 歌單 ────────────────────────────────────────────────────────
        public ObservableCollection<SearchResult> Playlist { get; } = [];

        [ObservableProperty] private SearchResult? _currentTrack;
        [ObservableProperty] private bool _autoAdvance = true;

        public bool HasPlaylist => Playlist.Count > 0;

        public string PlaylistSummary => Playlist.Count == 0
            ? "歌單是空的 —— 從搜尋結果按「加入」"
            : $"{Playlist.Count} 首"
              + (CurrentTrack is null ? "" : $"  ·  正在播放第 {Playlist.IndexOf(CurrentTrack) + 1} 首");

        private void RaisePlaylistChanged()
        {
            OnPropertyChanged(nameof(HasPlaylist));
            OnPropertyChanged(nameof(PlaylistSummary));
        }

        partial void OnCurrentTrackChanged(SearchResult? value)
            => OnPropertyChanged(nameof(PlaylistSummary));

        [RelayCommand]
        private void AddToPlaylist(SearchResult? item)
        {
            if (item is null) return;
            // 同一首不重複加入 —— 以網址為識別
            if (Playlist.Any(t => t.Url == item.Url))
            {
                StatusText = $"「{item.Title}」已在歌單中";
                return;
            }
            Playlist.Add(item);
            RaisePlaylistChanged();
            StatusText = $"已加入歌單:{item.Title}";
        }

        [RelayCommand]
        private void AddAllResults()
        {
            var added = 0;
            foreach (var item in SearchResults)
            {
                if (Playlist.Any(t => t.Url == item.Url)) continue;
                Playlist.Add(item);
                added++;
            }
            RaisePlaylistChanged();
            StatusText = added > 0 ? $"已加入 {added} 首" : "這些歌都已在歌單中";
        }

        [RelayCommand]
        private void RemoveFromPlaylist(SearchResult? item)
        {
            if (item is null) return;
            Playlist.Remove(item);
            if (ReferenceEquals(CurrentTrack, item)) CurrentTrack = null;
            RaisePlaylistChanged();
        }

        [RelayCommand]
        private void ClearPlaylist()
        {
            Playlist.Clear();
            CurrentTrack = null;
            RaisePlaylistChanged();
        }

        [RelayCommand]
        private void MoveUp(SearchResult? item)
        {
            if (item is null) return;
            var index = Playlist.IndexOf(item);
            if (index > 0) Playlist.Move(index, index - 1);
            RaisePlaylistChanged();
        }

        [RelayCommand]
        private void MoveDown(SearchResult? item)
        {
            if (item is null) return;
            var index = Playlist.IndexOf(item);
            if (index >= 0 && index < Playlist.Count - 1) Playlist.Move(index, index + 1);
            RaisePlaylistChanged();
        }

        /// <summary>播放歌單裡的指定曲目。</summary>
        [RelayCommand]
        private void PlayItem(SearchResult? item)
        {
            if (item is null) return;
            CurrentTrack = item;
            SourceInput = item.Url;
            TrackTitle = item.Title;
            _ = LoadUrlAsync(item.Url);
        }

        [RelayCommand]
        private void PlayNext() => Advance(1);

        [RelayCommand]
        private void PlayPrevious() => Advance(-1);

        private void Advance(int direction)
        {
            if (Playlist.Count == 0) return;
            var index = CurrentTrack is null ? -1 : Playlist.IndexOf(CurrentTrack);
            var next = index + direction;

            if (next < 0) next = Loop ? Playlist.Count - 1 : 0;
            else if (next >= Playlist.Count)
            {
                if (!Loop)
                {
                    StatusText = "歌單已播完";
                    return;
                }
                next = 0;
            }
            PlayItem(Playlist[next]);
        }

        // ── 傳輸 ────────────────────────────────────────────────────────
        [ObservableProperty] private bool _isPlaying;
        [ObservableProperty] private double _positionSeconds;
        [ObservableProperty] private double _durationSeconds;
        [ObservableProperty] private bool _loop;

        public string PositionText => FormatTime(PositionSeconds);
        public string DurationText => FormatTime(DurationSeconds);
        public string PlayPauseGlyph => IsPlaying ? "⏸" : "▶";

        partial void OnPositionSecondsChanged(double value)
        {
            OnPropertyChanged(nameof(PositionText));
            if (!_suppressPush && _userIsSeeking)
                _client.Post("seek", new { position = value });
        }

        partial void OnDurationSecondsChanged(double value) => OnPropertyChanged(nameof(DurationText));
        partial void OnIsPlayingChanged(bool value) => OnPropertyChanged(nameof(PlayPauseGlyph));

        partial void OnLoopChanged(bool value)
        {
            if (!_suppressPush) _client.Post("set_loop", new { value });
        }

        public void BeginSeek() => _userIsSeeking = true;

        public void EndSeek()
        {
            if (_userIsSeeking) _client.Post("seek", new { position = PositionSeconds });
            _userIsSeeking = false;
        }

        // ── 分離 ────────────────────────────────────────────────────────
        public ObservableCollection<string> SeparationEngines { get; } =
            ["即時中央消除(零延遲)", "Demucs 高品質(離線處理)"];

        [ObservableProperty] private int _separationEngineIndex;
        [ObservableProperty] private bool _removeVocals;
        [ObservableProperty] private bool _removeInstrumental;
        [ObservableProperty] private double _separationStrength = 100;
        [ObservableProperty] private double _separationLowCut = 180;
        [ObservableProperty] private double _separationHighCut = 9000;

        // ── 即時分離的演算法選擇 ────────────────────────────────────────
        public ObservableCollection<string> SeparatorQualities { get; } =
            ["品質優先(頻譜域逐格)", "低延遲(時域 mid/side)"];

        /// <summary>索引 0 = quality,索引 1 = fast。</summary>
        private static readonly string[] QualityKeys = ["quality", "fast"];

        [ObservableProperty] private int _separatorQualityIndex;
        [ObservableProperty] private double _separationSharpness = 2.0;
        [ObservableProperty] private string _separatorLatencyText = "";

        public bool IsSpectralSeparator => SeparatorQualityIndex == 0;

        public string SeparatorQualityHint => IsSpectralSeparator
            ? $"逐時頻格判斷置中程度,偏側樂器保留得更好。{SeparatorLatencyText}"
            : "零延遲,但對整個中頻段做同一決策,偏側樂器會被削掉一些。";

        partial void OnSeparatorQualityIndexChanged(int value)
        {
            OnPropertyChanged(nameof(IsSpectralSeparator));
            OnPropertyChanged(nameof(SeparatorQualityHint));
            if (_suppressPush) return;
            var key = QualityKeys[Math.Clamp(value, 0, QualityKeys.Length - 1)];
            _ = SafeCallAsync("set_separator_quality", new { quality = key });
        }

        partial void OnSeparatorLatencyTextChanged(string value)
            => OnPropertyChanged(nameof(SeparatorQualityHint));

        partial void OnSeparationSharpnessChanged(double value) => PushSeparatorParams();

        // ── Demucs 推論參數(UVR 也是調這兩個)──────────────────────────
        [ObservableProperty] private double _demucsShifts;
        [ObservableProperty] private double _demucsOverlap = 0.25;

        partial void OnDemucsShiftsChanged(double value) => PushDemucsParams();
        partial void OnDemucsOverlapChanged(double value) => PushDemucsParams();

        private void PushDemucsParams()
        {
            if (_suppressPush) return;
            _client.Post("set_demucs_params", new
            {
                shifts = (int)Math.Round(DemucsShifts),
                overlap = DemucsOverlap,
            });
            // 參數變了,已載入的歌要重跑才會反映
            if (DurationSeconds > 0 && !IsRealtimeSeparation) NeedsReload = true;
        }
        [ObservableProperty] private string _demucsStatus = "尚未偵測";
        [ObservableProperty] private bool _needsReload;

        public bool IsRealtimeSeparation => SeparationEngineIndex == 0;

        partial void OnSeparationEngineIndexChanged(int value)
        {
            OnPropertyChanged(nameof(IsRealtimeSeparation));
            if (_suppressPush) return;
            var mode = value == 0 ? "realtime" : "demucs";
            _ = SafeCallAsync("set_separation_engine", new { mode });
            NeedsReload = DurationSeconds > 0;
            if (value == 1) _ = ProbeDemucsAsync();
        }

        partial void OnRemoveVocalsChanged(bool value) => PushSeparationFlags();
        partial void OnRemoveInstrumentalChanged(bool value) => PushSeparationFlags();

        private void PushSeparationFlags()
        {
            if (_suppressPush) return;
            _client.Post("set_separation", new
            {
                remove_vocals = RemoveVocals,
                remove_instrumental = RemoveInstrumental,
            });
        }

        partial void OnSeparationStrengthChanged(double value) => PushSeparatorParams();
        partial void OnSeparationLowCutChanged(double value) => PushSeparatorParams();
        partial void OnSeparationHighCutChanged(double value) => PushSeparatorParams();

        private void PushSeparatorParams()
        {
            if (_suppressPush) return;
            _client.Post("set_separator_params", new
            {
                strength = SeparationStrength / 100.0,
                low_cut = SeparationLowCut,
                high_cut = SeparationHighCut,
                sharpness = SeparationSharpness,
            });
        }

        // ── 裝置 ────────────────────────────────────────────────────────
        public ObservableCollection<AudioDevice> OutputDevices { get; } = [];
        public ObservableCollection<AudioDevice> InputDevices { get; } = [];

        [ObservableProperty] private AudioDevice? _headphoneDevice;
        [ObservableProperty] private AudioDevice? _virtualDevice;
        [ObservableProperty] private AudioDevice? _micDevice;
        [ObservableProperty] private bool _hasVirtualOutput = true;
        [ObservableProperty] private bool _hasVirtualInput = true;
        [ObservableProperty] private string _virtualFamilies = "";

        // ── 首次啟動精靈 ────────────────────────────────────────────────
        private readonly DriverInstaller _driverInstaller = new();
        private readonly UiSettings _uiSettings = UiSettings.Load();

        [ObservableProperty] private bool _showSetupWizard;
        [ObservableProperty] private bool _isInstallingDriver;
        [ObservableProperty] private string _setupMessage = "";
        [ObservableProperty] private bool _setupFailed;
        [ObservableProperty] private bool _offerTestSigning;

        /// <summary>driver 資料夾裡是否已備妥驅動檔,決定要不要顯示「安裝」按鈕。</summary>
        [ObservableProperty] private bool _canInstallBundledDriver;

        public string VirtualStatusText => HasVirtualInput && HasVirtualOutput
            ? $"已偵測到虛擬音訊裝置:{VirtualFamilies}"
            : "沒有偵測到虛擬麥克風";

        // ── 低延遲(耳返品質)────────────────────────────────────────────
        /// <summary>目前分頁。0=YouTube 1=卡拉OK 2=EQ 3=設定</summary>
        [ObservableProperty] private int _selectedTabIndex;

        // ── 介面語言 ────────────────────────────────────────────────────
        public ObservableCollection<string> Languages { get; } =
            ["繁體中文", "English"];

        private static readonly string[] LanguageCodes = ["zh-Hant", "en"];

        [ObservableProperty] private int _languageIndex;

        public string LanguageHint => LanguageIndex == 0
            ? "介面語言。切換後立即生效。"
            : "Interface language. Takes effect immediately.";

        partial void OnLanguageIndexChanged(int value)
        {
            OnPropertyChanged(nameof(LanguageHint));
            if (_suppressPush) return;
            var code = LanguageCodes[Math.Clamp(value, 0, LanguageCodes.Length - 1)];
            _uiSettings.Language = code;
            _uiSettings.Save();
            Localization.Current = code;
            StatusText = value == 0
                ? "語言已切換為繁體中文"
                : "Language switched to English";
        }

        // 960 拿掉了 —— 實機量測中它沒有一次勝出過,只是把延遲往上推。
        // 32 是實測過的下限(獨佔模式 8.8 ms、0 xrun),但預算只剩 0.67 ms,
        // 慢一點的機器會爆音,所以標成「極限」而不是預設。
        public ObservableCollection<string> BlockSizes { get; } =
        [
            "32 取樣(0.7 ms · 極限)", "64 取樣(1.3 ms)", "128 取樣(2.7 ms)",
            "240 取樣(5 ms)", "480 取樣(10 ms)",
        ];

        private static readonly int[] BlockSizeValues = [32, 64, 128, 240, 480];

        private const int DefaultBlockSizeIndex = 2;   // 128

        [ObservableProperty] private bool _isTuning;
        [ObservableProperty] private string _tuneStatus = "";
        [ObservableProperty] private double _tuneProgress;

        /// <summary>
        /// 自動調校延遲。會在實機上逐一測試設定組合,挑出不掉幀的最低延遲。
        ///
        /// 掃描期間會反覆開關音訊串流,所以聲音會斷斷續續 —— 這是預期行為。
        /// </summary>
        [RelayCommand]
        private async Task TuneLatencyAsync()
        {
            if ((MicDevice?.Index ?? -1) < 0 || (HeadphoneDevice?.Index ?? -1) < 0)
            {
                TuneStatus = "需要同時選好麥克風與耳機才能調校";
                return;
            }

            IsTuning = true;
            TuneProgress = 0;
            TuneStatus = "準備中…";
            // 調校要不要試獨佔,直接看上面那個開關 —— 獨佔會鎖住裝置,
            // 那是同一個取捨,沒有必要問兩次。
            var ok = await SafeCallAsync("tune_latency", new
            {
                target_ms = 30.0,
                allow_exclusive = ExclusiveMode,
            });
            if (!ok)
            {
                IsTuning = false;
                TuneStatus = "無法開始調校";
            }
        }

        [ObservableProperty] private bool _exclusiveMode;
        [ObservableProperty] private int _blockSizeIndex = DefaultBlockSizeIndex;
        [ObservableProperty] private string _latencyText = "";
        [ObservableProperty] private string _exclusiveNote = "";

        partial void OnExclusiveModeChanged(bool value) => PushAudioOptions();
        partial void OnBlockSizeIndexChanged(int value) => PushAudioOptions();

        private void PushAudioOptions()
        {
            if (_suppressPush || !IsConnected) return;
            var index = Math.Clamp(BlockSizeIndex, 0, BlockSizeValues.Length - 1);

            // 低延遲設定是「調一次就該一直有效」的東西,存起來,
            // 否則每次重開都退回預設值,使用者得重調一遍。
            _uiSettings.ExclusiveMode = ExclusiveMode;
            _uiSettings.BlockSize = BlockSizeValues[index];
            _uiSettings.Save();

            _ = SafeCallAsync("set_audio_options", new
            {
                exclusive_mode = ExclusiveMode,
                blocksize = BlockSizeValues[index],
            });
        }

        partial void OnHeadphoneDeviceChanged(AudioDevice? value) => PushDevices();
        partial void OnVirtualDeviceChanged(AudioDevice? value) => PushDevices();
        partial void OnMicDeviceChanged(AudioDevice? value) => PushDevices();

        private void PushDevices()
        {
            if (_suppressPush || !IsConnected) return;
            _client.Post("set_devices", new
            {
                headphone = HeadphoneDevice?.Index ?? -1,
                @virtual = VirtualDevice?.Index ?? -1,
                mic = MicDevice?.Index ?? -1,
            });
        }

        // ── 混音推桿(全部以 dB 為單位)─────────────────────────────────
        [ObservableProperty] private double _musicFaderDb;
        [ObservableProperty] private double _micFaderDb;
        [ObservableProperty] private double _musicToHeadphoneDb;
        [ObservableProperty] private double _musicToVirtualDb = -2;
        [ObservableProperty] private double _micToVirtualDb;
        [ObservableProperty] private double _micMonitorDb = -6;
        [ObservableProperty] private double _masterHeadphoneDb = -2;
        [ObservableProperty] private double _masterVirtualDb = -2;

        [ObservableProperty] private bool _monitorSelf;
        [ObservableProperty] private bool _micMuted;
        [ObservableProperty] private bool _musicMuted;

        partial void OnMusicFaderDbChanged(double value) => PushGain("music_fader", value);
        partial void OnMicFaderDbChanged(double value) => PushGain("mic_fader", value);
        partial void OnMusicToHeadphoneDbChanged(double value) => PushGain("send_music_hp", value);
        partial void OnMusicToVirtualDbChanged(double value) => PushGain("send_music_vc", value);
        partial void OnMicToVirtualDbChanged(double value) => PushGain("send_mic_vc", value);
        partial void OnMicMonitorDbChanged(double value) => PushGain("send_mic_monitor", value);

        /// <summary>
        /// 送給 Discord 那一路的「音樂 vs 歌聲」對時(毫秒,可正可負)。
        /// 只影響「→ 虛擬麥」,自己的耳返不受影響。
        ///
        /// 正值延後音樂(歌聲相對變早)、負值延後歌聲(音樂相對變早)。
        /// 訊號是即時的,沒辦法真的把哪一路提前,所以「提前」一律靠延後
        /// 另一路達成 —— 對使用者來說就是一個可以往兩邊調的數字。
        /// </summary>
        [ObservableProperty] private double _dcDelayMs;

        /// <summary>
        /// 對時沒有功能上的上下限 —— 引擎的延遲線會跟著設定值長大。這個數字
        /// 是記憶體保險絲(60 秒),擋的是少打一個小數點那種手誤,不是擋使用者。
        /// </summary>
        public const double MaxDcDelayMs = 60000;

        /// <summary>推桿預設的半幅。實際會用到的範圍,行程才夠細。</summary>
        private const double DefaultDcSliderSpan = 300;

        private double _dcSliderSpan = DefaultDcSliderSpan;

        /// <summary>
        /// 推桿的兩端。必須跟著打進來的值長大 —— Avalonia 的 Slider 會把
        /// Value 夾在 Minimum/Maximum 之內再寫回綁定,所以推桿若固定在
        /// ±300,使用者在輸入框打的 5000 會被推桿默默改回 300。
        ///
        /// 只長不縮:拖動途中改變行程長度會讓滑塊當場跳位。要縮回細行程
        /// 按「歸零」。
        /// </summary>
        public double DcDelaySliderMax => _dcSliderSpan;
        public double DcDelaySliderMin => -_dcSliderSpan;

        partial void OnDcDelayMsChanged(double value)
        {
            var needed = Math.Max(DefaultDcSliderSpan,
                                  Math.Ceiling(Math.Abs(value) / 100) * 100);
            if (needed > _dcSliderSpan)
            {
                _dcSliderSpan = needed;
                OnPropertyChanged(nameof(DcDelaySliderMax));
                OnPropertyChanged(nameof(DcDelaySliderMin));
            }

            OnPropertyChanged(nameof(DcDelayText));
            OnPropertyChanged(nameof(DcDelayDirectionText));
            if (_suppressPush) return;
            _sender?.SetVcSyncMs(value);
        }

        /// <summary>
        /// 直接打數字的入口 —— 推桿一格好幾毫秒,對不準拍子,而這個值本來
        /// 就常常是抄來的(建議值、上次記下的數字)。負號要吃得下去。
        /// 打不成數字就維持原值,不要把輸入中的半截字當成 0。
        /// </summary>
        public string DcDelayText
        {
            get => DcDelayMs.ToString("0.#", CultureInfo.CurrentCulture);
            set => DcDelayMs = NumericText.ParseOr(value, DcDelayMs,
                                                   -MaxDcDelayMs, MaxDcDelayMs);
        }

        /// <summary>
        /// 把號誌的意思講白話,而且兩邊都要講 —— 只說「音樂延後」的話,
        /// 使用者還要自己推另一邊變成怎樣,而那正是他要判斷的事情。
        /// </summary>
        public string DcDelayDirectionText => DcDelayMs switch
        {
            > 0 => $"音樂放慢 {DcDelayMs:0.#} ms   ·   歌聲相對加快",
            < 0 => $"歌聲放慢 {-DcDelayMs:0.#} ms   ·   音樂相對加快",
            _ => "兩邊同步,沒有偏移",
        };

        [ObservableProperty] private double _dcDelaySuggestedMs;
        [ObservableProperty] private string _dcDelayHint = "";

        /// <summary>從建議起點開始調(= 目前的耳返延遲)。</summary>
        [RelayCommand]
        private void ApplySuggestedDcDelay()
        {
            if (DcDelaySuggestedMs > 0) DcDelayMs = Math.Round(DcDelaySuggestedMs, 1);
        }

        // ── 音遊式校準 ──────────────────────────────────────────────
        [ObservableProperty] private bool _isCalibrating;
        [ObservableProperty] private double _calibrationProgress;
        [ObservableProperty] private string _calibrationStatus = "";
        [ObservableProperty] private string _calibrationResult = "";
        [ObservableProperty] private double _calibrationOffsetMs;
        [ObservableProperty] private bool _hasCalibrationResult;

        /// <summary>
        /// 放四拍「1 2 3 4」讓你跟著念,量你的聲音實際落在哪裡。
        ///
        /// 比用耳朵調準的地方在於它量的是實際數字:你聽到拍子、開口、聲音被
        /// 收進混音,整個迴圈花了多久 —— 那正是要補的對時量。
        /// </summary>
        [RelayCommand]
        private async Task StartCalibrationAsync()
        {
            if (!IsAudioRunning)
            {
                CalibrationStatus = "請先按「啟動音訊」";
                return;
            }

            IsCalibrating = true;
            HasCalibrationResult = false;
            CalibrationResult = "";
            CalibrationProgress = 0;
            CalibrationStatus = "準備…";

            if (!await SafeCallAsync("start_calibration"))
            {
                IsCalibrating = false;
                CalibrationStatus = "無法開始校準 —— 麥克風與耳機都要選好";
            }
        }

        [RelayCommand]
        private void CancelCalibration()
        {
            _client.Post("cancel_calibration", new { });
            IsCalibrating = false;
            CalibrationStatus = "已取消";
        }

        private void OnCalibrationDone(JsonElement data)
        {
            IsCalibrating = false;
            CalibrationProgress = 100;

            if (!data.TryGetProperty("ok", out var ok) || !ok.GetBoolean())
            {
                HasCalibrationResult = false;
                CalibrationStatus = data.TryGetProperty("reason", out var why)
                    ? why.GetString() ?? "校準失敗" : "校準失敗";
                return;
            }

            var offset = data.TryGetProperty("offset_ms", out var o) ? o.GetDouble() : 0;
            var spread = data.TryGetProperty("spread_ms", out var sp) ? sp.GetDouble() : 0;
            var beats = data.TryGetProperty("beats_detected", out var bd) ? bd.GetInt32() : 0;
            var used = data.TryGetProperty("beats_used", out var bu) ? bu.GetInt32() : 0;
            var reliable = data.TryGetProperty("reliable", out var rl) && rl.GetBoolean();
            var warning = data.TryGetProperty("warning", out var wn) ? wn.GetString() ?? "" : "";

            CalibrationOffsetMs = Math.Round(offset, 1);
            HasCalibrationResult = true;

            var direction = offset > 0 ? "你的聲音比拍子晚" : "你的聲音比拍子早";
            CalibrationStatus = $"量到 {offset:+0.#;-0.#;0} ms —— {direction}";
            CalibrationResult =
                $"抓到 {beats}/{used} 拍,每拍差距 {spread:0.#} ms"
                + (reliable ? "" : "  ·  這次量得不穩,建議再測一次")
                + (string.IsNullOrEmpty(warning) ? "" : "  ·  " + warning);
        }

        /// <summary>把量到的偏移套進對時。</summary>
        [RelayCommand]
        private void ApplyCalibration()
        {
            if (!HasCalibrationResult) return;
            DcDelayMs = CalibrationOffsetMs;
            CalibrationStatus = $"已套用 {CalibrationOffsetMs:+0.#;-0.#;0} ms";
        }

        /// <summary>
        /// 對時歸零,回到「完全不動」的基準點。順便把推桿行程收回預設 ——
        /// 之前打過大數字的話行程會被撐長,細部就變得難調。這裡是唯一會
        /// 收縮的時機:值已經是 0,不可能正在拖動。
        /// </summary>
        [RelayCommand]
        private void ResetDcDelay()
        {
            DcDelayMs = 0;
            if (_dcSliderSpan == DefaultDcSliderSpan) return;
            _dcSliderSpan = DefaultDcSliderSpan;
            OnPropertyChanged(nameof(DcDelaySliderMax));
            OnPropertyChanged(nameof(DcDelaySliderMin));
        }

        /// <summary>
        /// 耳機改聽「送給 Discord 的那一份」。
        ///
        /// 平常耳機聽的是還沒對時的本地混音,所以對時怎麼調都感覺不到,
        /// 只能靠對方回報。打開這個之後同一份訊號分兩端 —— 一端進耳機、
        /// 一端進虛擬麥 —— 就能自己聽著調。
        /// </summary>
        [ObservableProperty] private bool _monitorSend;

        partial void OnMonitorSendChanged(bool value) => PushFlag("monitor_send", value);

        // ── 升 key / 降 key ─────────────────────────────────────────────
        //
        // 只動音樂,不動你的聲音 —— 目的正是讓歌配合你的音域,而不是反過來。
        // 引擎在 0 半音時整條旁通,所以「原調」是真的沒有任何處理。
        public const int MaxPitchSemitones = 12;

        [ObservableProperty] private double _musicPitchSemitones;
        [ObservableProperty] private string _musicPitchLatencyHint = "";

        /// <summary>「原調」/「+2」/「−3」。用真正的減號,對齊時比較好看。</summary>
        public string MusicPitchText => MusicPitchSemitones == 0
            ? "原調"
            : $"{(MusicPitchSemitones > 0 ? "+" : "−")}{Math.Abs(MusicPitchSemitones):0}";

        /// <summary>把半音換算成音樂人習慣的說法。</summary>
        public string MusicPitchDetail
        {
            get
            {
                var steps = (int)Math.Round(MusicPitchSemitones);
                if (steps == 0) return "沒有做任何變調處理";
                var direction = steps > 0 ? "升" : "降";
                var count = Math.Abs(steps);
                return count % 12 == 0
                    ? $"{direction} {count} 個半音(整整一個八度)"
                    : $"{direction} {count} 個半音";
            }
        }

        partial void OnMusicPitchSemitonesChanged(double value)
        {
            OnPropertyChanged(nameof(MusicPitchText));
            OnPropertyChanged(nameof(MusicPitchDetail));
            if (_suppressPush) return;
            _client.Post("set_music_pitch", new { semitones = Math.Round(value) });
        }

        [RelayCommand]
        private void PitchUp() => ShiftPitch(1);

        [RelayCommand]
        private void PitchDown() => ShiftPitch(-1);

        [RelayCommand]
        private void ResetPitch() => MusicPitchSemitones = 0;

        private void ShiftPitch(int steps) => MusicPitchSemitones = Math.Clamp(
            Math.Round(MusicPitchSemitones) + steps,
            -MaxPitchSemitones, MaxPitchSemitones);

        // ── 麥克風回音 ──────────────────────────────────────────────────
        //
        // 卡拉 OK 機那顆 ECHO 旋鈕。預設關閉 —— 開著唱歌好聽,講話會變得
        // 很難聽懂。四個參數都以百分比呈現,和其他效果的推桿一致。
        [ObservableProperty] private bool _micEchoEnabled;
        [ObservableProperty] private double _micEchoDelayMs = 180;
        [ObservableProperty] private double _micEchoFeedback = 35;
        [ObservableProperty] private double _micEchoMix = 25;
        [ObservableProperty] private double _micEchoDamping = 35;

        partial void OnMicEchoEnabledChanged(bool value) => PushMicEcho();
        partial void OnMicEchoDelayMsChanged(double value) => PushMicEcho();
        partial void OnMicEchoFeedbackChanged(double value) => PushMicEcho();
        partial void OnMicEchoMixChanged(double value) => PushMicEcho();
        partial void OnMicEchoDampingChanged(double value) => PushMicEcho();

        // ── 麥克風降噪(電流聲)────────────────────────────────────
        //
        // 「電流聲」通常是兩種東西:50/60 Hz 的電源哼聲(窄頻,陷波切得掉)
        // 與前級的寬頻嘶聲(跟人聲重疊,只能在沒出聲時壓下去)。兩個開關
        // 分開,因為使用者的環境往往只有其中一種。
        [ObservableProperty] private bool _micDenoiseEnabled;
        [ObservableProperty] private bool _micGateEnabled = true;
        [ObservableProperty] private int _micHumIndex;          // 0=關 1=50Hz 2=60Hz
        [ObservableProperty] private double _micGateMarginDb = 12;
        [ObservableProperty] private double _micGateReductionDb = 18;
        [ObservableProperty] private string _micNoiseFloorText = "";

        public string[] HumOptions { get; } =
            ["不切哼聲", "50 Hz(台灣以外多數地區)", "60 Hz(台灣 / 日本東部 / 美洲)"];

        private static readonly double[] HumFrequencies = [0, 50, 60];

        partial void OnMicDenoiseEnabledChanged(bool value) => PushMicDenoise();
        partial void OnMicGateEnabledChanged(bool value) => PushMicDenoise();
        partial void OnMicHumIndexChanged(int value) => PushMicDenoise();
        partial void OnMicGateMarginDbChanged(double value) => PushMicDenoise();
        partial void OnMicGateReductionDbChanged(double value) => PushMicDenoise();

        private void PushMicDenoise(bool learn = false)
        {
            if (_suppressPush) return;
            var hum = HumFrequencies[Math.Clamp(MicHumIndex, 0, HumFrequencies.Length - 1)];
            _client.Post("set_mic_denoise", new
            {
                enabled = MicDenoiseEnabled,
                gate_enabled = MicGateEnabled,
                hum_hz = hum,
                margin_db = MicGateMarginDb,
                reduction_db = MicGateReductionDb,
                learn,
                learn_seconds = 1.5,
            });
        }

        /// <summary>
        /// 重新學底噪:接下來一秒半收到的音量當作底噪,門檻跟著重設。
        /// 換麥克風或換環境之後按一次就好 —— 平常它自己會慢慢跟上。
        /// </summary>
        [RelayCommand]
        private void LearnNoiseFloor()
        {
            PushMicDenoise(learn: true);
            MicNoiseFloorText = "正在聽底噪…請保持安靜一秒半";
        }

        private void PushMicEcho()
        {
            if (_suppressPush) return;
            _client.Post("set_mic_echo", new
            {
                enabled = MicEchoEnabled,
                delay_ms = MicEchoDelayMs,
                feedback = MicEchoFeedback / 100.0,
                mix = MicEchoMix / 100.0,
                damping = MicEchoDamping / 100.0,
            });
        }
        partial void OnMasterHeadphoneDbChanged(double value) => PushGain("master_hp", value);
        partial void OnMasterVirtualDbChanged(double value) => PushGain("master_vc", value);

        partial void OnMonitorSelfChanged(bool value) => PushFlag("monitor_self", value);
        partial void OnMicMutedChanged(bool value) => PushFlag("mic_muted", value);
        partial void OnMusicMutedChanged(bool value) => PushFlag("music_muted", value);

        private void PushGain(string name, double db)
        {
            if (_suppressPush) return;
            _sender?.SetGainDb(name, db);
        }

        private void PushFlag(string name, bool value)
        {
            if (_suppressPush) return;
            _client.Post("set_flag", new { name, value });
        }

        // ── EQ 對引擎的出口(IEqBridge)─────────────────────────────────
        void IEqBridge.PushGains(string target, double[] gains)
            => _sender?.SetEqGains(target, gains);

        void IEqBridge.PushEnabled(string target, bool enabled)
            => _client.Post("eq_enable", new { target, enabled });

        void IEqBridge.PushBandShape(string target, int index, double frequency, double q)
            => _client.Post("eq_set_band", new { target, index, freq = frequency, q });

        async Task<EqBandSpec[]?> IEqBridge.AddBandAsync(string target, double frequency)
            => await CallEqStructureAsync("eq_add_band", new { target, freq = frequency });

        async Task<EqBandSpec[]?> IEqBridge.RemoveBandAsync(string target, int index)
            => await CallEqStructureAsync("eq_remove_band", new { target, index });

        /// <summary>
        /// 增刪頻段。先把待送的推桿值倒乾淨 —— eq_set_all 是照索引對應的,
        /// 讓它排在增刪之後才送達的話,整排增益會位移一格。
        /// </summary>
        private async Task<EqBandSpec[]?> CallEqStructureAsync(string command, object args)
        {
            _sender?.Flush();
            try
            {
                return ReadBands(await _client.CallAsync(command, args));
            }
            catch (Exception ex)
            {
                AppendLog($"{command}: {ex.Message}");
                StatusText = ex.Message;
                return null;
            }
        }

        /// <summary>把引擎回傳的 bands 陣列讀成頻段設定。</summary>
        private static EqBandSpec[]? ReadBands(JsonElement element)
        {
            if (element.ValueKind != JsonValueKind.Object
                || !element.TryGetProperty("bands", out var bands)
                || bands.ValueKind != JsonValueKind.Array)
            {
                return null;
            }

            return [.. bands.EnumerateArray().Select(band => new EqBandSpec(
                band.TryGetProperty("freq", out var f) ? f.GetDouble() : 1000,
                band.TryGetProperty("gain", out var g) ? g.GetDouble() : 0,
                band.TryGetProperty("q", out var q) ? q.GetDouble() : 1.41))];
        }

        /// <summary>
        /// 向引擎拿目前真正的頻段配置。頻段可增可刪,前端寫死的預設值只是
        /// 連線前的擺設,連上之後一定要以引擎為準。
        /// </summary>
        private async Task LoadEqLayoutAsync()
        {
            try
            {
                var info = await _client.CallAsync("eq_info");
                if (info.TryGetProperty("music", out var music)
                    && ReadBands(music) is { } musicBands)
                {
                    MusicEq.LoadBands(musicBands);
                }
                if (info.TryGetProperty("mic", out var mic)
                    && ReadBands(mic) is { } micBands)
                {
                    MicEq.LoadBands(micBands);
                }
            }
            catch (Exception ex)
            {
                AppendLog($"eq_info: {ex.Message}");
            }
        }

        // ── 啟動 ────────────────────────────────────────────────────────
        public async Task InitializeAsync()
        {
            try
            {
                StatusText = "正在啟動音訊引擎…";
                await _process.StartAsync();
                await _client.ConnectAsync(_process.Port, _process.Token);

                _client.EventReceived += OnEngineEvent;
                _client.Disconnected += OnDisconnected;
                _client.DispatchError += ex => Dispatcher.UIThread.Post(
                    () => AppendLog($"訊息處理錯誤(連線仍正常):{ex.Message}"));
                _sender = new ParameterSender(_client);

                IsConnected = true;
                EngineVersion = _process.EngineVersion;

                await RefreshDevicesAsync();
                await LoadStatusAsync();
                await LoadEqLayoutAsync();
                RestoreLowLatencySettings();
                PushAllGains();

                StatusText = "引擎已就緒 —— 選好裝置後按「啟動音訊」";
                AppendLog($"引擎 {EngineVersion} 已連線(埠 {_process.Port},Python: {_process.PythonPath})");

                if (!HasFfmpeg)
                    AppendLog("警告:找不到 ffmpeg,無法解碼音源。請執行 pip install imageio-ffmpeg");

                // 沒有虛擬麥克風時彈出首次啟動精靈(除非使用者說過略過)
                EvaluateSetupState();
            }
            catch (Exception ex)
            {
                IsConnected = false;
                StatusText = "引擎啟動失敗";
                AppendLog(ex.Message);
            }
        }

        /// <summary>
        /// 把上次調好的低延遲設定放回去。ApplyStatus() 剛把引擎的預設值
        /// 灌進 UI,所以這一步要排在它後面,否則會被蓋掉。
        /// </summary>
        private void RestoreLowLatencySettings()
        {
            var saved = Array.IndexOf(BlockSizeValues, _uiSettings.BlockSize);

            _suppressPush = true;
            try
            {
                ExclusiveMode = _uiSettings.ExclusiveMode;
                BlockSizeIndex = saved >= 0 ? saved : DefaultBlockSizeIndex;
            }
            finally
            {
                _suppressPush = false;
            }

            PushAudioOptions();
        }

        /// <summary>把 UI 上的預設值推給引擎,讓兩邊一開始就一致。</summary>
        private void PushAllGains()
        {
            _client.Post("set_gains", new
            {
                db = new Dictionary<string, double>
                {
                    ["music_fader"] = MusicFaderDb,
                    ["mic_fader"] = MicFaderDb,
                    ["send_music_hp"] = MusicToHeadphoneDb,
                    ["send_music_vc"] = MusicToVirtualDb,
                    ["send_mic_vc"] = MicToVirtualDb,
                    ["send_mic_monitor"] = MicMonitorDb,
                    ["master_hp"] = MasterHeadphoneDb,
                    ["master_vc"] = MasterVirtualDb,
                },
            });
        }

        private async Task LoadStatusAsync()
        {
            var status = await _client.CallAsync("get_status");
            ApplyStatus(status);
        }

        private void ApplyStatus(JsonElement status)
        {
            if (status.ValueKind != JsonValueKind.Object) return;
            _suppressPush = true;
            try
            {
                if (status.TryGetProperty("running", out var running))
                    IsAudioRunning = running.GetBoolean();
                if (status.TryGetProperty("ffmpeg", out var ffmpeg))
                    HasFfmpeg = ffmpeg.GetBoolean();

                if (status.TryGetProperty("music_eq", out var musicEq))
                    MusicEq.LoadFrom(ReadGains(musicEq), ReadEnabled(musicEq));
                if (status.TryGetProperty("mic_eq", out var micEq))
                    MicEq.LoadFrom(ReadGains(micEq), ReadEnabled(micEq));

                if (status.TryGetProperty("separator", out var sep))
                {
                    if (sep.TryGetProperty("strength", out var s)) SeparationStrength = s.GetDouble() * 100;
                    if (sep.TryGetProperty("low_cut", out var lo)) SeparationLowCut = lo.GetDouble();
                    if (sep.TryGetProperty("high_cut", out var hi)) SeparationHighCut = hi.GetDouble();
                }

                if (status.TryGetProperty("remove_vocals", out var rv)) RemoveVocals = rv.GetBoolean();
                if (status.TryGetProperty("remove_instrumental", out var ri)) RemoveInstrumental = ri.GetBoolean();

                if (status.TryGetProperty("player", out var player)) ApplyPlayerState(player);

                if (status.TryGetProperty("xruns", out var xruns)
                    && status.TryGetProperty("cpu", out var cpu))
                {
                    DiagnosticsText = $"CPU {cpu.GetDouble() * 100:0.#}%   ·   xrun {xruns.GetInt32()}";
                }

                if (status.TryGetProperty("vc_sync_ms", out var delay)
                    && delay.ValueKind == JsonValueKind.Number)
                {
                    DcDelayMs = delay.GetDouble();
                }

                if (status.TryGetProperty("music_pitch", out var pitch)
                    && pitch.ValueKind == JsonValueKind.Object)
                {
                    if (pitch.TryGetProperty("semitones", out var semitones))
                        MusicPitchSemitones = semitones.GetDouble();

                    // 變調是唯一會讓音樂路徑多出延遲的東西,而它剛好也是使用者
                    // 自己開的 —— 講清楚代價,不要讓他以為是機器變慢了。
                    var latency = pitch.TryGetProperty("latency_ms", out var ms)
                        ? ms.GetDouble() : 0;
                    MusicPitchLatencyHint = latency > 0
                        ? $"變調讓音樂多 {latency:0.#} ms 延遲(你的歌聲與對方聽到的相對時間不受影響)"
                        : "";
                }

                if (status.TryGetProperty("mic_echo", out var echo)
                    && echo.ValueKind == JsonValueKind.Object)
                {
                    if (echo.TryGetProperty("enabled", out var on))
                        MicEchoEnabled = on.GetBoolean();
                }

                if (status.TryGetProperty("mic_denoise", out var dn)
                    && dn.ValueKind == JsonValueKind.Object)
                {
                    if (dn.TryGetProperty("enabled", out var de))
                        MicDenoiseEnabled = de.GetBoolean();
                    if (dn.TryGetProperty("gate_enabled", out var dg))
                        MicGateEnabled = dg.GetBoolean();
                    if (dn.TryGetProperty("hum_hz", out var dh))
                    {
                        var idx = Array.IndexOf(HumFrequencies, dh.GetDouble());
                        if (idx >= 0) MicHumIndex = idx;
                    }
                    if (dn.TryGetProperty("floor_db", out var df)
                        && dn.TryGetProperty("threshold_db", out var dt))
                    {
                        MicNoiseFloorText = MicDenoiseEnabled
                            ? $"底噪 {df.GetDouble():0.#} dB · 閘門門檻 {dt.GetDouble():0.#} dB"
                            : "";
                    }
                    if (echo.TryGetProperty("delay_ms", out var ms))
                        MicEchoDelayMs = ms.GetDouble();
                    if (echo.TryGetProperty("feedback", out var fb))
                        MicEchoFeedback = fb.GetDouble() * 100;
                    if (echo.TryGetProperty("mix", out var mix))
                        MicEchoMix = mix.GetDouble() * 100;
                    if (echo.TryGetProperty("damping", out var damping))
                        MicEchoDamping = damping.GetDouble() * 100;
                }

                DcDelaySuggestedMs =
                    status.TryGetProperty("vc_sync_suggested_ms", out var sug)
                    && sug.ValueKind == JsonValueKind.Number ? sug.GetDouble() : 0;
                DcDelayHint = DcDelaySuggestedMs > 0
                    ? $"建議從 {DcDelaySuggestedMs:0.#} ms 起(= 目前耳返延遲),再往兩邊微調"
                    : "建議值需要同時選好麥克風與耳機";

                if (status.TryGetProperty("exclusive_mode", out var exclusive))
                    ExclusiveMode = exclusive.GetBoolean();
                if (status.TryGetProperty("blocksize", out var blocksize))
                {
                    var index = Array.IndexOf(BlockSizeValues, blocksize.GetInt32());
                    if (index >= 0) BlockSizeIndex = index;
                }

                ApplyLatency(status);
            }
            finally
            {
                _suppressPush = false;
            }
        }

        private void ApplyLatency(JsonElement status)
        {
            if (!status.TryGetProperty("latency", out var latency)
                || latency.ValueKind != JsonValueKind.Object)
            {
                return;
            }

            if (!IsAudioRunning)
            {
                LatencyText = "";
            }
            else if (latency.TryGetProperty("monitor_ms", out var monitor)
                     && monitor.ValueKind == JsonValueKind.Number)
            {
                var ms = monitor.GetDouble();
                LatencyText = $"耳返延遲 {ms:0.#} ms —— {LatencyVerdict(ms)}";
            }
            else
            {
                LatencyText = "耳返延遲:需要同時選好麥克風與耳機";
            }

            var active = status.TryGetProperty("exclusive_active", out var a)
                         && a.ValueKind == JsonValueKind.Array
                ? a.EnumerateArray().Select(x => x.GetString()).ToArray()
                : [];
            var notes = status.TryGetProperty("exclusive_notes", out var n)
                        && n.ValueKind == JsonValueKind.Array
                ? n.EnumerateArray().Select(x => x.GetString()).ToArray()
                : [];

            var formatNotes = status.TryGetProperty("format_notes", out var f)
                              && f.ValueKind == JsonValueKind.Array
                ? f.EnumerateArray().Select(x => x.GetString()).ToArray()
                : [];

            var lines = new List<string?>();
            if (ExclusiveMode)
            {
                if (notes.Length > 0)
                    lines.AddRange(notes);
                else if (active.Length > 0)
                    lines.Add($"獨佔生效:{string.Join("、", active)}");
            }
            // 取樣率轉換跟獨佔模式無關,開不開都要講 —— 這是「聽起來怪」
            // 的時候第一個要看的線索。
            lines.AddRange(formatNotes);
            ExclusiveNote = string.Join("\n", lines);

            ApplyHostApiWarning(status);
        }

        /// <summary>
        /// 麥克風或耳機掛在高延遲的 host API 上時講清楚。
        ///
        /// 同一支耳麥在 MME 下 PortAudio 回報 90 ms、DirectSound 120 ms,
        /// WASAPI 只有 3 ms —— 這個差距比底下所有緩衝調整加起來還大,
        /// 而裝置名稱在清單裡長得幾乎一樣,不講使用者不會知道自己選錯了。
        /// </summary>
        private void ApplyHostApiWarning(JsonElement status)
        {
            if (!status.TryGetProperty("slow_roles", out var slow)
                || slow.ValueKind != JsonValueKind.Array)
            {
                HostApiWarning = "";
                return;
            }

            var names = status.TryGetProperty("hostapis", out var h)
                        && h.ValueKind == JsonValueKind.Object ? h : default;

            var parts = new List<string>();
            foreach (var role in slow.EnumerateArray())
            {
                var key = role.GetString() ?? "";
                var label = key == "mic" ? "麥克風" : "耳機";
                var api = names.ValueKind == JsonValueKind.Object
                          && names.TryGetProperty(key, out var v)
                    ? v.GetString() ?? "?" : "?";
                parts.Add($"{label}走 {api}");
            }

            HostApiWarning = parts.Count == 0
                ? ""
                : $"{string.Join("、", parts)} —— 這是高延遲路徑。"
                  + "請在裝置清單裡改選同名的 WASAPI 版本,那一項就能省下數十毫秒。";
        }

        [ObservableProperty] private string _hostApiWarning = "";

        private static string LatencyVerdict(double ms) => ms switch
        {
            < 10 => "察覺不到",
            < 20 => "唱歌沒問題",
            < 30 => "明顯,講話會有點怪",
            _ => "會干擾發聲,建議開啟獨佔模式",
        };

        private static double[] ReadGains(JsonElement element) =>
            element.TryGetProperty("gains", out var g) && g.ValueKind == JsonValueKind.Array
                ? g.EnumerateArray().Select(x => x.GetDouble()).ToArray()
                : [];

        private static bool ReadEnabled(JsonElement element) =>
            !element.TryGetProperty("enabled", out var e) || e.GetBoolean();

        private void ApplyPlayerState(JsonElement player)
        {
            var wasSuppressed = _suppressPush;
            _suppressPush = true;
            try
            {
                if (player.TryGetProperty("playing", out var playing)) IsPlaying = playing.GetBoolean();
                if (player.TryGetProperty("duration", out var duration)) DurationSeconds = duration.GetDouble();
                if (!_userIsSeeking && player.TryGetProperty("position", out var position))
                {
                    PositionSeconds = position.GetDouble();
                    Lyrics.UpdatePosition(PositionSeconds);
                }
                if (player.TryGetProperty("loop", out var loop)) Loop = loop.GetBoolean();
            }
            finally
            {
                _suppressPush = wasSuppressed;
            }
        }

        // ── 引擎事件 ────────────────────────────────────────────────────
        private void OnEngineEvent(string name, JsonElement data)
        {
            Dispatcher.UIThread.Post(() =>
            {
                switch (name)
                {
                    case "meters":
                        UpdateMeters(data);
                        break;

                    case "player":
                        ApplyPlayerState(data);
                        break;

                    case "status":
                        ApplyStatus(data);
                        break;

                    case "progress":
                        if (data.TryGetProperty("stage", out var stage))
                            ProgressStage = stage.GetString() ?? "";
                        if (data.TryGetProperty("value", out var value))
                            Progress = value.GetDouble() * 100;
                        if (data.TryGetProperty("busy", out var busy))
                            IsBusy = busy.GetBoolean();
                        break;

                    case "loaded":
                        OnMediaLoaded(data);
                        break;

                    case "load_failed":
                        IsBusy = false;
                        var error = data.TryGetProperty("error", out var e) ? e.GetString() : "未知的錯誤";
                        StatusText = "載入失敗";
                        AppendLog($"載入失敗: {error}");
                        break;

                    case "load_cancelled":
                        IsBusy = false;
                        StatusText = "已取消載入";
                        break;

                    case "search_results":
                        OnSearchResults(data);
                        break;

                    case "lyrics":
                        Lyrics.Load(data);
                        Lyrics.UpdatePosition(PositionSeconds);
                        break;

                    case "lyrics_status":
                        OnLyricsStatus(data);
                        break;

                    case "calibration_progress":
                        IsCalibrating = true;
                        if (data.TryGetProperty("progress", out var cp))
                            CalibrationProgress = cp.GetDouble() * 100;
                        if (data.TryGetProperty("beat", out var cb))
                        {
                            var beat = cb.GetInt32();
                            CalibrationStatus = beat == 0
                                ? "準備…跟著念出來"
                                : $"跟著念:{((beat - 1) % 4) + 1}";
                        }
                        break;

                    case "calibration_done":
                        OnCalibrationDone(data);
                        break;

                    case "calibration_failed":
                        IsCalibrating = false;
                        CalibrationStatus = "校準失敗:"
                            + (data.TryGetProperty("error", out var ce) ? ce.GetString() : "");
                        break;

                    case "tune_progress":
                        IsTuning = true;
                        if (data.TryGetProperty("value", out var tp))
                            TuneProgress = tp.GetDouble() * 100;
                        if (data.TryGetProperty("stage", out var ts))
                            TuneStatus = ts.GetString() ?? "";
                        break;

                    case "tune_done":
                        OnTuneDone(data);
                        break;

                    case "tune_failed":
                        IsTuning = false;
                        TuneStatus = "調校失敗:"
                            + (data.TryGetProperty("error", out var te) ? te.GetString() : "");
                        break;

                    case "search_failed":
                        IsSearching = false;
                        var reason = data.TryGetProperty("error", out var se) ? se.GetString() : "";
                        SearchHint = $"搜尋失敗:{reason}";
                        StatusText = "搜尋失敗";
                        break;

                    case "track_finished":
                        // 單曲循環由引擎自己處理(Loop 旗標),這裡只管歌單推進。
                        // 兩者都開時以引擎的單曲循環優先,不會跳走。
                        if (AutoAdvance && !Loop && Playlist.Count > 1
                            && CurrentTrack is not null)
                        {
                            StatusText = "播放結束 —— 換下一首";
                            Advance(1);
                        }
                        else
                        {
                            StatusText = "播放結束";
                        }
                        break;

                    case "log":
                        if (data.TryGetProperty("message", out var message))
                            AppendLog(message.GetString() ?? "");
                        break;
                }
            });
        }

        private void UpdateMeters(JsonElement data)
        {
            if (data.ValueKind != JsonValueKind.Object) return;
            if (data.TryGetProperty("music_in", out var a)) MusicInMeter.Update(a);
            if (data.TryGetProperty("music_out", out var b)) MusicOutMeter.Update(b);
            if (data.TryGetProperty("mic_in", out var c)) MicInMeter.Update(c);
            if (data.TryGetProperty("mic_out", out var d)) MicOutMeter.Update(d);
            if (data.TryGetProperty("hp_out", out var e)) HeadphoneMeter.Update(e);
            if (data.TryGetProperty("vc_out", out var f)) VirtualMeter.Update(f);
        }

        private void OnLyricsStatus(JsonElement data)
        {
            var state = data.TryGetProperty("state", out var s) ? s.GetString() : "";
            var reason = data.TryGetProperty("reason", out var r) ? r.GetString() : "";

            switch (state)
            {
                case "searching":
                    Lyrics.Clear();
                    Lyrics.IsWorking = true;
                    Lyrics.Status = "尋找歌詞中…";
                    break;

                case "none":
                    Lyrics.IsWorking = false;
                    Lyrics.Status = string.IsNullOrEmpty(reason) ? "找不到歌詞" : reason;
                    // 這次沒抓到,但清單要更新 —— 不然使用者連「有哪些語言」
                    // 都看不到,也就無從換起。
                    Lyrics.SetLanguages(data, "");
                    break;

                case "failed":
                    Lyrics.IsWorking = false;
                    Lyrics.Status = $"歌詞取得失敗:{reason}";
                    break;
            }
        }

        private void OnTuneDone(JsonElement data)
        {
            IsTuning = false;
            TuneProgress = 100;

            if (!data.TryGetProperty("best", out var best)
                || best.ValueKind != JsonValueKind.Object)
            {
                TuneStatus = "找不到可用的設定 —— 所有組合都掉幀或無法開啟";
                return;
            }

            var latency = best.TryGetProperty("monitor_ms", out var m) ? m.GetDouble() : 0;
            var exclusive = best.TryGetProperty("exclusive", out var x) && x.GetBoolean();
            var block = best.TryGetProperty("blocksize", out var b) ? b.GetInt32() : 0;
            var met = data.TryGetProperty("met_target", out var mt) && mt.GetBoolean();

            var mode = exclusive ? "獨佔模式" : "共享模式";
            TuneStatus = met
                ? $"已套用:{mode} · {block} 取樣 → 耳返 {latency:0.#} ms"
                : $"最佳僅能到 {latency:0.#} ms({mode} · {block} 取樣)。"
                  + (exclusive ? "" : "開啟獨佔模式可再大幅降低。");

            AppendLog(TuneStatus);
            // 引擎已經套用設定,把 UI 的顯示同步過來
            _ = LoadStatusAsync();
        }

        private void OnSearchResults(JsonElement data)
        {
            IsSearching = false;
            var query = data.TryGetProperty("query", out var q) ? q.GetString() : "";

            _suppressPush = true;
            try
            {
                SearchResults.Clear();
                SelectedResult = null;
                if (data.TryGetProperty("results", out var results)
                    && results.ValueKind == JsonValueKind.Array)
                {
                    foreach (var item in results.EnumerateArray())
                    {
                        if (item.Deserialize<SearchResult>() is { } result)
                            SearchResults.Add(result);
                    }
                }
            }
            finally
            {
                _suppressPush = false;
            }

            OnPropertyChanged(nameof(HasSearchResults));
            SearchHint = SearchResults.Count > 0
                ? $"「{query}」找到 {SearchResults.Count} 首 —— 點一下就載入"
                : $"「{query}」沒有找到結果";
            StatusText = SearchResults.Count > 0 ? "請選擇一首" : "沒有搜尋結果";
        }

        private void OnMediaLoaded(JsonElement data)
        {
            IsBusy = false;
            NeedsReload = false;
            if (data.TryGetProperty("media", out var media))
            {
                TrackTitle = media.TryGetProperty("title", out var title)
                    ? title.GetString() ?? "(無標題)" : "(無標題)";
                if (media.TryGetProperty("duration", out var duration))
                    DurationSeconds = duration.GetDouble();
            }
            if (data.TryGetProperty("player", out var player)) ApplyPlayerState(player);
            StatusText = "音源已就緒";
            AppendLog($"已載入: {TrackTitle}");
        }

        private void OnDisconnected(Exception? error)
        {
            Dispatcher.UIThread.Post(() =>
            {
                IsConnected = false;
                IsAudioRunning = false;
                StatusText = "與引擎的連線中斷";
                AppendLog(error?.Message ?? "引擎已關閉。");
            });
        }

        // ── 指令 ────────────────────────────────────────────────────────
        [RelayCommand]
        private async Task RefreshDevicesAsync()
        {
            var result = await _client.CallAsync("get_devices", new { refresh = true });
            var devices = result.Deserialize<DeviceList>() ?? new DeviceList();

            _suppressPush = true;
            try
            {
                var previousHeadphone = HeadphoneDevice?.Index ?? -1;
                var previousVirtual = VirtualDevice?.Index ?? -1;
                var previousMic = MicDevice?.Index ?? -1;

                OutputDevices.Clear();
                OutputDevices.Add(AudioDevice.None);
                foreach (var device in devices.Outputs) OutputDevices.Add(device);

                InputDevices.Clear();
                InputDevices.Add(AudioDevice.None);
                foreach (var device in devices.Inputs) InputDevices.Add(device);

                HasVirtualOutput = devices.HasVirtualOutput;
                HasVirtualInput = devices.HasVirtualInput;
                VirtualFamilies = devices.VirtualFamilies is { Length: > 0 } f
                    ? string.Join("、", f) : "";
                OnPropertyChanged(nameof(VirtualStatusText));

                HeadphoneDevice = Pick(OutputDevices, previousHeadphone)
                    ?? devices.Outputs.FirstOrDefault(d => d.IsDefault && !d.IsVirtual)
                    ?? devices.Outputs.FirstOrDefault(d => !d.IsVirtual)
                    ?? AudioDevice.None;

                VirtualDevice = Pick(OutputDevices, previousVirtual)
                    ?? (devices.SuggestedVirtual is { } suggested
                        ? OutputDevices.FirstOrDefault(d => d.Index == suggested)
                        : null)
                    ?? AudioDevice.None;

                MicDevice = Pick(InputDevices, previousMic)
                    ?? devices.Inputs.FirstOrDefault(d => d.IsDefault)
                    ?? devices.Inputs.FirstOrDefault()
                    ?? AudioDevice.None;
            }
            finally
            {
                _suppressPush = false;
            }
            PushDevices();
        }

        private static AudioDevice? Pick(IEnumerable<AudioDevice> devices, int index) =>
            index < 0 ? null : devices.FirstOrDefault(d => d.Index == index);

        // ── 首次啟動精靈 ────────────────────────────────────────────────

        /// <summary>沒有虛擬麥克風、且使用者沒說過「略過」時,自動顯示引導。</summary>
        private void EvaluateSetupState()
        {
            _driverInstaller.Refresh();
            CanInstallBundledDriver = _driverInstaller.IsAvailable
                                      && _driverInstaller.HasDriverFiles;

            var ready = HasVirtualInput && HasVirtualOutput;
            if (ready)
            {
                if (!_uiSettings.SetupCompleted)
                {
                    _uiSettings.SetupCompleted = true;
                    _uiSettings.Save();
                }
                ShowSetupWizard = false;
                return;
            }

            if (!_uiSettings.SetupSkipped) ShowSetupWizard = true;
        }

        [RelayCommand]
        private void OpenSetupWizard()
        {
            SetupFailed = false;
            SetupMessage = "";
            OfferTestSigning = false;
            _driverInstaller.Refresh();
            CanInstallBundledDriver = _driverInstaller.IsAvailable
                                      && _driverInstaller.HasDriverFiles;
            ShowSetupWizard = true;
        }

        [RelayCommand]
        private void CloseSetupWizard() => ShowSetupWizard = false;

        [RelayCommand]
        private void SkipSetup()
        {
            _uiSettings.SetupSkipped = true;
            _uiSettings.Save();
            ShowSetupWizard = false;
            AppendLog("已略過虛擬麥克風設定。之後可從右欄的「設定虛擬麥克風」再開啟。");
        }

        /// <summary>使用者自行裝好驅動後按這個 —— 重新掃描裝置。</summary>
        [RelayCommand]
        private async Task RecheckDevicesAsync()
        {
            SetupMessage = "重新掃描裝置…";
            SetupFailed = false;
            await RefreshDevicesAsync();

            if (HasVirtualInput && HasVirtualOutput)
            {
                SetupMessage = $"找到了:{VirtualFamilies}";
                _uiSettings.SetupCompleted = true;
                _uiSettings.Save();
                await Task.Delay(900);
                ShowSetupWizard = false;
            }
            else
            {
                SetupFailed = true;
                SetupMessage = "還是沒有偵測到虛擬麥克風。剛裝好的驅動可能需要重新開機才會出現。";
            }
        }

        [RelayCommand]
        private Task InstallDriverAsync() => RunDriverInstallAsync(testSigning: false);

        [RelayCommand]
        private Task InstallDriverTestSigningAsync() => RunDriverInstallAsync(testSigning: true);

        private async Task RunDriverInstallAsync(bool testSigning)
        {
            IsInstallingDriver = true;
            SetupFailed = false;
            OfferTestSigning = false;
            SetupMessage = testSigning
                ? "正在以測試簽章模式安裝驅動…請在 UAC 對話框按「是」。"
                : "正在安裝驅動…請在 UAC 對話框按「是」。";

            try
            {
                var result = await _driverInstaller.InstallAsync(testSigning);
                if (!string.IsNullOrWhiteSpace(result.Log))
                    foreach (var line in result.Log.Split('\n', StringSplitOptions.RemoveEmptyEntries))
                        AppendLog($"[driver] {line.TrimEnd()}");

                SetupMessage = result.Message;
                SetupFailed = !result.Success;
                OfferTestSigning = result.SignatureRejected && !testSigning;

                if (result.Success)
                {
                    await RefreshDevicesAsync();
                    if (HasVirtualInput && HasVirtualOutput)
                    {
                        _uiSettings.SetupCompleted = true;
                        _uiSettings.Save();
                        SetupMessage = $"完成!已選好虛擬裝置:{VirtualFamilies}";
                        await Task.Delay(1200);
                        ShowSetupWizard = false;
                    }
                    else
                    {
                        SetupMessage = "驅動已安裝,但裝置還沒出現 —— 通常重新開機後就會好。";
                    }
                }
                else if (testSigning && result.Success == false && result.ExitCode == 0)
                {
                    SetupMessage = "測試簽章已啟用,請重新開機後再安裝一次驅動。";
                }
            }
            catch (Exception ex)
            {
                SetupFailed = true;
                SetupMessage = $"安裝失敗:{ex.Message}";
                AppendLog($"驅動安裝例外:{ex}");
            }
            finally
            {
                IsInstallingDriver = false;
            }
        }

        /// <summary>用預設瀏覽器開啟連結(驅動下載頁)。</summary>
        [RelayCommand]
        private void OpenUrl(string? url)
        {
            if (string.IsNullOrWhiteSpace(url)) return;
            try
            {
                System.Diagnostics.Process.Start(
                    new System.Diagnostics.ProcessStartInfo(url) { UseShellExecute = true });
            }
            catch (Exception ex)
            {
                AppendLog($"無法開啟連結:{ex.Message}");
            }
        }

        [RelayCommand]
        private async Task ToggleAudioAsync()
        {
            if (IsAudioRunning)
            {
                await SafeCallAsync("engine_stop");
                IsAudioRunning = false;
                StatusText = "音訊已停止";
                foreach (var meter in AllMeters()) meter.Reset();
            }
            else
            {
                if ((HeadphoneDevice?.Index ?? -1) < 0 && (VirtualDevice?.Index ?? -1) < 0)
                {
                    StatusText = "請至少選一個輸出裝置";
                    return;
                }
                var ok = await SafeCallAsync("engine_start");
                IsAudioRunning = ok;
                StatusText = ok ? "音訊執行中" : "無法啟動音訊";
            }
        }

        /// <summary>
        /// 「載入」是一個入口:貼網址或檔案路徑就直接載入,
        /// 打關鍵字就當成搜尋並列出候選讓使用者挑。
        /// </summary>
        [RelayCommand]
        private async Task LoadSourceAsync()
        {
            var input = (SourceInput ?? "").Trim();
            if (string.IsNullOrEmpty(input))
            {
                StatusText = "請輸入歌名關鍵字、YouTube 網址,或選擇本機檔案";
                return;
            }

            if (LooksLikeUrl(input) || File.Exists(input))
            {
                await LoadUrlAsync(input);
                return;
            }

            await SearchAsync(input);
        }

        private async Task LoadUrlAsync(string source)
        {
            IsBusy = true;
            Progress = 0;
            StatusText = "處理中…";
            if (!await SafeCallAsync("load", new { source, autoplay = true }))
                IsBusy = false;
        }

        [RelayCommand]
        private async Task SearchAsync(string? query = null)
        {
            query = (query ?? SourceInput ?? "").Trim();
            if (string.IsNullOrEmpty(query))
            {
                StatusText = "請輸入要搜尋的關鍵字";
                return;
            }

            ClearSearch();
            IsSearching = true;
            SearchHint = $"搜尋「{query}」…";
            StatusText = "搜尋中…";
            if (!await SafeCallAsync("search", new { query, limit = 12 }))
            {
                IsSearching = false;
                SearchHint = "";
            }
        }

        private static bool LooksLikeUrl(string text) =>
            Uri.TryCreate(text, UriKind.Absolute, out var uri)
            && (uri.Scheme == Uri.UriSchemeHttp || uri.Scheme == Uri.UriSchemeHttps);

        [RelayCommand]
        private async Task BrowseAsync()
        {
            if (PickAudioFileAsync is null) return;
            var path = await PickAudioFileAsync();
            if (!string.IsNullOrEmpty(path))
            {
                SourceInput = path;
                await LoadSourceAsync();
            }
        }

        [RelayCommand]
        private async Task CancelLoadAsync() => await SafeCallAsync("cancel_load");

        [RelayCommand]
        private async Task PlayPauseAsync() => await SafeCallAsync("transport", new { action = "toggle" });

        [RelayCommand]
        private async Task StopAsync() => await SafeCallAsync("transport", new { action = "stop" });

        [RelayCommand]
        private async Task ProbeDemucsAsync()
        {
            DemucsStatus = "偵測中…";
            try
            {
                var result = await _client.CallAsync("demucs_probe");
                if (result.TryGetProperty("available", out var available) && available.GetBoolean())
                {
                    var torch = result.GetProperty("torch").GetString();
                    var cuda = result.TryGetProperty("cuda", out var c) && c.GetBoolean();
                    DemucsStatus = $"可用 · torch {torch} · {(cuda ? "GPU 加速" : "CPU")}";
                }
                else
                {
                    var reason = result.TryGetProperty("reason", out var r) ? r.GetString() : "";
                    DemucsStatus = $"未安裝 —— {reason}";
                }
            }
            catch (Exception ex)
            {
                DemucsStatus = $"偵測失敗: {ex.Message}";
            }
        }

        [RelayCommand]
        private async Task SaveSettingsAsync()
        {
            if (await SafeCallAsync("save_settings")) StatusText = "設定已儲存";
        }

        [RelayCommand]
        private void ClearLog() => LogLines.Clear();

        private async Task<bool> SafeCallAsync(string command, object? args = null)
        {
            try
            {
                var result = await _client.CallAsync(command, args);
                if (command is "engine_start" or "engine_stop" or "engine_restart")
                    ApplyStatus(result);
                return true;
            }
            catch (Exception ex)
            {
                AppendLog($"{command}: {ex.Message}");
                StatusText = ex.Message;
                return false;
            }
        }

        private IEnumerable<MeterViewModel> AllMeters()
        {
            yield return MusicInMeter;
            yield return MusicOutMeter;
            yield return MicInMeter;
            yield return MicOutMeter;
            yield return HeadphoneMeter;
            yield return VirtualMeter;
        }

        private void AppendLog(string message)
        {
            if (string.IsNullOrWhiteSpace(message)) return;
            LogLines.Add($"[{DateTime.Now:HH:mm:ss}] {message}");
            while (LogLines.Count > 300) LogLines.RemoveAt(0);
        }

        private static string FormatTime(double seconds)
        {
            if (double.IsNaN(seconds) || seconds < 0) seconds = 0;
            var span = TimeSpan.FromSeconds(seconds);
            return span.TotalHours >= 1
                ? span.ToString(@"h\:mm\:ss", CultureInfo.InvariantCulture)
                : span.ToString(@"m\:ss", CultureInfo.InvariantCulture);
        }

        public async ValueTask DisposeAsync()
        {
            _sender?.Dispose();
            try
            {
                if (IsConnected) await _client.CallAsync("shutdown");
            }
            catch (Exception)
            {
                // 引擎可能已經先走了
            }
            await _client.DisposeAsync();
            _process.Dispose();
        }
    }
}
