using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Globalization;
using System.Linq;
using System.Threading.Tasks;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using KTISV.Models;
using KTISV.Services;

namespace KTISV.ViewModels
{
    /// <summary>
    /// EQ 對引擎的出口。抽成介面是為了讓 <see cref="EqViewModel"/> 不必認識
    /// 連線、批次送出那一整套東西 —— 它只要知道「有兩組 EQ,各自能推參數」。
    /// </summary>
    public interface IEqBridge
    {
        /// <summary>推桿的高頻更新,會被收斂成批次。</summary>
        void PushGains(string target, double[] gains);

        void PushEnabled(string target, bool enabled);

        /// <summary>改單一頻段的頻率與 Q。不會動到濾波器狀態,所以不會爆音。</summary>
        void PushBandShape(string target, int index, double frequency, double q);

        /// <summary>新增頻段;回傳引擎排序、夾範圍之後的完整結果。</summary>
        Task<EqBandSpec[]?> AddBandAsync(string target, double frequency);

        Task<EqBandSpec[]?> RemoveBandAsync(string target, int index);
    }

    /// <summary>EQ 的一個頻段。頻率、增益、Q 都可以調,也可以整段刪掉。</summary>
    public sealed partial class EqBandViewModel : ViewModelBase
    {
        public const double MinFrequency = 20;
        public const double MaxFrequency = 20000;
        public const double MinQ = 0.1;
        public const double MaxQ = 18;

        private readonly EqViewModel _owner;
        private bool _suppress;

        public EqBandViewModel(EqViewModel owner, EqBandSpec spec)
        {
            _owner = owner;
            _frequency = spec.Frequency;
            _gainDb = spec.GainDb;
            _q = spec.Q;
        }

        [ObservableProperty] private double _frequency;
        [ObservableProperty] private double _gainDb;
        [ObservableProperty] private double _q;

        /// <summary>「刪除」按鈕要看整組還剩幾段才知道能不能按。</summary>
        public EqViewModel Owner => _owner;

        /// <summary>推桿下方的頻率標籤(1 kHz 以上改用 k 表示)。</summary>
        public string Label => Frequency >= 1000
            ? $"{Frequency / 1000:0.##}k"
            : $"{Frequency:0.##}";

        public string GainText => $"{GainDb:+0.0;-0.0;0.0}";

        /// <summary>輸入框裡的文字。看不懂就原樣留著,不要擅自改成 0。</summary>
        public string FrequencyText
        {
            get => Frequency.ToString(Frequency >= 100 ? "0" : "0.##",
                                      CultureInfo.CurrentCulture);
            set => Frequency = NumericText.ParseOr(value, Frequency,
                                                   MinFrequency, MaxFrequency);
        }

        public string QText
        {
            get => Q.ToString("0.##", CultureInfo.CurrentCulture);
            set => Q = NumericText.ParseOr(value, Q, MinQ, MaxQ);
        }

        /// <summary>兩端會自動變成 shelf,那時 Q 控制的是轉折斜率而不是頻寬。</summary>
        public string ShapeTip => _owner.IsShelf(this)
            ? "這一段在最外側,會自動變成 shelf;Q 控制的是轉折斜率。"
            : "peaking:Q 越大影響的頻率範圍越窄。";

        partial void OnGainDbChanged(double value)
        {
            OnPropertyChanged(nameof(GainText));
            if (!_suppress) _owner.OnGainChanged();
        }

        partial void OnFrequencyChanged(double value)
        {
            OnPropertyChanged(nameof(Label));
            OnPropertyChanged(nameof(FrequencyText));
            if (!_suppress) _owner.OnShapeChanged(this);
        }

        partial void OnQChanged(double value)
        {
            OnPropertyChanged(nameof(QText));
            if (!_suppress) _owner.OnShapeChanged(this);
        }

        [RelayCommand]
        private Task RemoveAsync() => _owner.RemoveBandAsync(this);

        /// <summary>從引擎回填時不要再送回去,避免無限往返。</summary>
        public void SetSilently(EqBandSpec spec)
        {
            _suppress = true;
            try
            {
                Frequency = spec.Frequency;
                GainDb = spec.GainDb;
                Q = spec.Q;
            }
            finally
            {
                _suppress = false;
            }
        }

        public void SetGainSilently(double value)
        {
            if (Math.Abs(GainDb - value) < 1e-6) return;
            _suppress = true;
            try { GainDb = value; }
            finally { _suppress = false; }
        }

        public EqBandSpec ToSpec() => new(Frequency, GainDb, Q);

        public void RaiseShapeTip() => OnPropertyChanged(nameof(ShapeTip));
    }

    /// <summary>一組可增刪頻段的 EQ。</summary>
    public sealed partial class EqViewModel : ViewModelBase
    {
        /// <summary>新增頻段時預設落在哪裡 —— 人聲咬字的區段,調的機會最大。</summary>
        private const double NewBandFrequency = 2000;

        public const int MaxBands = 24;

        private readonly IEqBridge _bridge;
        private bool _suppress;

        public EqViewModel(string target, string title,
                           IEnumerable<EqBandSpec> bands, IEqBridge bridge)
        {
            Target = target;
            Title = title;
            _bridge = bridge;
            Bands = [];
            Bands.CollectionChanged += (_, _) => RaiseStructureChanged();
            LoadBands(bands);
        }

        public string Target { get; }
        public string Title { get; }
        public ObservableCollection<EqBandViewModel> Bands { get; }

        [ObservableProperty] private bool _isEnabled = true;
        [ObservableProperty] private bool _isBusy;

        public bool CanAddBand => Bands.Count < MaxBands;
        public bool CanRemoveBand => Bands.Count > 1;

        public string BandSummary => $"{Bands.Count} 段";

        /// <summary>最低與最高的那一段由引擎自動當成 shelf。</summary>
        public bool IsShelf(EqBandViewModel band)
        {
            if (Bands.Count < 2) return false;
            var index = Bands.IndexOf(band);
            return index == 0 || index == Bands.Count - 1;
        }

        private void RaiseStructureChanged()
        {
            OnPropertyChanged(nameof(CanAddBand));
            OnPropertyChanged(nameof(CanRemoveBand));
            OnPropertyChanged(nameof(BandSummary));
            foreach (var band in Bands) band.RaiseShapeTip();
        }

        partial void OnIsEnabledChanged(bool value)
        {
            if (!_suppress) _bridge.PushEnabled(Target, value);
        }

        // ── 來自頻段的回呼 ──────────────────────────────────────────────
        internal void OnGainChanged()
        {
            if (_suppress) return;
            _bridge.PushGains(Target, [.. Bands.Select(b => b.GainDb)]);
        }

        internal void OnShapeChanged(EqBandViewModel band)
        {
            if (_suppress) return;
            var index = Bands.IndexOf(band);
            if (index < 0) return;
            _bridge.PushBandShape(Target, index, band.Frequency, band.Q);
        }

        // ── 增刪 ────────────────────────────────────────────────────────
        [RelayCommand]
        private async Task AddBandAsync()
        {
            if (IsBusy || !CanAddBand) return;
            IsBusy = true;
            try
            {
                // 引擎會依頻率排序、夾範圍,所以整組以它回傳的為準,
                // 而不是在這裡自己猜新的那一段會插到哪。
                var bands = await _bridge.AddBandAsync(Target, SuggestNewFrequency());
                if (bands is not null) LoadBands(bands);
            }
            finally
            {
                IsBusy = false;
            }
        }

        internal async Task RemoveBandAsync(EqBandViewModel band)
        {
            if (IsBusy || !CanRemoveBand) return;
            var index = Bands.IndexOf(band);
            if (index < 0) return;

            IsBusy = true;
            try
            {
                var bands = await _bridge.RemoveBandAsync(Target, index);
                if (bands is not null) LoadBands(bands);
            }
            finally
            {
                IsBusy = false;
            }
        }

        /// <summary>
        /// 新頻段放在最大的那個空隙中間(以對數計)。直接固定放 2 kHz 的話,
        /// 連按兩次就會疊在同一個地方,兩根推桿看起來一模一樣。
        /// </summary>
        private double SuggestNewFrequency()
        {
            if (Bands.Count < 2) return NewBandFrequency;

            var frequencies = Bands.Select(b => b.Frequency).OrderBy(f => f).ToList();
            var best = NewBandFrequency;
            var widest = 0.0;
            for (var i = 1; i < frequencies.Count; i++)
            {
                var gap = Math.Log10(frequencies[i]) - Math.Log10(frequencies[i - 1]);
                if (gap <= widest) continue;
                widest = gap;
                best = Math.Sqrt(frequencies[i - 1] * frequencies[i]);
            }
            return Math.Round(best);
        }

        // ── 預設集 ──────────────────────────────────────────────────────
        //
        // 以「頻率 → 增益」的曲線描述,套用時再依目前的頻段取值。頻段可以被
        // 使用者改成任何配置,寫死 10 個數字的話,刪掉一段就整條曲線錯位。
        private static readonly (double Hz, double Db)[] VoiceCurve =
            [(31.25, -6), (62.5, -4), (125, -1), (250, 0), (500, 1),
             (1000, 2.5), (2000, 3), (4000, 2), (8000, 1), (16000, -1)];

        private static readonly (double Hz, double Db)[] RadioCurve =
            [(31.25, -12), (62.5, -10), (125, -6), (250, 0), (500, 3),
             (1000, 4), (2000, 3), (4000, 0), (8000, -6), (16000, -12)];

        private static readonly (double Hz, double Db)[] SmileCurve =
            [(31.25, 4), (62.5, 3), (125, 1.5), (250, 0), (500, -1.5),
             (1000, -2), (2000, -1), (4000, 1), (8000, 3), (16000, 4)];

        [RelayCommand]
        private void Reset() => ApplyCurve([(20, 0), (20000, 0)]);

        /// <summary>人聲:切掉隆隆聲、拉出咬字的中高頻。</summary>
        [RelayCommand]
        private void PresetVoice() => ApplyCurve(VoiceCurve);

        /// <summary>電話音色:適合做效果,或壓掉背景低頻噪音。</summary>
        [RelayCommand]
        private void PresetRadio() => ApplyCurve(RadioCurve);

        /// <summary>微笑曲線:給音樂用的常見取向。</summary>
        [RelayCommand]
        private void PresetSmile() => ApplyCurve(SmileCurve);

        private void ApplyCurve((double Hz, double Db)[] curve)
        {
            foreach (var band in Bands)
                band.SetGainSilently(Interpolate(curve, band.Frequency));
            _bridge.PushGains(Target, [.. Bands.Select(b => b.GainDb)]);
        }

        /// <summary>在對數頻率軸上線性內插;兩端就取端點值。</summary>
        private static double Interpolate((double Hz, double Db)[] curve, double hz)
        {
            if (hz <= curve[0].Hz) return curve[0].Db;
            if (hz >= curve[^1].Hz) return curve[^1].Db;

            for (var i = 1; i < curve.Length; i++)
            {
                if (hz > curve[i].Hz) continue;
                var (loHz, loDb) = curve[i - 1];
                var (hiHz, hiDb) = curve[i];
                var t = (Math.Log10(hz) - Math.Log10(loHz))
                        / (Math.Log10(hiHz) - Math.Log10(loHz));
                return Math.Round(loDb + (hiDb - loDb) * t, 1);
            }
            return curve[^1].Db;
        }

        // ── 從引擎回填 ──────────────────────────────────────────────────
        /// <summary>整組頻段以引擎為準重建。只在增刪、或明確查詢之後呼叫。</summary>
        public void LoadBands(IEnumerable<EqBandSpec> bands)
        {
            var specs = bands.ToList();
            if (specs.Count == 0) return;

            _suppress = true;
            try
            {
                // 段數相同就原地更新 —— 重建整個集合會讓正在編輯的輸入框
                // 連同焦點一起被換掉。
                if (specs.Count == Bands.Count)
                {
                    for (var i = 0; i < specs.Count; i++) Bands[i].SetSilently(specs[i]);
                }
                else
                {
                    Bands.Clear();
                    foreach (var spec in specs) Bands.Add(new EqBandViewModel(this, spec));
                }
            }
            finally
            {
                _suppress = false;
            }
            RaiseStructureChanged();
        }

        /// <summary>週期性狀態回填 —— 只碰增益,結構交給 <see cref="LoadBands"/>。</summary>
        public void LoadFrom(double[] gains, bool enabled)
        {
            _suppress = true;
            try
            {
                // 段數對不上就是這筆狀態比某次增刪還舊 —— 照索引硬套會讓整排
                // 增益錯位一格。跳過就好,下一筆狀態自然是對的。
                if (gains.Length == Bands.Count)
                {
                    for (var i = 0; i < Bands.Count; i++)
                        Bands[i].SetGainSilently(gains[i]);
                }
                IsEnabled = enabled;
            }
            finally
            {
                _suppress = false;
            }
        }
    }
}
