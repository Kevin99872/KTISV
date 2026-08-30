using System.Text.Json;
using CommunityToolkit.Mvvm.ComponentModel;

namespace KTISV.ViewModels
{
    /// <summary>單一量測點的電平顯示。</summary>
    public sealed partial class MeterViewModel(string caption) : ViewModelBase
    {
        [ObservableProperty] private double _rmsDb = -72;
        [ObservableProperty] private double _peakDb = -72;
        [ObservableProperty] private double _holdDb = -72;
        [ObservableProperty] private bool _isClipping;

        public string Caption { get; } = caption;

        /// <summary>削波指示要亮一下才看得到,所以用倒數而不是單一 block 的旗標。</summary>
        private int _clipCountdown;

        public void Update(JsonElement data)
        {
            if (data.ValueKind != JsonValueKind.Object) return;

            if (data.TryGetProperty("rms", out var rms)) RmsDb = rms.GetDouble();
            if (data.TryGetProperty("peak", out var peak)) PeakDb = peak.GetDouble();
            if (data.TryGetProperty("hold", out var hold)) HoldDb = hold.GetDouble();

            if (data.TryGetProperty("clip", out var clip) && clip.GetBoolean())
                _clipCountdown = 25;               // 約 1 秒 @ 25 Hz
            else if (_clipCountdown > 0)
                _clipCountdown--;

            IsClipping = _clipCountdown > 0;
        }

        public void Reset()
        {
            RmsDb = PeakDb = HoldDb = -72;
            _clipCountdown = 0;
            IsClipping = false;
        }
    }
}
