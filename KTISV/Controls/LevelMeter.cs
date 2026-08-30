using System;
using Avalonia;
using Avalonia.Controls;
using Avalonia.Media;

namespace KTISV.Controls
{
    /// <summary>
    /// 水平電平表:RMS 實心條 + 峰值保持線 + 削波指示。
    /// 刻度為 dBFS,以 -60 dB 為底、0 dB 為滿格,並在 -18 / -6 dB 畫刻度線。
    /// </summary>
    public sealed class LevelMeter : Control
    {
        public static readonly StyledProperty<double> RmsDbProperty =
            AvaloniaProperty.Register<LevelMeter, double>(nameof(RmsDb), -72.0);

        public static readonly StyledProperty<double> PeakDbProperty =
            AvaloniaProperty.Register<LevelMeter, double>(nameof(PeakDb), -72.0);

        public static readonly StyledProperty<double> HoldDbProperty =
            AvaloniaProperty.Register<LevelMeter, double>(nameof(HoldDb), -72.0);

        public static readonly StyledProperty<bool> IsClippingProperty =
            AvaloniaProperty.Register<LevelMeter, bool>(nameof(IsClipping));

        public double RmsDb { get => GetValue(RmsDbProperty); set => SetValue(RmsDbProperty, value); }
        public double PeakDb { get => GetValue(PeakDbProperty); set => SetValue(PeakDbProperty, value); }
        public double HoldDb { get => GetValue(HoldDbProperty); set => SetValue(HoldDbProperty, value); }
        public bool IsClipping { get => GetValue(IsClippingProperty); set => SetValue(IsClippingProperty, value); }

        private const double FloorDb = -60.0;
        private const double CeilingDb = 0.0;

        private static readonly IBrush TrackBrush = new SolidColorBrush(Color.FromRgb(0x22, 0x25, 0x2b));
        private static readonly IBrush TickBrush = new SolidColorBrush(Color.FromArgb(0x60, 0xff, 0xff, 0xff));
        private static readonly IBrush PeakBrush = new SolidColorBrush(Color.FromRgb(0xe8, 0xee, 0xf5));
        private static readonly IBrush ClipBrush = new SolidColorBrush(Color.FromRgb(0xff, 0x45, 0x3a));
        private static readonly IBrush ClipIdleBrush = new SolidColorBrush(Color.FromRgb(0x3a, 0x3f, 0x48));

        private static readonly LinearGradientBrush BarBrush = new()
        {
            StartPoint = new RelativePoint(0, 0, RelativeUnit.Relative),
            EndPoint = new RelativePoint(1, 0, RelativeUnit.Relative),
            GradientStops =
            {
                new GradientStop(Color.FromRgb(0x2e, 0xcc, 0x71), 0.00),
                new GradientStop(Color.FromRgb(0x2e, 0xcc, 0x71), 0.62),
                new GradientStop(Color.FromRgb(0xf1, 0xc4, 0x0f), 0.82),
                new GradientStop(Color.FromRgb(0xff, 0x6b, 0x35), 0.94),
                new GradientStop(Color.FromRgb(0xff, 0x45, 0x3a), 1.00),
            },
        };

        static LevelMeter()
        {
            AffectsRender<LevelMeter>(RmsDbProperty, PeakDbProperty,
                                      HoldDbProperty, IsClippingProperty);
        }

        protected override Size MeasureOverride(Size availableSize) => new(availableSize.Width, 14);

        public override void Render(DrawingContext context)
        {
            var bounds = Bounds;
            if (bounds.Width <= 1 || bounds.Height <= 0) return;

            const double clipWidth = 8.0;
            const double gap = 3.0;
            var trackWidth = Math.Max(0, bounds.Width - clipWidth - gap);
            var track = new Rect(0, 0, trackWidth, bounds.Height);

            context.DrawRectangle(TrackBrush, null, track, 2, 2);

            var rms = Normalize(RmsDb);
            if (rms > 0)
            {
                var barRect = new Rect(0, 0, trackWidth * rms, bounds.Height);
                // 漸層要對整條軌道取色,否則短音量條也會出現紅色
                using (context.PushClip(barRect))
                    context.DrawRectangle(BarBrush, null, track, 2, 2);
            }

            foreach (var tick in new[] { -18.0, -6.0 })
            {
                var x = Math.Round(trackWidth * Normalize(tick)) + 0.5;
                context.DrawLine(new Pen(TickBrush, 1),
                                 new Point(x, 1), new Point(x, bounds.Height - 1));
            }

            var hold = Normalize(HoldDb);
            if (hold > 0.001)
            {
                var x = Math.Min(trackWidth - 1.5, trackWidth * hold);
                context.DrawRectangle(PeakBrush, null,
                                      new Rect(x, 0, 1.5, bounds.Height));
            }

            context.DrawRectangle(IsClipping ? ClipBrush : ClipIdleBrush, null,
                                  new Rect(trackWidth + gap, 0, clipWidth, bounds.Height), 2, 2);
        }

        private static double Normalize(double db)
        {
            if (double.IsNaN(db) || db <= FloorDb) return 0.0;
            if (db >= CeilingDb) return 1.0;
            return (db - FloorDb) / (CeilingDb - FloorDb);
        }
    }
}
