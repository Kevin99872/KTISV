using System;
using System.Collections.Generic;
using Avalonia;
using Avalonia.Controls;
using Avalonia.Data;
using Avalonia.Input;
using Avalonia.Media;

namespace KTISV.Controls
{
    /// <summary>
    /// 以音軌波形呈現的進度條:整首歌的峰值包絡,已播放的部分亮起。點擊或拖動即跳轉。
    /// 只有原曲時上下對稱;有人聲分軌時上半是原曲、下半是人聲 —— 疊在一起的話,
    /// 人聲為主的段落兩者幾乎一樣高,原曲會被整個蓋掉。
    /// 包絡由引擎在載入時算好(0–255),這裡只負責畫。
    /// </summary>
    public sealed class WaveformSeekBar : Control
    {
        public static readonly StyledProperty<double> ValueProperty =
            AvaloniaProperty.Register<WaveformSeekBar, double>(nameof(Value),
                defaultBindingMode: BindingMode.TwoWay);

        public static readonly StyledProperty<double> MaximumProperty =
            AvaloniaProperty.Register<WaveformSeekBar, double>(nameof(Maximum));

        public static readonly StyledProperty<IReadOnlyList<byte>?> PeaksProperty =
            AvaloniaProperty.Register<WaveformSeekBar, IReadOnlyList<byte>?>(nameof(Peaks));

        public static readonly StyledProperty<IReadOnlyList<byte>?> VocalPeaksProperty =
            AvaloniaProperty.Register<WaveformSeekBar, IReadOnlyList<byte>?>(nameof(VocalPeaks));

        public double Value { get => GetValue(ValueProperty); set => SetValue(ValueProperty, value); }
        public double Maximum { get => GetValue(MaximumProperty); set => SetValue(MaximumProperty, value); }
        public IReadOnlyList<byte>? Peaks { get => GetValue(PeaksProperty); set => SetValue(PeaksProperty, value); }
        public IReadOnlyList<byte>? VocalPeaks { get => GetValue(VocalPeaksProperty); set => SetValue(VocalPeaksProperty, value); }

        private static readonly IBrush BackgroundBrush = new SolidColorBrush(Color.FromRgb(0x1c, 0x1f, 0x25));
        private static readonly IBrush PlayedBrush = new SolidColorBrush(Color.FromRgb(0x4a, 0x92, 0xea));
        private static readonly IBrush UnplayedBrush = new SolidColorBrush(Color.FromRgb(0x3a, 0x40, 0x4a));
        private static readonly IBrush VocalPlayedBrush = new SolidColorBrush(Color.FromRgb(0xf0, 0xc6, 0x74));
        private static readonly IBrush VocalUnplayedBrush = new SolidColorBrush(Color.FromRgb(0x5a, 0x50, 0x3c));
        private static readonly IPen CenterPen = new Pen(new SolidColorBrush(Color.FromRgb(0x2a, 0x2e, 0x36)), 1);
        private static readonly IPen PlayheadPen = new Pen(new SolidColorBrush(Color.FromRgb(0xe8, 0xee, 0xf5)), 2);
        private static readonly IPen HoverPen = new Pen(new SolidColorBrush(Color.FromArgb(0x80, 0xe8, 0xee, 0xf5)), 1);

        private const double BarWidth = 2.0;
        private const double BarGap = 1.0;

        /// <summary>開始拖動(按下)。在第一次改變 <see cref="Value"/> 之前觸發。</summary>
        public event EventHandler? SeekStarted;

        /// <summary>結束拖動(放開或失去指標擷取)。</summary>
        public event EventHandler? SeekCompleted;

        private double? _hoverX;
        private bool _dragging;

        static WaveformSeekBar()
        {
            AffectsRender<WaveformSeekBar>(ValueProperty, MaximumProperty,
                                           PeaksProperty, VocalPeaksProperty);
            FocusableProperty.OverrideDefaultValue<WaveformSeekBar>(false);
        }

        public WaveformSeekBar()
        {
            Cursor = new Cursor(StandardCursorType.Hand);
        }

        protected override Size MeasureOverride(Size availableSize)
            => new(double.IsInfinity(availableSize.Width) ? 200 : availableSize.Width, 40);

        public override void Render(DrawingContext context)
        {
            var w = Bounds.Width;
            var h = Bounds.Height;
            if (w <= 1 || h <= 1) return;

            context.DrawRectangle(BackgroundBrush, null, new Rect(0, 0, w, h), 4, 4);
            var mid = Math.Round(h / 2) + 0.5;
            var half = h / 2 - 3;
            var progress = Maximum > 0 ? Math.Clamp(Value / Maximum, 0, 1) : 0;
            var playX = progress * w;

            var peaks = Peaks;
            if (peaks is null || peaks.Count == 0)
            {
                // 還沒有波形(沒載入):退化成一條細軌道,仍可顯示進度
                context.DrawLine(CenterPen, new Point(0, mid), new Point(w, mid));
                if (playX > 0)
                    context.DrawRectangle(PlayedBrush, null, new Rect(0, mid - 1.5, playX, 3), 1.5, 1.5);
            }
            else
            {
                var vocals = VocalPeaks is { Count: > 0 } v ? v : null;
                DrawBars(context, peaks, w, mid, half, playX, PlayedBrush, UnplayedBrush,
                         up: true, down: vocals is null);
                if (vocals is not null)
                    DrawBars(context, vocals, w, mid, half, playX, VocalPlayedBrush, VocalUnplayedBrush,
                             up: false, down: true);
            }

            if (_hoverX is { } hover && !_dragging)
                context.DrawLine(HoverPen, new Point(hover, 2), new Point(hover, h - 2));
            if (Maximum > 0)
            {
                var x = Math.Clamp(playX, 1, w - 1);
                context.DrawLine(PlayheadPen, new Point(x, 1), new Point(x, h - 1));
            }
        }

        private static void DrawBars(DrawingContext context, IReadOnlyList<byte> peaks, double w,
                                     double mid, double half, double playX, IBrush played, IBrush unplayed,
                                     bool up, bool down)
        {
            var step = BarWidth + BarGap;
            var count = Math.Max(1, (int)(w / step));
            for (var i = 0; i < count; i++)
            {
                // 一根柱子可能涵蓋好幾格包絡:取其中最大值,才不會漏掉短促的峰
                var from = (int)((long)i * peaks.Count / count);
                var to = Math.Max(from + 1, (int)((long)(i + 1) * peaks.Count / count));
                byte peak = 0;
                for (var j = from; j < to && j < peaks.Count; j++)
                    if (peaks[j] > peak) peak = peaks[j];

                var x = i * step;
                var height = Math.Max(1.0, peak / 255.0 * half);
                var brush = x + BarWidth / 2 < playX ? played : unplayed;
                var top = up ? mid - height : mid + 0.5;
                var bottom = down ? mid + height : mid - 0.5;
                context.DrawRectangle(brush, null, new Rect(x, top, BarWidth, bottom - top));
            }
        }

        // ── 互動 ────────────────────────────────────────────────────────
        protected override void OnPointerPressed(PointerPressedEventArgs e)
        {
            base.OnPointerPressed(e);
            if (!e.GetCurrentPoint(this).Properties.IsLeftButtonPressed || Maximum <= 0) return;
            _dragging = true;
            SeekStarted?.Invoke(this, EventArgs.Empty);
            e.Pointer.Capture(this);
            SeekTo(e.GetPosition(this).X);
            e.Handled = true;
        }

        protected override void OnPointerMoved(PointerEventArgs e)
        {
            base.OnPointerMoved(e);
            var x = e.GetPosition(this).X;
            _hoverX = Math.Clamp(x, 0, Bounds.Width);
            if (_dragging) SeekTo(x);
            else InvalidateVisual();
        }

        protected override void OnPointerReleased(PointerReleasedEventArgs e)
        {
            base.OnPointerReleased(e);
            if (!_dragging) return;
            e.Pointer.Capture(null);   // 觸發 OnPointerCaptureLost → 結束拖動
            EndDrag();
            e.Handled = true;
        }

        protected override void OnPointerCaptureLost(PointerCaptureLostEventArgs e)
        {
            base.OnPointerCaptureLost(e);
            EndDrag();
        }

        private void EndDrag()
        {
            if (!_dragging) return;
            _dragging = false;
            SeekCompleted?.Invoke(this, EventArgs.Empty);
            InvalidateVisual();
        }

        protected override void OnPointerExited(PointerEventArgs e)
        {
            base.OnPointerExited(e);
            _hoverX = null;
            InvalidateVisual();
        }

        private void SeekTo(double x)
        {
            if (Bounds.Width <= 0 || Maximum <= 0) return;
            Value = Math.Clamp(x / Bounds.Width, 0, 1) * Maximum;
        }
    }
}
