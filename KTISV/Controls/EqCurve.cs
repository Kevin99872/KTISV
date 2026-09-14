using System;
using System.Collections.Generic;
using System.Globalization;
using Avalonia;
using Avalonia.Controls;
using Avalonia.Media;

namespace KTISV.Controls
{
    /// <summary>
    /// EQ 的合成頻率響應曲線。橫軸 20 Hz–20 kHz(對數),縱軸 ±18 dB。
    ///
    /// 曲線的數值由引擎的 <c>eq_response</c> 算出來 —— 不在這裡用 C# 再寫一份
    /// biquad 公式。兩份實作遲早會走鐘,而畫錯的曲線不會報錯,只會讓使用者
    /// 照著一條假的曲線調音。點是在對數頻率軸上等距取樣的。
    /// </summary>
    public sealed class EqCurve : Control
    {
        public static readonly StyledProperty<IReadOnlyList<double>?> ResponseDbProperty =
            AvaloniaProperty.Register<EqCurve, IReadOnlyList<double>?>(nameof(ResponseDb));

        public static readonly StyledProperty<bool> IsActiveProperty =
            AvaloniaProperty.Register<EqCurve, bool>(nameof(IsActive), true);

        public IReadOnlyList<double>? ResponseDb
        {
            get => GetValue(ResponseDbProperty);
            set => SetValue(ResponseDbProperty, value);
        }

        /// <summary>EQ 停用時曲線改成灰色,但仍顯示設定的形狀。</summary>
        public bool IsActive
        {
            get => GetValue(IsActiveProperty);
            set => SetValue(IsActiveProperty, value);
        }

        private const double MinHz = 20.0;
        private const double MaxHz = 20000.0;
        private const double RangeDb = 18.0;

        private static readonly IBrush Background = new SolidColorBrush(Color.FromRgb(0x1a, 0x1d, 0x22));
        private static readonly Pen GridPen = new(new SolidColorBrush(Color.FromArgb(0x30, 0xff, 0xff, 0xff)), 1);
        private static readonly Pen ZeroPen = new(new SolidColorBrush(Color.FromArgb(0x70, 0xff, 0xff, 0xff)), 1);
        private static readonly Pen ActivePen = new(new SolidColorBrush(Color.FromRgb(0x4f, 0xc3, 0xf7)), 2);
        private static readonly Pen InactivePen = new(new SolidColorBrush(Color.FromRgb(0x6b, 0x72, 0x80)), 2);
        private static readonly IBrush LabelBrush = new SolidColorBrush(Color.FromArgb(0x90, 0xff, 0xff, 0xff));
        private static readonly IBrush FillBrush = new SolidColorBrush(Color.FromArgb(0x28, 0x4f, 0xc3, 0xf7));

        private static readonly double[] GridHz = [50, 100, 200, 500, 1000, 2000, 5000, 10000];
        private static readonly double[] GridDb = [-12, -6, 6, 12];

        static EqCurve()
        {
            AffectsRender<EqCurve>(ResponseDbProperty, IsActiveProperty);
        }

        protected override Size MeasureOverride(Size availableSize)
            => new(double.IsInfinity(availableSize.Width) ? 400 : availableSize.Width, 120);

        public override void Render(DrawingContext context)
        {
            var bounds = Bounds;
            if (bounds.Width < 10 || bounds.Height < 10) return;
            var w = bounds.Width;
            var h = bounds.Height;

            context.DrawRectangle(Background, null, new Rect(0, 0, w, h), 4, 4);

            double X(double hz) => Math.Log10(hz / MinHz) / Math.Log10(MaxHz / MinHz) * w;
            double Y(double db) => h / 2 - Math.Clamp(db, -RangeDb, RangeDb) / RangeDb * (h / 2 - 4);

            var typeface = new Typeface(FontFamily.Default);
            foreach (var hz in GridHz)
            {
                var x = Math.Round(X(hz)) + 0.5;
                context.DrawLine(GridPen, new Point(x, 0), new Point(x, h));
                var label = hz >= 1000 ? $"{hz / 1000:0}k" : hz.ToString("0", CultureInfo.InvariantCulture);
                context.DrawText(new FormattedText(label, CultureInfo.InvariantCulture,
                    FlowDirection.LeftToRight, typeface, 9, LabelBrush), new Point(x + 2, h - 12));
            }
            foreach (var db in GridDb)
            {
                var y = Math.Round(Y(db)) + 0.5;
                context.DrawLine(GridPen, new Point(0, y), new Point(w, y));
                context.DrawText(new FormattedText($"{db:+0;-0}", CultureInfo.InvariantCulture,
                    FlowDirection.LeftToRight, typeface, 9, LabelBrush), new Point(2, y - 11));
            }
            var zero = Math.Round(Y(0)) + 0.5;
            context.DrawLine(ZeroPen, new Point(0, zero), new Point(w, zero));

            var points = ResponseDb;
            if (points is null || points.Count < 2) return;

            var line = new StreamGeometry();
            var fill = new StreamGeometry();
            using (var lc = line.Open())
            using (var fc = fill.Open())
            {
                fc.BeginFigure(new Point(0, zero), true);
                for (var i = 0; i < points.Count; i++)
                {
                    var p = new Point(i / (double)(points.Count - 1) * w, Y(points[i]));
                    if (i == 0) lc.BeginFigure(p, false);
                    else lc.LineTo(p);
                    fc.LineTo(p);
                }
                fc.LineTo(new Point(w, zero));
                fc.EndFigure(true);
                lc.EndFigure(false);
            }
            if (IsActive) context.DrawGeometry(FillBrush, null, fill);
            context.DrawGeometry(null, IsActive ? ActivePen : InactivePen, line);
        }
    }
}
