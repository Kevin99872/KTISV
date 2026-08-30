using System;
using System.Globalization;

namespace KTISV.Services
{
    /// <summary>
    /// 把輸入框裡的字轉成數字。
    ///
    /// 刻意寬鬆:使用者會把介面上顯示的字整串貼回去(「16k」「250 ms」「1.5kHz」),
    /// 那些都該讀得懂,而不是丟一個紅框回去。真的看不懂就回 false,由呼叫端
    /// 決定要不要保留原本的值 —— 這比擅自改成 0 安全得多。
    /// </summary>
    public static class NumericText
    {
        private static readonly string[] Units = ["ms", "hz", "db", "s"];

        public static bool TryParse(string? text, out double value)
        {
            value = 0;
            var span = (text ?? "").Trim();
            if (span.Length == 0) return false;

            foreach (var unit in Units)
            {
                if (span.EndsWith(unit, StringComparison.OrdinalIgnoreCase))
                {
                    span = span[..^unit.Length].TrimEnd();
                    break;
                }
            }

            var multiplier = 1.0;
            if (span.EndsWith('k') || span.EndsWith('K'))
            {
                multiplier = 1000.0;
                span = span[..^1].TrimEnd();
            }

            const NumberStyles styles = NumberStyles.Float;
            if (!double.TryParse(span, styles, CultureInfo.CurrentCulture, out var parsed)
                && !double.TryParse(span, styles, CultureInfo.InvariantCulture, out parsed))
            {
                return false;
            }

            if (double.IsNaN(parsed) || double.IsInfinity(parsed)) return false;
            value = parsed * multiplier;
            return true;
        }

        /// <summary>解析並夾在範圍內;讀不懂就回傳 <paramref name="fallback"/>。</summary>
        public static double ParseOr(string? text, double fallback,
                                     double minimum, double maximum) =>
            TryParse(text, out var value)
                ? Math.Clamp(value, minimum, maximum)
                : fallback;
    }
}
