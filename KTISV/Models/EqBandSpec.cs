namespace KTISV.Models
{
    /// <summary>EQ 一個頻段的完整設定,與引擎的 band_info() 一一對應。</summary>
    /// <param name="Frequency">中心頻率(Hz)。最低與最高的那一段會自動變成 shelf。</param>
    /// <param name="GainDb">增益,±15 dB。</param>
    /// <param name="Q">peaking 的話是頻寬,shelf 的話是轉折斜率。</param>
    public sealed record EqBandSpec(double Frequency, double GainDb, double Q);
}
