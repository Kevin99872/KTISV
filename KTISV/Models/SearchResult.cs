using System.Text.Json.Serialization;

namespace KTISV.Models
{
    /// <summary>YouTube 搜尋結果的一筆。</summary>
    public sealed class SearchResult
    {
        [JsonPropertyName("url")] public string Url { get; set; } = "";
        [JsonPropertyName("title")] public string Title { get; set; } = "";
        [JsonPropertyName("duration")] public double Duration { get; set; }
        [JsonPropertyName("duration_text")] public string DurationText { get; set; } = "—";
        [JsonPropertyName("uploader")] public string Uploader { get; set; } = "";
        [JsonPropertyName("video_id")] public string VideoId { get; set; } = "";
        [JsonPropertyName("views")] public long Views { get; set; }

        /// <summary>「頻道 · 觀看次數」的副標。</summary>
        public string Subtitle => Views > 0
            ? $"{Uploader}  ·  {FormatViews(Views)} 次觀看"
            : Uploader;

        private static string FormatViews(long views) => views switch
        {
            >= 100_000_000 => $"{views / 100_000_000.0:0.#} 億",
            >= 10_000 => $"{views / 10_000.0:0.#} 萬",
            _ => views.ToString(),
        };
    }
}
