using System.Text.Json.Serialization;

namespace KTISV.Models
{
    /// <summary>引擎回報的一個音訊裝置。</summary>
    public sealed class AudioDevice
    {
        [JsonPropertyName("index")] public int Index { get; set; } = -1;
        [JsonPropertyName("name")] public string Name { get; set; } = "";
        [JsonPropertyName("hostapi")] public string HostApi { get; set; } = "";
        [JsonPropertyName("channels")] public int Channels { get; set; }
        [JsonPropertyName("samplerate")] public int SampleRate { get; set; }
        [JsonPropertyName("virtual")] public bool IsVirtual { get; set; }
        [JsonPropertyName("default")] public bool IsDefault { get; set; }

        /// <summary>下拉選單顯示用。虛擬音效卡加上記號,方便一眼找到。</summary>
        public string Display => IsVirtual ? $"🔌 {Name}  ·  {HostApi}" : $"{Name}  ·  {HostApi}";

        public override string ToString() => Display;

        /// <summary>「不使用」的佔位項目。</summary>
        public static AudioDevice None { get; } = new() { Index = -1, Name = "(不使用)", HostApi = "" };
    }

    public sealed class DeviceList
    {
        [JsonPropertyName("inputs")] public AudioDevice[] Inputs { get; set; } = [];
        [JsonPropertyName("outputs")] public AudioDevice[] Outputs { get; set; } = [];
        [JsonPropertyName("has_virtual_output")] public bool HasVirtualOutput { get; set; }
        [JsonPropertyName("has_virtual_input")] public bool HasVirtualInput { get; set; }
        [JsonPropertyName("virtual_families")] public string[] VirtualFamilies { get; set; } = [];
        [JsonPropertyName("suggested_virtual")] public int? SuggestedVirtual { get; set; }
    }
}