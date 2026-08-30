using System;
using System.IO;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace KTISV.Services
{
    /// <summary>
    /// 前端自己的小設定檔(引擎的 settings.json 存的是音訊參數,
    /// 這裡存的是介面狀態,例如首次啟動精靈有沒有被略過)。
    /// </summary>
    public sealed class UiSettings
    {
        [JsonPropertyName("setup_completed")]
        public bool SetupCompleted { get; set; }

        /// <summary>使用者按過「先略過」—— 之後不再自動彈出精靈。</summary>
        [JsonPropertyName("setup_skipped")]
        public bool SetupSkipped { get; set; }

        /// <summary>介面語言代碼(zh-Hant / en)。</summary>
        [JsonPropertyName("language")]
        public string Language { get; set; } = "zh-Hant";

        /// <summary>WASAPI 獨佔模式。</summary>
        [JsonPropertyName("exclusive_mode")]
        public bool ExclusiveMode { get; set; }

        /// <summary>
        /// 音訊 block 大小(取樣)。低延遲設定是使用者在自己機器上調出來的,
        /// 不記住的話每次重開都要重調一遍。
        /// </summary>
        [JsonPropertyName("blocksize")]
        public int BlockSize { get; set; } = 128;

        private static string FilePath
        {
            get
            {
                var dir = Path.Combine(
                    Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
                    "KTISV");
                Directory.CreateDirectory(dir);
                return Path.Combine(dir, "ui-settings.json");
            }
        }

        public static UiSettings Load()
        {
            try
            {
                if (File.Exists(FilePath))
                {
                    var json = File.ReadAllText(FilePath);
                    return JsonSerializer.Deserialize<UiSettings>(json) ?? new UiSettings();
                }
            }
            catch (Exception)
            {
                // 設定檔壞了就當作全新安裝,不值得打斷啟動
            }
            return new UiSettings();
        }

        public void Save()
        {
            try
            {
                var json = JsonSerializer.Serialize(this,
                    new JsonSerializerOptions { WriteIndented = true });
                File.WriteAllText(FilePath, json);
            }
            catch (Exception)
            {
                // 存不起來只影響「下次還會再問一次」,不是致命問題
            }
        }
    }
}
