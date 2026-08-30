using System;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Threading.Tasks;

namespace KTISV.Services
{
    /// <summary>驅動安裝的結果。</summary>
    public sealed record DriverInstallResult(
        bool Success,
        int ExitCode,
        string Message,
        string Log,
        bool SignatureRejected = false,
        bool Cancelled = false);

    /// <summary>
    /// 安裝 / 移除選用的虛擬音訊驅動。
    ///
    /// 實際工作交給 driver\install-driver.ps1(需要系統管理員權限),
    /// 這裡負責定位資料夾、以 UAC 提權執行、把結束碼翻成看得懂的訊息。
    /// </summary>
    public sealed class DriverInstaller
    {
        /// <summary>驅動資料夾;找不到時為 null。</summary>
        public string? DriverDirectory { get; private set; }

        /// <summary>資料夾裡是否已備妥驅動檔(有 .inf)。</summary>
        public bool HasDriverFiles { get; private set; }

        /// <summary>是否備妥 nefconw.exe(建立裝置節點必需)。</summary>
        public bool HasNefcon { get; private set; }

        public DriverInstaller() => Refresh();

        public void Refresh()
        {
            DriverDirectory = LocateDriverDirectory();
            HasDriverFiles = DriverDirectory is not null
                && Directory.EnumerateFiles(DriverDirectory, "*.inf", SearchOption.AllDirectories).Any();
            HasNefcon = DriverDirectory is not null
                && Directory.EnumerateFiles(DriverDirectory, "nefconw.exe", SearchOption.AllDirectories).Any();
        }

        public string ScriptPath =>
            DriverDirectory is null ? "" : Path.Combine(DriverDirectory, "install-driver.ps1");

        public bool IsAvailable => DriverDirectory is not null && File.Exists(ScriptPath);

        /// <summary>安裝驅動。<paramref name="testSigning"/> 是簽章被拒時的最後手段。</summary>
        public Task<DriverInstallResult> InstallAsync(bool testSigning = false) =>
            RunScriptAsync(testSigning ? ["-TestSigning"] : []);

        public Task<DriverInstallResult> UninstallAsync() => RunScriptAsync(["-Uninstall"]);

        /// <summary>只解析 INF 印出硬體 ID —— 不需提權,用來確認檔案是否備妥。</summary>
        public Task<DriverInstallResult> ParseOnlyAsync() =>
            RunScriptAsync(["-ParseOnly"], elevate: false);

        private async Task<DriverInstallResult> RunScriptAsync(string[] extraArgs,
                                                               bool elevate = true)
        {
            if (!IsAvailable)
            {
                return new DriverInstallResult(false, -1,
                    "找不到驅動安裝腳本。請確認 driver 資料夾存在,並依其 README 放入驅動檔。",
                    "");
            }
            if (!HasDriverFiles)
            {
                return new DriverInstallResult(false, 4,
                    "driver 資料夾裡沒有 .inf 驅動檔。請先依 driver\\README.md 下載並放入檔案。",
                    "");
            }

            var logPath = Path.Combine(Path.GetTempPath(),
                                       $"ktisv-driver-{Guid.NewGuid():N}.log");
            try
            {
                var exitCode = await RunPowerShellAsync(extraArgs, logPath, elevate);
                var log = File.Exists(logPath) ? await File.ReadAllTextAsync(logPath) : "";
                return Interpret(exitCode, log);
            }
            catch (OperationCanceledException)
            {
                return new DriverInstallResult(false, -2, "已取消(使用者未同意提權)。", "",
                                               Cancelled: true);
            }
            catch (Exception ex)
            {
                return new DriverInstallResult(false, -3, $"執行安裝腳本失敗:{ex.Message}", "");
            }
            finally
            {
                try { if (File.Exists(logPath)) File.Delete(logPath); }
                catch (IOException) { /* 暫存檔刪不掉無所謂 */ }
            }
        }

        private Task<int> RunPowerShellAsync(string[] extraArgs, string logPath, bool elevate)
        {
            // UAC 提權(Verb=runas)必須 UseShellExecute=true,而那時就不能重導向
            // 標準輸出。所以讓 PowerShell 自己把所有輸出串流寫進暫存檔,事後再讀。
            var args = string.Join(" ", extraArgs);
            var command =
                $"& {Quote(ScriptPath)} -DriverPath {Quote(DriverDirectory!)} {args} " +
                $"*> {Quote(logPath)}; exit $LASTEXITCODE";

            var info = new ProcessStartInfo
            {
                FileName = "powershell.exe",
                UseShellExecute = true,
                CreateNoWindow = false,
                WindowStyle = ProcessWindowStyle.Hidden,
            };
            info.ArgumentList.Add("-NoProfile");
            info.ArgumentList.Add("-ExecutionPolicy");
            info.ArgumentList.Add("Bypass");
            info.ArgumentList.Add("-Command");
            info.ArgumentList.Add(command);

            if (elevate) info.Verb = "runas";

            var process = Process.Start(info);
            if (process is null)
            {
                // 使用者在 UAC 對話框按了「否」
                throw new OperationCanceledException();
            }

            var tcs = new TaskCompletionSource<int>();
            process.EnableRaisingEvents = true;
            process.Exited += (_, _) =>
            {
                tcs.TrySetResult(process.ExitCode);
                process.Dispose();
            };
            if (process.HasExited)
            {
                tcs.TrySetResult(process.ExitCode);
                process.Dispose();
            }
            return tcs.Task;
        }

        /// <summary>PowerShell 單引號字串:內部的單引號要成對加倍。</summary>
        private static string Quote(string value) => "'" + value.Replace("'", "''") + "'";

        /// <summary>把腳本的結束碼翻成使用者看得懂的說明(對應 driver\README.md)。</summary>
        private static DriverInstallResult Interpret(int exitCode, string log) => exitCode switch
        {
            0 => new DriverInstallResult(true, 0,
                "驅動安裝完成。KTISV 會重新掃描裝置;Discord 的輸入請選「Virtual Mic」。", log),

            2 => new DriverInstallResult(false, 2,
                "需要系統管理員權限才能安裝驅動。", log),

            3 or 4 => new DriverInstallResult(false, exitCode,
                "driver 資料夾裡沒有驅動檔。請依 driver\\README.md 下載並放入。", log),

            5 => new DriverInstallResult(false, 5,
                "無法從 INF 解析硬體 ID。這個驅動的格式可能不受支援。", log),

            6 => new DriverInstallResult(false, 6,
                "缺少 nefconw.exe。虛擬裝置沒有實體硬體可觸發安裝,需要它建立裝置節點。", log),

            7 => new DriverInstallResult(false, 7,
                "驅動安裝被拒絕,通常是簽章問題。這個開源驅動的憑證不是 Microsoft 核心驅動"
                + "認證,在開啟 Secure Boot 的系統上可能無法載入。建議改用 VB-CABLE。",
                log, SignatureRejected: true),

            8 => new DriverInstallResult(false, 8,
                "建立裝置節點失敗。", log),

            9 => new DriverInstallResult(false, 9,
                "裝置已建立但無法正常啟動,通常是簽章未被系統接受。建議改用 VB-CABLE。",
                log, SignatureRejected: true),

            _ => new DriverInstallResult(false, exitCode,
                $"驅動安裝失敗(結束碼 {exitCode})。詳見記錄。", log),
        };

        /// <summary>往上找 driver 資料夾:發佈版在 exe 旁邊,開發樹在方案根目錄。</summary>
        private static string? LocateDriverDirectory()
        {
            var dir = new DirectoryInfo(AppContext.BaseDirectory);
            for (var depth = 0; depth < 8 && dir is not null; depth++, dir = dir.Parent)
            {
                var candidate = Path.Combine(dir.FullName, "driver");
                if (File.Exists(Path.Combine(candidate, "install-driver.ps1")))
                    return candidate;
            }
            return null;
        }
    }
}
