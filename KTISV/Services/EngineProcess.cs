using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Text.RegularExpressions;
using System.Threading;
using System.Threading.Tasks;

namespace KTISV.Services
{
    /// <summary>
    /// 負責找到 Python、啟動 <c>python -m ktisv_engine</c>,並從 stdout 解析握手訊息。
    /// </summary>
    public sealed partial class EngineProcess : IDisposable
    {
        private Process? _process;

        public int Port { get; private set; }
        public string Token { get; private set; } = "";
        public string EngineVersion { get; private set; } = "";
        public string PythonPath { get; private set; } = "";
        public string EngineDirectory { get; private set; } = "";

        public bool IsRunning => _process is { HasExited: false };

        /// <summary>引擎自己印到 stderr 的訊息(通常是 Python 例外)。</summary>
        public event Action<string>? DiagnosticReceived;

        [GeneratedRegex(@"^KTISV_ENGINE port=(\d+) token=(\w+) version=(\S+)")]
        private static partial Regex HandshakeRegex();

        /// <summary>一種啟動引擎的方式:打包好的 exe,或某個 python 直譯器。</summary>
        private readonly record struct LaunchSpec(string FileName, string[] Args, bool IsPackaged)
        {
            public string Describe() => IsPackaged
                ? FileName
                : $"{FileName} {string.Join(' ', Args)}".Trim();
        }

        public bool UsingPackagedEngine { get; private set; }

        public async Task StartAsync(CancellationToken cancellation = default)
        {
            var (packaged, workingDir) = LocatePackagedEngine();
            EngineDirectory = LocateEngineDirectory() ?? workingDir ?? "";

            if (packaged is null && string.IsNullOrEmpty(EngineDirectory))
            {
                throw new InvalidOperationException(
                    "找不到引擎。預期會有打包好的 ktisv-engine.exe,或原始碼的 " +
                    "engine\\ktisv_engine\\__main__.py。");
            }

            var lastError = "";
            foreach (var spec in LaunchSpecs(packaged))
            {
                try
                {
                    if (await TryStartAsync(spec, cancellation))
                    {
                        PythonPath = spec.Describe();
                        UsingPackagedEngine = spec.IsPackaged;
                        return;
                    }
                }
                catch (Exception ex)
                {
                    lastError = ex.Message;
                }
                StopProcess();
            }

            throw new InvalidOperationException(
                "無法啟動音訊引擎。若是從原始碼執行,請在 engine 目錄執行 `uv sync`" +
                "(或 pip install -r requirements.txt);若是打包版,請確認 " +
                "ktisv-engine.exe 就在應用程式旁邊。\n" +
                (string.IsNullOrEmpty(lastError) ? "" : $"最後的錯誤: {lastError}"));
        }

        private IEnumerable<LaunchSpec> LaunchSpecs(string? packaged)
        {
            // 打包好的 exe 優先:不需要使用者裝 Python
            if (packaged is not null)
                yield return new LaunchSpec(packaged, [], IsPackaged: true);

            // 原始碼模式:試各種 python
            foreach (var (fileName, prefixArgs) in PythonCandidates())
                yield return new LaunchSpec(fileName, [.. prefixArgs, "-m", "ktisv_engine"],
                                            IsPackaged: false);
        }

        private async Task<bool> TryStartAsync(LaunchSpec spec, CancellationToken cancellation)
        {
            var workingDir = spec.IsPackaged
                ? Path.GetDirectoryName(spec.FileName) ?? EngineDirectory
                : EngineDirectory;

            var info = new ProcessStartInfo
            {
                FileName = spec.FileName,
                WorkingDirectory = workingDir,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                UseShellExecute = false,
                CreateNoWindow = true,
            };
            foreach (var arg in spec.Args) info.ArgumentList.Add(arg);
            info.Environment["PYTHONIOENCODING"] = "utf-8";
            info.Environment["PYTHONUNBUFFERED"] = "1";

            _process = Process.Start(info);
            if (_process is null) return false;

            _process.ErrorDataReceived += (_, e) =>
            {
                if (!string.IsNullOrWhiteSpace(e.Data))
                    DiagnosticReceived?.Invoke(e.Data);
            };
            _process.BeginErrorReadLine();

            // 等握手訊息;引擎啟動時要載入 numpy/scipy,給它久一點
            using var timeout = CancellationTokenSource.CreateLinkedTokenSource(cancellation);
            timeout.CancelAfter(TimeSpan.FromSeconds(60));

            while (!timeout.IsCancellationRequested)
            {
                var line = await _process.StandardOutput.ReadLineAsync(timeout.Token);
                if (line is null) break;   // 行程結束了

                var match = HandshakeRegex().Match(line);
                if (match.Success)
                {
                    Port = int.Parse(match.Groups[1].Value);
                    Token = match.Groups[2].Value;
                    EngineVersion = match.Groups[3].Value;
                    _ = PumpRemainingOutputAsync();
                    return true;
                }
                DiagnosticReceived?.Invoke(line);
            }
            return false;
        }

        private async Task PumpRemainingOutputAsync()
        {
            if (_process is null) return;
            try
            {
                while (await _process.StandardOutput.ReadLineAsync() is { } line)
                    DiagnosticReceived?.Invoke(line);
            }
            catch (Exception)
            {
                // 行程關閉時的正常結果
            }
        }

        private IEnumerable<(string FileName, string[] Args)> PythonCandidates()
        {
            var explicitPath = Environment.GetEnvironmentVariable("KTISV_PYTHON");
            if (!string.IsNullOrWhiteSpace(explicitPath) && File.Exists(explicitPath))
                yield return (explicitPath, []);

            foreach (var relative in new[]
            {
                Path.Combine(".venv", "Scripts", "python.exe"),
                Path.Combine(".venv", "bin", "python"),
            })
            {
                var candidate = Path.Combine(EngineDirectory, relative);
                if (File.Exists(candidate)) yield return (candidate, []);
            }

            yield return ("python", []);
            if (OperatingSystem.IsWindows()) yield return ("py", ["-3"]);
            yield return ("python3", []);
        }

        private static string? LocateEngineDirectory()
        {
            var dir = new DirectoryInfo(AppContext.BaseDirectory);
            for (var depth = 0; depth < 8 && dir is not null; depth++, dir = dir.Parent)
            {
                var candidate = Path.Combine(dir.FullName, "engine");
                if (File.Exists(Path.Combine(candidate, "ktisv_engine", "__main__.py")))
                    return candidate;
            }
            return null;
        }

        /// <summary>找打包好的 ktisv-engine.exe。回傳 (exe 路徑, 其所在資料夾)。</summary>
        private static (string? Exe, string? Dir) LocatePackagedEngine()
        {
            var exeName = OperatingSystem.IsWindows() ? "ktisv-engine.exe" : "ktisv-engine";
            var dir = new DirectoryInfo(AppContext.BaseDirectory);

            for (var depth = 0; depth < 8 && dir is not null; depth++, dir = dir.Parent)
            {
                foreach (var relative in new[]
                {
                    Path.Combine("engine-bin", exeName),          // 隨前端一起發佈時
                    exeName,                                       // 就放在旁邊
                    Path.Combine("engine", "dist", "ktisv-engine", exeName),  // 開發樹
                })
                {
                    var candidate = Path.Combine(dir.FullName, relative);
                    if (File.Exists(candidate))
                        return (candidate, Path.GetDirectoryName(candidate));
                }
            }
            return (null, null);
        }

        private void StopProcess()
        {
            if (_process is null) return;
            try
            {
                if (!_process.HasExited)
                {
                    _process.Kill(entireProcessTree: true);
                    _process.WaitForExit(3000);
                }
            }
            catch (Exception)
            {
                // 行程可能已經自己結束了
            }
            finally
            {
                _process.Dispose();
                _process = null;
            }
        }

        public void Dispose() => StopProcess();
    }
}
