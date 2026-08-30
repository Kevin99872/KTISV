using System;
using System.Collections.Concurrent;
using System.IO;
using System.Net.Sockets;
using System.Text;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;

namespace KTISV.Services
{
    /// <summary>引擎回傳 ok=false 時丟出。</summary>
    public sealed class EngineCommandException(string command, string message)
        : Exception($"{command}: {message}")
    {
        public string Command { get; } = command;
    }

    /// <summary>
    /// 與 Python 引擎對話的 JSON-lines 客戶端。
    /// 指令採請求/回應配對,引擎主動推送的則以 <see cref="EventReceived"/> 發出。
    /// </summary>
    public sealed class EngineClient : IAsyncDisposable
    {
        private readonly ConcurrentDictionary<int, TaskCompletionSource<JsonElement>> _pending = new();
        private static readonly JsonSerializerOptions JsonOptions = new()
        {
            Encoder = System.Text.Encodings.Web.JavaScriptEncoder.UnsafeRelaxedJsonEscaping,
        };

        private TcpClient? _tcp;
        private NetworkStream? _stream;
        private StreamWriter? _writer;
        private CancellationTokenSource? _cancellation;
        private Task? _readerTask;
        private int _nextId;
        private readonly SemaphoreSlim _writeLock = new(1, 1);

        public bool IsConnected => _tcp?.Connected == true;

        /// <summary>引擎推送的事件:(事件名稱, 資料)。在背景執行緒上觸發。</summary>
        public event Action<string, JsonElement>? EventReceived;

        /// <summary>連線中斷(引擎當掉或關閉)。</summary>
        public event Action<Exception?>? Disconnected;

        /// <summary>某一則訊息處理失敗,但連線仍然存活。</summary>
        public event Action<Exception>? DispatchError;

        public async Task ConnectAsync(int port, string token,
                                       CancellationToken cancellation = default)
        {
            _tcp = new TcpClient { NoDelay = true };
            await _tcp.ConnectAsync("127.0.0.1", port, cancellation);
            _stream = _tcp.GetStream();
            _writer = new StreamWriter(_stream, new UTF8Encoding(false)) { AutoFlush = false };

            _cancellation = new CancellationTokenSource();
            _readerTask = Task.Run(() => ReadLoopAsync(_cancellation.Token));

            await CallAsync("hello", new { token }, cancellation);
        }

        public async Task<JsonElement> CallAsync(string command, object? args = null,
                                                 CancellationToken cancellation = default)
        {
            if (_writer is null) throw new InvalidOperationException("尚未連線到引擎。");

            var id = Interlocked.Increment(ref _nextId);
            var tcs = new TaskCompletionSource<JsonElement>(
                TaskCreationOptions.RunContinuationsAsynchronously);
            _pending[id] = tcs;

            var payload = JsonSerializer.Serialize(
                new { id, cmd = command, args = args ?? new { } }, JsonOptions);

            await _writeLock.WaitAsync(cancellation);
            try
            {
                await _writer.WriteAsync(payload.AsMemory(), cancellation);
                await _writer.WriteAsync("\n".AsMemory(), cancellation);
                await _writer.FlushAsync(cancellation);
            }
            catch
            {
                _pending.TryRemove(id, out _);
                throw;
            }
            finally
            {
                _writeLock.Release();
            }

            using var registration = cancellation.Register(() => tcs.TrySetCanceled());
            return await tcs.Task;
        }

        /// <summary>送出指令但不等回應 —— 給推桿這種高頻更新用。</summary>
        public void Post(string command, object? args = null)
        {
            _ = Task.Run(async () =>
            {
                try { await CallAsync(command, args); }
                catch (Exception) { /* 推桿掉一次更新不值得打擾使用者 */ }
            });
        }

        private async Task ReadLoopAsync(CancellationToken cancellation)
        {
            Exception? failure = null;
            try
            {
                using var reader = new StreamReader(_stream!, Encoding.UTF8);
                while (!cancellation.IsCancellationRequested)
                {
                    var line = await reader.ReadLineAsync(cancellation);
                    if (line is null) break;
                    if (line.Length == 0) continue;

                    // 單一訊息(或它的事件處理常式)出錯,絕不能拖垮整條連線。
                    // I/O 層的錯誤才是真正該斷線的,那會從 ReadLineAsync 拋出。
                    try
                    {
                        Dispatch(line);
                    }
                    catch (Exception ex)
                    {
                        DispatchError?.Invoke(ex);
                    }
                }
            }
            catch (OperationCanceledException)
            {
                // 正常關閉
            }
            catch (Exception ex)
            {
                failure = ex;
            }
            finally
            {
                foreach (var (_, tcs) in _pending)
                    tcs.TrySetException(failure ?? new IOException("引擎連線已中斷。"));
                _pending.Clear();
                if (!cancellation.IsCancellationRequested)
                    Disconnected?.Invoke(failure);
            }
        }

        private void Dispatch(string line)
        {
            JsonDocument document;
            try
            {
                document = JsonDocument.Parse(line);
            }
            catch (JsonException)
            {
                return;
            }

            var root = document.RootElement;

            // JsonElement 一旦 document 被 Dispose 就不能再存取。所有要用的值
            // (字串、clone)都必須在 using 區塊內取出;clone 才能安全帶出去。
            using (document)
            {
                if (root.TryGetProperty("event", out var eventName))
                {
                    var name = eventName.GetString() ?? "";
                    var data = root.TryGetProperty("data", out var d) ? d.Clone() : default;
                    EventReceived?.Invoke(name, data);
                    return;
                }

                if (!root.TryGetProperty("id", out var idElement)
                    || idElement.ValueKind != JsonValueKind.Number
                    || !_pending.TryRemove(idElement.GetInt32(), out var tcs))
                {
                    return;
                }

                var ok = root.TryGetProperty("ok", out var okElement) && okElement.GetBoolean();
                if (ok)
                {
                    var result = root.TryGetProperty("result", out var r) ? r.Clone() : default;
                    tcs.TrySetResult(result);
                }
                else
                {
                    var message = root.TryGetProperty("error", out var e)
                        ? e.GetString() ?? "未知的錯誤" : "未知的錯誤";
                    tcs.TrySetException(new EngineCommandException("引擎", message));
                }
            }
        }

        public async ValueTask DisposeAsync()
        {
            _cancellation?.Cancel();
            try
            {
                if (_readerTask is not null) await _readerTask.WaitAsync(TimeSpan.FromSeconds(2));
            }
            catch (Exception)
            {
                // 關閉時的逾時無所謂
            }

            _writer?.Dispose();
            _stream?.Dispose();
            _tcp?.Dispose();
            _cancellation?.Dispose();
            _writeLock.Dispose();
        }
    }
}
