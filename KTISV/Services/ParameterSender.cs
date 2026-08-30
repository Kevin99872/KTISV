using System;
using System.Collections.Generic;
using System.Threading;
using Avalonia.Threading;

namespace KTISV.Services
{
    /// <summary>
    /// 把推桿的高頻更新收斂成固定節奏的批次送出。
    /// 拖動一次推桿會產生上百個變更事件,逐一送出既浪費也會讓事件佇列塞車;
    /// 這裡只保留每個參數的最新值,每 40 ms 送一次。
    /// </summary>
    public sealed class ParameterSender : IDisposable
    {
        private readonly EngineClient _client;
        private readonly DispatcherTimer _timer;
        private readonly Dictionary<string, double> _pendingGainsDb = [];
        private readonly Dictionary<string, double[]> _pendingEq = [];
        private readonly Lock _lock = new();

        public ParameterSender(EngineClient client)
        {
            _client = client;
            _timer = new DispatcherTimer(TimeSpan.FromMilliseconds(40),
                                         DispatcherPriority.Background, (_, _) => Flush());
            _timer.Start();
        }

        public void SetGainDb(string name, double db)
        {
            lock (_lock) _pendingGainsDb[name] = db;
        }

        public void SetEqGains(string target, double[] gains)
        {
            lock (_lock) _pendingEq[target] = gains;
        }

        /// <summary>送給 Discord 那一路的對時(毫秒,可正可負)。跟推桿一樣需要收斂。</summary>
        public void SetVcSyncMs(double ms)
        {
            lock (_lock) _pendingDelayMs = ms;
        }

        private double? _pendingDelayMs;

        /// <summary>立刻送出目前累積的變更(例如放開推桿或關閉視窗時)。</summary>
        public void Flush()
        {
            Dictionary<string, double>? gains = null;
            Dictionary<string, double[]>? eq = null;
            double? delayMs;

            lock (_lock)
            {
                delayMs = _pendingDelayMs;
                _pendingDelayMs = null;
                if (_pendingGainsDb.Count > 0)
                {
                    gains = new Dictionary<string, double>(_pendingGainsDb);
                    _pendingGainsDb.Clear();
                }
                if (_pendingEq.Count > 0)
                {
                    eq = new Dictionary<string, double[]>(_pendingEq);
                    _pendingEq.Clear();
                }
            }

            if (gains is not null)
                _client.Post("set_gains", new { db = gains });

            if (eq is not null)
                foreach (var (target, values) in eq)
                    _client.Post("eq_set_all", new { target, gains = values });

            if (delayMs is { } ms)
                _client.Post("set_vc_sync", new { ms });
        }

        public void Dispose()
        {
            _timer.Stop();
            Flush();
        }
    }
}
