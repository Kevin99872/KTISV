using System.ComponentModel;
using Avalonia.Controls;
using Avalonia.Threading;
using KTISV.ViewModels;

namespace KTISV.Views
{
    public partial class YouTubeTab : UserControl
    {
        private LyricsViewModel? _lyrics;

        public YouTubeTab()
        {
            InitializeComponent();
            DataContextChanged += OnDataContextChanged;
        }

        private void OnDataContextChanged(object? sender, System.EventArgs e)
        {
            if (_lyrics is not null)
                _lyrics.PropertyChanged -= OnLyricsPropertyChanged;

            _lyrics = (DataContext as MainWindowViewModel)?.Lyrics;

            if (_lyrics is not null)
                _lyrics.PropertyChanged += OnLyricsPropertyChanged;
        }

        /// <summary>
        /// 讓正在唱的那一行保持在畫面內。
        ///
        /// 捲動只能在 View 做 —— ViewModel 不該知道 ListBox 的存在。
        /// 使用者關掉「自動捲動」時就不干預,讓他自己翻。
        /// </summary>
        private void OnLyricsPropertyChanged(object? sender, PropertyChangedEventArgs e)
        {
            if (e.PropertyName != nameof(LyricsViewModel.CurrentLine)) return;
            if (_lyrics is null || !_lyrics.AutoScroll) return;

            var line = _lyrics.CurrentLine;
            if (line is null) return;

            // 排到 UI 佇列尾端:此刻項目容器可能還沒產生出來
            Dispatcher.UIThread.Post(
                () => LyricsList.ScrollIntoView(line),
                DispatcherPriority.Background);
        }
    }
}
