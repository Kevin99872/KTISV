using Avalonia;
using Avalonia.Controls.ApplicationLifetimes;
using Avalonia.Markup.Xaml;
using Avalonia.Threading;
using KTISV.ViewModels;
using KTISV.Views;

namespace KTISV
{
    public partial class App : Application
    {
        private MainWindowViewModel? _viewModel;

        public override void Initialize()
        {
            AvaloniaXamlLoader.Load(this);
        }

        public override void OnFrameworkInitializationCompleted()
        {
            if (ApplicationLifetime is IClassicDesktopStyleApplicationLifetime desktop)
            {
                _viewModel = new MainWindowViewModel();
                desktop.MainWindow = new MainWindow { DataContext = _viewModel };

                // 視窗顯示之後再啟動 Python 引擎,才不會卡住開場畫面
                Dispatcher.UIThread.Post(async () => await _viewModel.InitializeAsync(),
                                         DispatcherPriority.Background);

                desktop.ShutdownRequested += async (_, e) =>
                {
                    if (_viewModel is null) return;
                    var viewModel = _viewModel;
                    _viewModel = null;
                    e.Cancel = true;                       // 先讓引擎行程收尾
                    await viewModel.DisposeAsync();
                    desktop.Shutdown();
                };
            }

            base.OnFrameworkInitializationCompleted();
        }
    }
}
