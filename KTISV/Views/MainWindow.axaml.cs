using System.Linq;
using System.Threading.Tasks;
using Avalonia.Controls;
using Avalonia.Platform.Storage;
using KTISV.ViewModels;

namespace KTISV.Views
{
    public partial class MainWindow : Window
    {
        public MainWindow()
        {
            InitializeComponent();

            // 拖動音軌時要先停止引擎回填位置,否則游標會被拉回去
            SeekBar.SeekStarted += (_, _) => ViewModel?.BeginSeek();
            SeekBar.SeekCompleted += (_, _) => ViewModel?.EndSeek();
        }

        private MainWindowViewModel? ViewModel => DataContext as MainWindowViewModel;

        protected override void OnDataContextChanged(System.EventArgs e)
        {
            base.OnDataContextChanged(e);
            if (ViewModel is { } viewModel)
                viewModel.PickAudioFileAsync = PickAudioFileAsync;
        }

        private async Task<string?> PickAudioFileAsync()
        {
            var files = await StorageProvider.OpenFilePickerAsync(new FilePickerOpenOptions
            {
                Title = "選擇音訊或影片檔",
                AllowMultiple = false,
                FileTypeFilter =
                [
                    new FilePickerFileType("音訊 / 影片")
                    {
                        Patterns = ["*.mp3", "*.m4a", "*.wav", "*.flac", "*.ogg", "*.opus",
                                    "*.aac", "*.wma", "*.mp4", "*.mkv", "*.webm", "*.mov"],
                    },
                    new FilePickerFileType("所有檔案") { Patterns = ["*"] },
                ],
            });

            return files.FirstOrDefault()?.TryGetLocalPath();
        }
    }
}
