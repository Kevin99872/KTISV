<div align="center">

<img src="docs/ktisv-logo.svg" width="112" alt="KTISV" />

# KTISV

**YouTube 人聲分離混音台**
一邊唱,一邊讓 Discord 那頭聽見完整的混音。

![平台](https://img.shields.io/badge/Windows-10%20%2F%2011-7C3AED?style=flat-square)
![前端](https://img.shields.io/badge/前端-Avalonia%2012%20%C2%B7%20net10.0-5331D0?style=flat-square)
![引擎](https://img.shields.io/badge/引擎-Python%203.12-22C7EE?style=flat-square&labelColor=5331D0)
![版本](https://img.shields.io/badge/版本-0.1.0-F4557E?style=flat-square&labelColor=5331D0)
![授權](https://img.shields.io/badge/授權-MIT-7C3AED?style=flat-square)

</div>

---

## 這是什麼

把 YouTube 上的歌抓下來、剝掉人聲、和你的麥克風混在一起,然後**分成兩路**送出:
一路給你的耳機(可以即時聽自己),一路灌進虛擬音效卡 —— Discord、OBS 或任何吃麥克風的
軟體,聽到的就是跟你耳機裡一樣的東西。

```
                 ┌─ 分離 ─┐  ┌ 變調 ┐  ┌ 影片 EQ ┐
  YouTube 音源 ──┤        ├─▶│      ├─▶│         ├─▶ 音樂推桿 ┬▶ ×送耳機 ┐
                 └────────┘  └──────┘  └─────────┘            │          │
                                                              └▶ ×送虛擬 ┼▶ 虛擬音效卡
  麥克風 ─▶ 麥克風 EQ ─▶ 回音(可選)─▶ 麥克風推桿 ─┬─▶ ×送虛擬 ─────────┘  (→ Discord)
                                                    │
                                                    └─▶ ×監聽(可勾選)───▶ 耳機
```

兩路走的都是本程式自己開的裝置,不依賴 Windows 的應用程式音訊路由 —— 所以耳返不會被
對方聽到兩份,系統音效也不會漏進去。

## 特色

| | |
|---|---|
| **零延遲人聲分離** | mid/side 相消,即時、無模型、可邊播邊調(實測把置中人聲壓下 27 dB) |
| **高品質離線分離** | 內建 Demucs(CPU 版 PyTorch),整首跑完後有快取,第二次載入直接用 |
| **升 key / 降 key** | 相位聲碼器 + 重新取樣,音高誤差 < 2 音分、**速度誤差 0 ms**,而且只動音樂不動你的聲音 |
| **兩組參數式 EQ** | 影片一組、麥克風一組,頻率 / Q / 段數全可調(最多 24 段),兩端自動變 shelf |
| **麥克風回音** | 卡拉 OK 機那顆旋鈕:回音時間、重複次數、回音量、高頻衰減 |
| **耳返監聽** | 只送耳機、不進虛擬麥克風;UI 即時顯示實測延遲,開 WASAPI 獨佔可到 ~13 ms |
| **YouTube 搜尋與歌詞** | 直接打關鍵字選歌、貼網址、或選本機檔案;歌詞自動抓取與捲動 |
| **一個 App 就好** | 發佈產物自帶 .NET 與 Python 引擎,目標電腦什麼都不用裝 |

介面分四個分頁:**YouTube 搜尋**、**卡拉 OK 設定**、**EQ 調整**、**其他設定**;
首次啟動有設定精靈。介面支援正體中文與英文。

## 專案結構

```
KTISV/                  ← 版本庫根目錄
├─ KTISV/               C# / Avalonia 前端(見 KTISV/README.md,最詳盡的一份文件)
│  ├─ Views/            四個分頁的 XAML + 設定精靈
│  ├─ ViewModels/       MVVM 的 VM 層,所有對引擎的指令都從這裡送出
│  ├─ Services/         引擎行程管理、TCP 客戶端、參數批次送出、驅動安裝、在地化
│  ├─ Controls/         自訂控制項(電平表、數值輸入框)
│  └─ Assets/ktisv.ico  應用程式圖示
├─ engine/              Python 即時音訊引擎 —— 執行期的本體
│  └─ ktisv_engine/
│     ├─ audio/         裝置列舉、串流開關、混音核心(engine.py 是心臟)、延遲校正
│     ├─ dsp/           EQ、變調、回音、延遲線、環形緩衝、分離、降噪、電平表
│     ├─ media/         YouTube 下載、ffmpeg 解碼、歌詞、Demucs / ONNX 分離
│     ├─ ipc.py         JSON-lines 伺服器
│     └─ session.py     指令分派 —— 前端看得到的所有指令都在這裡
├─ driver/              選用的虛擬音效卡驅動(版本庫裡只有腳本,二進位另外抓)
├─ installer/           Inno Setup 打包 → KTISV-Setup.exe
├─ research/            自訓輕量分離模型的訓練與評估。**不參與執行期**
├─ docs/                品牌資產(可縮放的圖示)
└─ publish.ps1          一鍵產出可散布的整包
```

前端啟動時自己把引擎行程拉起來、從 stdout 讀出埠號與 token 再連上 localhost TCP,
關閉時一併收掉 —— 你不需要手動啟動引擎。

> `research/` 刻意與 `engine/` 分開:訓練相依(PyTorch GPU 版約 2 GB)不該進入要發佈的
> 即時引擎。發佈產物完全不含 `research/`。

## 快速上手(使用者)

1. **裝虛擬音效卡** —— 要送給 Discord 的話必要。安裝 [VB-CABLE](https://vb-audio.com/Cable/)
   後重開機。**裝完務必把 Windows 預設播放裝置改回耳機**(設定 → 系統 → 音效 → 輸出),
   否則所有系統聲音都會灌進那條線路。
2. **裝 KTISV** —— 執行安裝檔,或解壓縮整包後雙擊 `KTISV.exe`。
   `KTISV.exe` 和旁邊的 `engine-bin\` **必須放在一起**,少了它程式開得起來但沒有聲音。
3. **接線** —— KTISV 的「虛擬麥克風」選 `CABLE Input`,Discord 的輸入裝置選 `CABLE Output`;
   耳機和麥克風選你真正的裝置。記得在 Discord 語音設定裡**關掉回音消除與噪音抑制**,
   否則音樂會被當成噪音削得斷斷續續。

完整的使用流程、每個旋鈕的作用與取捨,見 **[KTISV/README.md](KTISV/README.md)**;
給朋友看的白話版在 **[DC.md](DC.md)**。

## 開發

```bash
# 1. Python 引擎(建議用 uv,會自動抓對的 Python 版本並建立隔離環境)
cd engine && uv sync

# 2. 驗證引擎環境
uv run python -m ktisv_engine --selftest

# 3. 跑前端(會自動拉起引擎)
dotnet run --project KTISV/KTISV.csproj
```

環境釘在 Python 3.12/3.13(PyTorch 還沒有完整的 cp314 wheel),`engine/.venv` 會被前端
自動偵測到;放別處就設 `KTISV_PYTHON`。`imageio-ffmpeg` 自帶 ffmpeg,不需要另外安裝。

### 建置與發佈

```powershell
powershell -ExecutionPolicy Bypass -File publish.ps1              # 整包(exe + engine-bin\)
powershell -ExecutionPolicy Bypass -File installer\build-installer.ps1   # 安裝檔
```

`publish.ps1` 先用 PyInstaller 打包引擎,再把前端發佈成自包含的 `KTISV.exe`,並自動把
引擎帶進 `engine-bin\`。預設是**資料夾部署**而非單一檔案 —— 自解壓縮的單檔正是加殼器的
行為特徵,防毒誤判率高,而既然用安裝檔散布也沒好處(真的需要時加 `-SingleFile`)。

### 測試

```bash
cd engine
python -m tests.test_dsp        # 離線 DSP —— 不開任何裝置,隨時可跑
python -m tests.test_ipc        # 端對端:啟動真的引擎行程,走一輪所有指令
python -m tests.test_devices    # 實體裝置:真的開 WASAPI 串流(全程靜音)
python -m tests.test_loopback   # VB-CABLE 回送:驗證對端到底收到什麼
python -m tests.test_latency    # 耳返延遲:拆解各段,掃描 block 大小與獨佔模式
```

前端用編譯期繫結(`AvaloniaUseCompiledBindingsByDefault`),繫結路徑打錯會在
`dotnet build` 就報錯 —— 建置通過本身就是一層驗證。

## 文件

| 文件 | 內容 |
|---|---|
| **[KTISV/README.md](KTISV/README.md)** | 主要文件:完整安裝、每個功能的原理與取捨、疑難排解 |
| [DC.md](DC.md) | 給使用者的白話說明(發佈時貼在社群用) |
| [NOTICE.md](NOTICE.md) | 第三方元件與授權,以及散布時的義務(**Demucs 權重那節務必先讀**) |
| [SIGNING.md](SIGNING.md) | 程式碼簽章 |
| [driver/README.md](driver/README.md) | 為什麼一定要驅動、各方案比較、自帶驅動的作法 |
| [research/MODELS.md](research/MODELS.md) | 現成分離模型比較與選型理由 |
| [research/DATASETS.md](research/DATASETS.md) | 語料調查與授權能不能商用 |
| [research/TRAINING.md](research/TRAINING.md) | 訓練管線、損失函數、評估指標與實測結果 |

## 圖示與配色

<img src="docs/ktisv-logo.svg" width="72" align="left" hspace="16" vspace="4" alt="" />

圖示是一組電平柱:**左邊那根紅色是你的麥克風,右邊那根青色是送進虛擬音效卡、給對面聽的
那一路**,中間三根白/淡紫的是混音本身。左右對稱,因為兩路輸出拿到的是同一份混音 ——
整個程式在做的事,就是這張圖。

`KTISV/Assets/ktisv.ico` 是實際使用的 16×16 應用程式圖示;
[`docs/ktisv-logo.svg`](docs/ktisv-logo.svg) 是依它重畫的可縮放版本,給文件與網頁用。

| | 色碼 | 用途 |
|---|---|---|
| 🟣 | `#5331D0` → `#7C3AED` | 底色漸層(左上 → 右下),也是文件與 UI 的主色 |
| ⚪ | `#FFFFFF` / `#DCCFF9` | 混音的三根柱子 |
| 🔴 | `#F4557E` | 麥克風那一路 |
| 🔵 | `#22C7EE` | 送往虛擬音效卡(Discord)那一路 |

## 授權

KTISV 本身以 [MIT](KTISV/LICENSE) 釋出。第三方元件的授權與散布義務整理在
[NOTICE.md](NOTICE.md) —— 尤其 **Demucs 的預訓練權重授權不明確**,目前刻意不隨安裝檔
散布,而是由使用者第一次分離時自行下載。

> 下載與播放 YouTube 內容請遵守 YouTube 服務條款與當地著作權法規。這個工具本身不規避
> 任何存取限制,使用者需自行確認手上的內容有合法的使用權。
