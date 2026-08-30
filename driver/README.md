# 虛擬音訊驅動

這個資料夾放**選用的**虛擬音訊驅動。KTISV 沒有它也能跑(耳機那一路完全正常),
只是無法把混音送進 Discord —— 那需要一個虛擬麥克風。

**版本庫裡這個資料夾只有腳本與這份說明。** 驅動是二進位檔,不屬於原始碼,
用 `fetch-driver.ps1` 抓下來:

```powershell
powershell -ExecutionPolicy Bypass -File driver\fetch-driver.ps1
```

抓完之後 `installer\build-installer.ps1` 就會把驅動一起打包進安裝檔,
App 的設定精靈也會出現「安裝內建驅動」按鈕(靠 `.inf` 是否存在判斷)。
沒抓的話安裝檔照樣編得出來,只是精靈退回引導使用者自己裝 VB-CABLE。

---

## 先搞懂:為什麼一定要驅動

Windows 上「其他程式能選的麥克風」是在**驅動層**列舉出來的。任何一般應用程式
(包括 KTISV)都沒辦法在執行時憑空生出一個讓 Discord 看得到的錄音裝置。

所以每一個方案 —— VB-CABLE、VoiceMeeter、VAC、NVIDIA Broadcast —— 全都附帶一個
核心模式驅動。沒有「不裝驅動就有虛擬麥克風」這種選項。

---

## 選項比較

| 方案 | 免費 | 可散布 | 簽章可靠度 | 備註 |
|---|---|---|---|---|
| **VB-CABLE** | 個人免費 | ✗ 商業散布需授權 | ✅ 正式簽章 | 最省事,推薦一般使用者 |
| **VoiceMeeter** | 捐贈制 | ✗ 同上 | ✅ | 功能多但較重 |
| **VAC** | ✗ 付費 | ✗ 需 OEM 授權 | ✅ 最穩 | 未授權版滿 30 分鐘會插入語音提醒 |
| **Virtual Audio Driver** | ✅ | ✅ MIT | ⚠️ 見下方 | 本資料夾的腳本針對它設計 |

### Virtual Audio Driver 的簽章限制(重要)

[VirtualDrivers/Virtual-Audio-Driver](https://github.com/VirtualDrivers/Virtual-Audio-Driver)
是 MIT 授權、可自由散布,這是它最大的優勢。但要誠實說明:

它用的是 SignPath Foundation 憑證,**不等於** Microsoft 的核心驅動認證
(attestation / WHQL)。這不是推測 —— 25.7.14 的 release notes 原文就寫著
「Free code signing on Windows provided by SignPath.io, certificate by
SignPath Foundation」,而該專案 README 至今仍要求 `bcdedit /set testsigning on`。

在開啟 Secure Boot 的系統上——尤其 Windows 11 24H2——有使用者回報裝完在
裝置管理員出現黃色驚嘆號、或裝置根本沒出現(見該專案的 Issue #1、#15)。

**2026/4 之後更嚴格。** Microsoft 的
[Windows Driver Policy](https://support.microsoft.com/en-us/windows/hardware/drivers/the-windows-driver-policy)
自 2026 年 4 月起,核心模式驅動要預設載入必須經過 **WHCP 認證**;舊的
cross-signed 程式失去預設信任,系統累積 250 小時執行時間 + 3 次重開後
會從記錄模式轉為強制封鎖。影響 Windows 11 24H2 / 25H2 / 26H1 與 Server 2025。

而這個專案的最新 release 是 **2025-07-14**,從未針對該政策調整過。
所以在較新的機器上,「裝得起來但載不動」是要預期的結果,不是意外。

若遇到這種情況,你有兩個選擇:

1. **改用已認證的方案**(VB-CABLE 最省事)——建議大多數人這樣做
2. **啟用測試簽章模式**——`install-driver.ps1 -TestSigning`。代價是系統安全性
   降低、桌面右下角出現浮水印、且需要重新開機。**不建議一般使用者採用。**

KTISV 的設定精靈會在 `pnputil` 回報簽章被拒(結束碼 7)或裝置狀態異常
(結束碼 9)時說明原因,並把上述兩個選項攤出來,不會讓使用者卡在原地。

---

## 安裝 Virtual Audio Driver

### 1. 下載檔案

```powershell
powershell -ExecutionPolicy Bypass -File fetch-driver.ps1
```

會抓兩個來源並放進這個資料夾:

```
driver/
├─ VirtualAudioDriver.inf     驅動描述檔      ← VirtualDrivers/Virtual-Audio-Driver
├─ VirtualAudioDriver.sys     驅動本體
├─ virtualaudiodriver.cat     簽章目錄
└─ nefconw.exe                裝置節點建立工具 ← nefarius/nefcon
```

`nefconw.exe` **是必要的**:虛擬裝置沒有實體硬體可以觸發安裝,
`pnputil` 把驅動放進存放區之後不會有任何裝置出現,必須手動建立 root 列舉的裝置節點。
nefcon 同時發佈 x86 / x64 / arm64,`fetch-driver.ps1` 會挑 x64 ——
挑錯架構的話會在建立裝置節點那一步無聲失敗。

> **25.7.14 沒有附 `.cer`。** 安裝腳本會跳過憑證匯入那一步(它有處理這個情況),
> 代價是安裝時會跳出「您要安裝這個裝置軟體嗎?」的系統對話框,無法靜默。
> 要靜默就得從 `.cat` 取出簽章憑證再匯入 `TrustedPublisher` —— 但那只是
> 讓對話框不跳,**不會**讓一個未經 WHCP 認證的驅動變成載得起來。

要釘住特定版本:`fetch-driver.ps1 -DriverVersion 25.7.14`;重抓加 `-Force`。

### 2. 執行安裝

以**系統管理員**身分:

```powershell
powershell -ExecutionPolicy Bypass -File install-driver.ps1
```

腳本會依序:匯入憑證 → `pnputil` 加入驅動存放區 → `nefconw` 建立裝置節點 → 驗證裝置狀態。

硬體 ID 預設從 `.inf` 自動解析(不同版本的 ID 可能不同,所以不寫死)。
要覆寫的話用 `-HardwareId "Root\XXX"`。

### 3. 確認

重新啟動 KTISV,它會自動偵測到虛擬裝置並選好。Discord 的輸入裝置選「Virtual Mic」。

---

## 其他指令

```powershell
# 只解析 INF、印出硬體 ID(不需管理員,不做任何變更 —— 除錯用)
powershell -ExecutionPolicy Bypass -File install-driver.ps1 -ParseOnly

# 移除驅動
powershell -ExecutionPolicy Bypass -File install-driver.ps1 -Uninstall

# 簽章被拒時的退路(有安全性代價,需重開機)
powershell -ExecutionPolicy Bypass -File install-driver.ps1 -TestSigning
```

## 結束碼

| 碼 | 意義 |
|---|---|
| 0 | 成功 |
| 2 | 沒有系統管理員權限 |
| 3 | 找不到驅動資料夾 |
| 4 | 資料夾裡沒有 `.inf` |
| 5 | 無法從 INF 解析硬體 ID |
| 6 | 找不到 `nefconw.exe` |
| 7 | `pnputil` 失敗(常見原因是簽章被拒) |
| 8 | 建立裝置節點失敗 |
| 9 | 裝置已建立但狀態異常(通常是簽章未被接受) |

KTISV 的安裝精靈會依這些碼給出對應的說明。

---

## 給改用其他驅動的人

這裡的腳本是針對 Virtual Audio Driver 寫的。若你用 VB-CABLE / VoiceMeeter / VAC,
它們**自己就有安裝程式**,直接跑那個即可,不需要這個資料夾。
KTISV 一樣認得它們並會自動選好裝置。
