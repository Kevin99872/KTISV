# 程式碼簽章與憑證

散布給其他人時,憑證有兩個獨立的問題。**它們的性質與解法完全不同**,
不要混在一起看。

---

## 問題 1:SmartScreen 警告「發行者:不明」

### 症狀

使用者在別台電腦執行安裝檔時看到:

> **Windows 已保護您的電腦**
> Microsoft Defender SmartScreen 已防止某個無法辨識的應用程式啟動。
> 發行者:不明

### 原因

程式和安裝檔**沒有數位簽章**。目前狀態:

```
KTISV.exe          NotSigned
KTISV-Setup.exe    NotSigned
```

### 解法

**只能買一張程式碼簽章憑證。沒有任何程式技巧可以繞過。**

任何宣稱能「關掉 SmartScreen」的做法,本質都是要求使用者降低自己電腦的
安全設定 —— 那不是解法,而是把問題轉嫁給使用者。

| 憑證類型 | 大約年費 | 效果 |
|---|---|---|
| **OV**(組織驗證) | US$200–400 | 消除「發行者:不明」,但 SmartScreen 仍需累積下載信譽,初期可能還是有警告 |
| **EV**(延伸驗證) | US$300–600 | **立即取得 SmartScreen 信譽**,是唯一能馬上消除警告的選項。需要硬體金鑰(HSM / USB token) |
| 自簽 | 免費 | **對別人的電腦沒有用**。除非對方主動信任你的根憑證,而那是要求別人降低系統安全性 |

常見的 CA:DigiCert、Sectigo、SSL.com、GlobalSign。
個人開發者也能申請,但需要身分驗證文件。

### 拿到憑證後怎麼用

```bash
cd installer

# .pfx 檔(OV 憑證常見)
.\sign.ps1 -Path ..\KTISV\bin\Release\net10.0\win-x64\publish\KTISV.exe -PfxPath cert.pfx

# 憑證存放區裡的憑證(EV 用硬體金鑰時)
.\sign.ps1 -Path Output\KTISV-Setup.exe -Thumbprint A1B2C3...

# 只檢查目前狀態
.\sign.ps1 -Path Output\KTISV-Setup.exe -Verify
```

**兩個都要簽**:`KTISV.exe` 和 `KTISV-Setup.exe`。使用者先碰到安裝檔,
但安裝完執行主程式時 SmartScreen 會再檢查一次。

`sign.ps1` 預設會加時間戳記(`-tr`)。**這一步不能省** —— 沒有時間戳記的話,
憑證到期後連已經簽好的舊版程式都會變成無效簽章。

密碼不要寫進腳本或 CI 設定。省略 `-PfxPassword` 會互動詢問。

---

## 問題 2:驅動安裝會動到系統的憑證信任(已修正)

### 原本的問題

`driver\install-driver.ps1` 舊版做了這件事:

```powershell
certutil -addstore -f Root $cert.FullName    # ← 已移除
```

把驅動的憑證加入**「受信任的根憑證授權單位」**。這個存放區的意義是
「**這是一個憑證授權單位,它簽署的任何東西我都信任**」—— 包含網站的
TLS 憑證、任何軟體、任何驅動。

為了裝一個音訊驅動而動這個存放區,範圍大得離譜。防毒軟體與有警覺的
使用者會標記它,是完全合理的。

### 現在的行為

| 存放區 | 預設 | 意義 |
|---|---|---|
| `TrustedPublisher` | ✅ 會寫入 | 「安裝這個發行者的驅動時不用再問」。範圍僅限驅動/軟體發行者,是驅動安裝真正需要的 |
| `Root` | ❌ **不寫入** | 需要 `-TrustRootCertificate` 明確指定,且會印出多行警告 |

移除程式時會把兩個存放區裡的憑證都清掉,不會留下被降低的信任基準。

### 如果驅動因憑證鏈無法驗證而裝不起來

那代表該驅動的憑證不是由公開受信任的 CA 簽發。此時**建議改用已正式簽章
的方案**(VB-CABLE / VoiceMeeter),而不是降低系統的信任基準。

真的要繼續的話:

```bash
.\install-driver.ps1 -TrustRootCertificate
```

腳本會先把後果講清楚再執行。這應該是使用者自己的決定,不是安裝程式偷偷做掉。

---

## 簡短建議

**如果只是自己和朋友用**:不必買憑證。教他們在 SmartScreen 警告上按
「其他資訊 → 仍要執行」即可。

**如果要公開散布**:EV 憑證是唯一能立即消除警告的選項。OV 便宜一些,
但初期仍會有警告,要等下載量累積信譽。

**任何情況下都不要**:叫使用者關閉 SmartScreen、或安裝你的自簽根憑證。
那是把安全成本轉嫁給使用者。
