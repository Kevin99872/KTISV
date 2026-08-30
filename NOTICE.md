# 第三方元件與授權

KTISV 本身以 MIT 授權釋出(見 `KTISV/LICENSE`)。本文件列出所有第三方元件
及其授權,以及散布時需要注意的義務。

> 授權資訊取自各套件的 metadata,整理於 2026-07。**這不是法律意見。**
> 版本更新時授權可能改變,商業使用前請自行確認。

---

## ⚠️ 需要特別注意的元件

以下幾項不是寬鬆授權,**散布時有額外義務**。是否受影響取決於你怎麼散布。

### Demucs 的**預訓練權重** — 授權不明確

> 2026-08 新增。Demucs 改為內建之後(見 `engine/pyproject.toml`),
> 這一項變成散布前最需要釐清的事。

要分清楚兩件事:

| 項目 | 授權 | 說明 |
|---|---|---|
| Demucs **程式碼** | MIT | 明確,可自由散布 |
| Demucs **預訓練權重** | **不明確** | 見下方 |

目前權重**不隨安裝檔散布** —— demucs 在第一次分離時自己從 Hugging Face
下載(`adefossez/HTDemucs` 的 `955717e8.safetensors`,80.1 MB),存到使用者的
`~/.cache/huggingface/`。**那是使用者自己取得權重,不是我們轉散布。**

這個設計是刻意的。查證結果:

* facebookresearch/demucs 的 [Issue #327](https://github.com/facebookresearch/demucs/issues/327)
  問的正是「能不能把預訓練模型放進商業產品散布」—— **維護者從未給出明確答覆**,
  而該 repo 已於 2025-01 封存,不再受理新問題。
* 公開說法互相矛盾:有來源說權重比照程式碼為 MIT,也有來源說是
  **CC-BY-NC 4.0**(禁止商業使用,源自訓練資料的限制)。

**在釐清之前,不要把權重打包進安裝檔。** 代價只是使用者第一次分離需要連網;
換來的是不必去賭一個連原作者都沒回答過的授權問題。

若之後決定內附權重(讓程式完全離線可用),請先取得該權重授權的明確依據。

### PyTorch — BSD-3-Clause

> 2026-08 新增。內建 Demucs 一併帶進來的,打包後約 454 MB。

寬鬆授權,可自由散布,只需保留版權聲明與免責聲明。刻意用 **CPU 版**:
CUDA 版是 2753 MB,會讓安裝檔多出 2.3 GB,而分離是離線工作且結果有快取。

### FFmpeg — GPL v3

透過 `imageio-ffmpeg` 內附。實際建置參數含 `--enable-gpl --enable-version3`,
所以是 **GPL v3**,不是 LGPL。

| 你的做法 | 影響 |
|---|---|
| **只開源程式碼**,不附 binary | ✅ 無影響 |
| 使用者自行安裝 ffmpeg | ✅ 無影響 |
| **把 ffmpeg binary 打包進安裝檔散布** | ⚠️ 你在散布 GPL 軟體 |

程式碼層面,KTISV 是用 subprocess 呼叫 ffmpeg(獨立行程),一般認為不構成
衍生作品,不會讓 MIT 程式碼被傳染。但**散布 binary 本身**要遵守 GPL:
附上授權全文、提供原始碼取得方式、不得附加額外限制。

**規避方式:** 改用 LGPL 建置的 ffmpeg,或不內附、由使用者自行安裝。

### lameenc — LGPL-3.0-or-later

Demucs 的相依(MP3 輸出用)。LGPL 允許動態連結而不傳染,但**散布時**需要
提供該元件的原始碼或取得方式,並允許使用者替換該元件。

若你的散布方式不含 Python 環境(例如只開源程式碼),則不受影響。

### tqdm — MPL-2.0 AND MIT

MPL 是檔案層級的弱 copyleft:只有**修改過的 MPL 檔案**需要開源,不影響
你自己的程式碼。實務上只要不改 tqdm 的原始碼就沒事。

---

## ✅ 寬鬆授權(可自由使用與散布)

### 前端(C# / .NET)

| 元件 | 授權 |
|---|---|
| Avalonia | MIT |
| Avalonia.Desktop / Themes.Fluent / Fonts.Inter | MIT |
| CommunityToolkit.Mvvm | MIT |
| .NET Runtime | MIT |

### 音訊引擎(Python)

| 元件 | 版本 | 授權 |
|---|---|---|
| numpy | 2.5.1 | BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0 |
| scipy | 1.18.0 | BSD |
| sounddevice | 0.5.5 | MIT |
| PortAudio(sounddevice 內附) | — | MIT |
| yt-dlp | 2026.7.4 | Unlicense |
| imageio-ffmpeg(封裝本身) | 0.6.0 | BSD-2-Clause |
| cffi | 2.1.0 | MIT-0 |

> `imageio-ffmpeg` 這個 **Python 封裝**是 BSD,但它**內附的 ffmpeg binary**
> 是 GPL v3 —— 兩者要分開看。

### 研究與訓練(Python)

| 元件 | 版本 | 授權 |
|---|---|---|
| PyTorch | 2.13.0+cu130 | Apache-2.0 AND BSD-2/3-Clause AND BSL-1.0 AND MIT |
| torchaudio | 2.11.0+cu130 | BSD |
| Demucs(程式碼) | 4.1.0 | MIT |
| einops | 0.8.2 | MIT |
| julius | 0.2.8 | MIT |
| soundfile | 0.14.0 | BSD-3-Clause |
| libsndfile(soundfile 內附) | — | LGPL-2.1 |
| requests | 2.34.2 | Apache-2.0 |
| fast-bss-eval | — | MIT |

---

## 🔶 不隨專案散布,但使用者會裝的東西

這些**不在**專案裡,由使用者自行安裝。列出來是為了說明為何不能內附。

### 虛擬音訊驅動

| 方案 | 授權 | 能否內附散布 |
|---|---|---|
| **VB-CABLE** | 捐贈制 | ❌ 商業散布需 VB-Audio 授權 |
| **VoiceMeeter** | 捐贈制 | ❌ 同上 |
| **VAC** | 商業授權 | ❌ 需 OEM 授權 |
| **Virtual-Audio-Driver** | MIT | ✅ 可以,但簽章在部分系統上有問題 |

KTISV 採「偵測 + 引導」設計:不內附任何驅動,由使用者自行安裝。
這個設計正好避開上述所有授權問題。

`driver/` 資料夾預設是空的;若你放入 Virtual-Audio-Driver 的檔案並一起散布,
記得附上其 MIT 授權與著作權聲明。

---

## 🔴 模型權重與訓練資料

**這一節與程式碼授權是分開的,而且經常被忽略。**

### 預訓練模型

| 模型 | 程式碼授權 | 權重授權 |
|---|---|---|
| Demucs htdemucs | MIT | **需自行向專案確認** |

程式碼是 MIT **不代表**權重也是 MIT。商業使用前務必確認權重的授權條款。

### 訓練資料的傳染性

若你未來自行訓練模型並散布權重,**訓練資料的授權會影響權重的可用範圍**:

| 資料集 | 授權性質 | 對產出權重的影響 |
|---|---|---|
| MUSDB18 / MUSDB18-HQ | 研究用途 | 產出的權重能否商用有疑義 |
| MIR-1K | 學術用途 | 同上 |
| MoisesDB | 需註冊,依其條款 | 依條款 |
| 自行錄製 / 已授權素材 | 你擁有權利 | ✅ 最乾淨 |

要做商業可用的模型,**必須從資料來源開始規劃**,不能等訓練完再處理。

### 使用者自行下載的音訊

KTISV 可以從影片網站下載音訊供本機使用。這些檔案:

- 是版權內容,**不可再散布**
- 其分離結果是衍生物,**同樣不可再散布**
- 存放於 `%LOCALAPPDATA%\KTISV\cache\`,不在專案內
- `research/.gitignore` 已設定排除所有音訊檔,避免誤推上公開 repo

使用者需自行遵守所在地的著作權法規與各平台服務條款。

---

## 散布方式建議

| 散布方式 | 需要處理的義務 |
|---|---|
| **只開源程式碼** | 保留各元件的著作權聲明即可(本文件) |
| 開源 + 提供安裝檔 | 加上 FFmpeg 的 GPL 義務、lameenc 的 LGPL 義務 |
| 開源模型權重 | 追加訓練資料的授權檢視 |
| 商業使用 | 上述全部重新確認,特別是模型權重 |

**最單純的路線是只開源程式碼、不內附任何 binary**,讓使用者自行安裝
ffmpeg 與虛擬音訊驅動。KTISV 現有的偵測與引導設計已經支援這個模式。
