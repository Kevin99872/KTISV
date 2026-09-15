# TensorFlow 訓練路線

`research/` 原本只有 PyTorch 一條路(`ktisv_research/`)。這份文件講的是
並行的第二條路 `ktisv_tf/`:**同樣的架構、同樣的 ONNX 契約,改用
TensorFlow 訓練**,並且用 KTISV 自己的執行期快取當訓練資料。

相關文件:**MODELS.md**(架構選型)、**DATASETS.md**(語料與授權)、
**TRAINING.md**(PyTorch 版的訓練管線與評估方法)。

---

## 〇、先講清楚這條路的性質

訓練資料是**用 htdemucs 標出來的**(`ktisv_research/label_cache.py`),
也就是知識蒸餾:htdemucs 當老師,訓練一個小得多的學生。三件事要先講明白:

1. **品質上限就是 htdemucs。** 學生不可能超過老師,只能逼近。目標是
   「接近的品質、可即時的成本」,不是「更好的分離」。
2. **標籤帶著老師的假象。** htdemucs 漏掉的和聲、殘留的鼓聲,學生會照學。
3. **權重不可散布。** 標籤衍生自影片網站的版權音源,權重同樣是衍生物。
   自用與驗證管線沒問題;要跟 KTISV 一起發佈,得換成 `DATASETS.md` 裡
   授權明確的資料重訓。管線不用改,只換 `--data` 指到的目錄。

換句話說:這條路解決的是 `DATASETS.md` 標紅的那個缺口 ——「還沒有訓練資料
所以整條管線從沒真的跑過」。現在它跑過了,而且是端到端跑到 ONNX。

---

## 一、環境

TensorFlow 2.9 只支援 Python 3.7–3.10,而 `research/.venv` 是 3.12。
所以 TF 住在**另一個環境** `research/.venv-tf`:

```bash
cd research
uv venv --python 3.10 .venv-tf
VIRTUAL_ENV=.venv-tf uv pip install -r requirements-tf.txt
VIRTUAL_ENV=.venv-tf uv pip install --no-deps \
    --index-url https://download.pytorch.org/whl/cu118 torch==2.3.1+cu118
```

兩個環境的分工:

| 環境 | Python | 用途 |
|---|---|---|
| `.venv` | 3.12 | demucs 產生標籤、PyTorch 訓練、指標計算 |
| `.venv-tf` | 3.10 | TensorFlow 訓練與 ONNX 匯出 |

共用的只有 `ktisv_research.mixing`(合成混音規則)與
`ktisv_research.metrics`(SI-SDR 等指標)—— 這兩個是純 numpy 的,兩邊都能
匯入。**刻意不複製一份到 `ktisv_tf/`**:同一件事有兩份實作,遲早會有一份
悄悄走鐘,而聲音變差是不會報錯的。

### GPU:為什麼要在 TF 環境裡裝 torch

TF 2.9/2.10 是 Windows 上**最後幾版還有原生 GPU 支援**的 TensorFlow
(2.11 起 Windows 只剩 CPU,官方建議改用 WSL2)。代價是它寫死要 CUDA 11.2
世代的 DLL:`cudart64_110.dll`、`cublas64_11.dll`、`cudnn64_8.dll`…

這台機器裝的是 CUDA 13.0 / 13.2,版號對不上,TF 會**默默退回 CPU**
(只在 log 裡留一行 `Could not load dynamic library`)。

正規解法是再裝一套 CUDA Toolkit 11.2 + cuDNN 8.1 到系統上。但那會為了一個
虛擬環境裡的套件動到整台機器,而且 cuDNN 還要 NVIDIA 帳號。

這裡走另一條路:**PyTorch 的 Windows cu118 wheel 自己就捆了整組 CUDA 11
執行期 DLL**,而且就在同一個 venv 裡。把 `torch/lib` 加進 `PATH`,TF 就
找得到了。`ktisv_tf/__init__.py` 的 `enable_cuda11()` 會自動做這件事,
所以平常不必管它。

> 版本很挑:**torch 2.4 起捆的是 cuDNN 9**(`cudnn64_9.dll`),TF 2.9 要的是
> `cudnn64_8.dll`。所以釘 2.3.1。

實測在 RTX 1000 Ada(6 GB,sm_89)上可用 —— TF 2.9 的核心是用 CUDA 11.2
編的,對 Ada 走的是 PTX JIT,第一次跑會多花幾秒編譯,之後正常。

想強制用 CPU:`--cpu`,或設環境變數 `KTISV_TF_NO_CUDA_SHIM=1` 把上面那個
機制整個關掉。

---

## 二、資料:把執行期快取變成訓練集

```bash
cd research
.venv\Scripts\python -m ktisv_research.label_cache
```

它做兩件事:

1. `%LOCALAPPDATA%\KTISV\cache\stems\` 裡**已經跑過 Demucs** 的結果直接搬過來
2. `%LOCALAPPDATA%\KTISV\cache\downloads\` 裡沒有標籤的音檔,跑一次 htdemucs

輸出到 `research/data/ktisv-cache/<歌>/{vocals,accompaniment}.wav`,
這個版面 `ktisv_research.data.load_pair_folders` 與 `ktisv_tf.data.load_tracks`
都讀得懂 —— 訓練端不必知道資料是從哪來的。

常用選項:

| 選項 | 作用 |
|---|---|
| `--max-minutes 12` | 每首最多取幾分鐘,避免單一長檔佔滿資料集 |
| `--device cpu` | 沒有 GPU 時 |
| `--segment 5` | VRAM 不夠時把 demucs 的分段調小 |
| `--force` | 重做已完成的項目 |

已完成的項目會跳過,可以隨時中斷再續。進度記在 `manifest.json`。

> 資料量會直接決定結果。這台機器上的快取產出 **34 首 / 121 分鐘**,
> 以分離模型的標準來說仍然很少(MUSDB18 是 150 首完整歌曲,約 10 小時)。
> 多用 KTISV 聽幾首歌、快取變大,重跑一次這個指令就會變多。

---

## 三、訓練

```bash
cd research
.venv-tf\Scripts\python -m ktisv_tf.train --preset small --steps 8000
```

| 選項 | 預設 | 說明 |
|---|---|---|
| `--data` | `data/ktisv-cache` | 分軌資料夾的根目錄 |
| `--preset` | `small` | `tiny` / `small` / `medium` / `large` |
| `--seconds` | `3.0` | 訓練片段長度(會自動對齊到 2^depth 的整數幀) |
| `--batch-size` | `4` | 6 GB 卡上 `small` + 3 秒片段的安全值 |
| `--steps` | `4000` | 總步數。學習率照這個數做 cosine 衰減 |
| `--independent-prob` | `0.5` | 多少比例的樣本把人聲與伴奏隨機重配 |
| `--spectral-weight` | `1.0` | 幅度譜損失的權重;`0` 表示只用波形 L1 |
| `--val-every` | `250` | 每幾步驗證一次 |
| `--cpu` | — | 強制用 CPU |
| `--resume` | — | 接續某個 `.h5` 權重 |

驗證集是**整首整首**抽走的歌,訓練時一秒都沒出現過。看的數字是
`改善` 這一欄 —— 也就是 `si_sdr_vocals` 減掉「把混音原封不動當人聲交出去」
的基準線。**這個數字必須是正的**,否則模型不如不用。

輸出在 `data/runs/tf/`:`best.h5` / `best.json`(改善最高的那一步)、
`last.h5` / `last.json`、以及 `history.jsonl`。

---

## 四、匯出成 ONNX

```bash
.venv-tf\Scripts\python -m ktisv_tf.export data/runs/tf/best \
    --out data/models/vocals-tf.onnx
```

契約與 PyTorch 版 `ktisv_research.export` **完全相同**:

```
輸入  mixture         float32  (batch, 2, segment_samples)
輸出  vocals          float32  (batch, 2, segment_samples)
      accompaniment   float32  (batch, 2, segment_samples)
```

`samplerate` / `segment_samples` / `channels` / `n_fft` / `hop_length` /
`framework` 寫在 ONNX 的 metadata 裡,不是只寫在這份文件裡 —— 文件會和
檔案走散,metadata 不會。

匯出後會用 ONNX Runtime 實跑一次、和 TensorFlow 逐點比對;誤差超過 1e-3
就直接判定失敗、不產出可用的檔案。這一步不能省:運算子語意的差異不會讓
程式崩潰,只會讓聲音悄悄變差。

片段長度是**固定的**(理由見 `export.py`),整首歌由 C# 端切塊、重疊相加,
最後一塊補零。訓練用 3 秒、匯出用 6 秒沒問題:U-Net 是全卷積的、
STFT / iSTFT 是常數層,所以同一份權重套到任何對齊過的長度都能跑。

但要注意一件反直覺的事:**輸出不是逐點相同的**。U-Net 的感受野比片段還長
(depth 5 把 3 秒的 256 幀壓到 8 幀),換一個長度就換了邊界的補零條件,
而那個差異會傳遍整段、不只是頭尾。實測兩者的 SI-SDR 落在 46–52 dB ——
遠在可聽閾之下,但不是零。`tests/test_tf.py` 驗的就是這個門檻,
不是逐點相等。

> 檔案約 55 MB,其中一半是 STFT / iSTFT 的卷積核常數(兩組
> 2048×2050 的矩陣)。這是「把 STFT 寫進圖裡」的固定成本,PyTorch 版
> 也一樣。換來的是 C# 端不必自己實作 FFT。

---

## 五、試聽:拿真實檔案跑一次

```bash
.venv-tf\Scripts\python -m ktisv_tf.separate 歌.webm --onnx data/models/vocals-tf.onnx
```

輸出 `data/separated/<檔名>/{vocals,accompaniment}.wav`。

**每次訓練完都該真的聽一次。** 驗證分數會告訴你模型有沒有在學,但分數好
聽起來仍然可能很糟 —— SI-SDR 對某些人耳很敏感的假象(音樂噪聲、高頻抖動)
幾乎沒有反應。

`--onnx` 走的是 C# 端會用的那個檔案與那套執行引擎,是最接近實際部署的一次
演練;`--checkpoint data/runs/tf/best` 則直接用 TensorFlow 跑,訓練完不必先
匯出就能聽。兩者輸出應該幾乎一樣,差很多就代表匯出有問題。

整首歌由這個工具自己切塊、以半塊間距滑動、乘 Hann 窗重疊相加 —— 直接切了
再接會在接縫留下喀噠聲。伴奏用「原曲減人聲」得到,所以兩軌相加必定還原
成原曲(實測誤差 6e-8)。

---

## 六、實測結果

環境:RTX 1000 Ada(6 GB)、TF 2.9.3、`small` 預設(516 萬參數)、
3 秒片段、batch 4、8000 步、資料 34 首 / 121 分鐘。

**訓練**

| | |
|---|---|
| 訓練時間 | 30.2 分鐘(約 4.4 step/s) |
| VRAM 尖峰 | 3.68 GB(6 GB 的卡有餘裕) |
| 最佳驗證改善 | **+8.58 dB**(人聲 +10.69,基準線 +2.12) |

驗證曲線:`+3.05 → +5.04 → +6.54 → +7.56 → +8.41 → +8.58 dB`。
中間在 6–8 dB 之間震盪過幾次 —— 資料量只有 29 首訓練歌曲時很正常,
`best.h5` 存的是最高的那一步。

**在一首沒訓練過的歌上,對照 htdemucs 的答案**(前 1.5 分鐘)

| | 模型 | 完全不分離 | 改善 |
|---|---|---|---|
| 人聲 | +6.45 dB | −0.78 dB | **+7.2 dB** |
| 伴奏 | +8.68 dB | +2.35 dB | **+6.3 dB** |

**推論速度**(ONNX Runtime、**CPU**、單執行緒設定)

1.5 分鐘的歌花 7.5 秒 ≈ **12 倍即時**。這個數字才是重點:htdemucs 在
CPU 上要跑好幾分鐘,所以 KTISV 目前只能「先跑完整首再播」。12 倍即時
代表這個模型有機會做成邊播邊分離,而那正是當初訓練它的動機。

> 但別把 +8.58 dB 讀成「品質接近 htdemucs」。標籤本身就是 htdemucs 的輸出,
> 所以這些數字量的是「學生逼近老師的程度」,不是絕對品質。真正的判準還是
> 第五節:自己聽。

---

## 六之二、續訓:只留樂器版(2026-09)

KTISV 最常見的用法是「剝離人聲、只留樂器」,而第一版模型的伴奏裡聽得到一層
淡淡的歌聲。這一輪針對這件事續訓,產物是 `data/models/instrumental-best.onnx`,
安裝到 `%LOCALAPPDATA%\KTISV\models\ktisv-instrumental.onnx`。

```bash
.venv-tf\Scripts\python -m ktisv_tf.train --preset small --steps 20000     --lr 1.5e-4 --warmup 800 --val-every 500 --val-batches 16     --accompaniment-weight 1 --leak-weight 2 --select instrumental     --val-tracks dl-0fdcc8c99fa9963c,dl-8a6a72df155e23c3,dl-d5766c6cc3000724,dl-cc20a296046ccb08,dl-c182856f08ee0ae9     --resume data/runs/tf-small/best.h5 --out data/runs/tf-small-inst
```

**改了什麼**

| | 為什麼 |
|---|---|
| 資料 34 → 39 首(121 → 141 分鐘) | 重跑 `label_cache` 標了新的快取 |
| `--accompaniment-weight` | 伴奏幅度譜 L1。波形域上伴奏損失與人聲損失相同,幅度譜上不同 |
| `--leak-weight` | 只罰「伴奏比真值多出來」的能量 —— 那就是漏進來的人聲 |
| `--warmup` | 續訓時優化器狀態是全新的,第一步用滿學習率會把權重打散(實測從 +8.97 掉到 +7.8 dB) |
| `--select instrumental` | 以「伴奏 SI-SDR 改善 + 0.5 × 人聲殘留 SIR 改善」挑 best |
| 人聲殘留 SIR 指標 | SI-SDR 分不出「漏人聲」與「樂器被削薄」,SIR 只看前者 |
| `--val-tracks` | **續訓一定要固定驗證集**,理由見下 |

**一個差點踩到的坑:驗證集污染。** 資料從 34 首變 39 首,隨機切分就換了一批,
新的 6 首驗證歌裡有 5 首是舊權重訓練時聽過的。續訓繼承了那份權重,在那批歌上
算出來的分數漂亮但毫無意義。所以改成固定用**舊模型當初的驗證歌**,兩個模型都
沒聽過。

**訓練**:20000 步 / 76 分鐘,best 在 step 11500。

**結果**(5 首未見歌曲、各前 90 秒、ONNX Runtime、對照 htdemucs 標籤)

| 模型 | 伴奏 SI-SDR | 人聲殘留 SIR |
|---|---|---|
| 完全不分離 | — | 2.9 dB |
| 第一版 `vocals-tf.onnx` | 10.50 dB | 22.5 dB |
| **只留樂器版 `instrumental-best.onnx`** | **11.53 dB** | **33.9 dB** |

伴奏裡的人聲能量少了 11 dB(約 1/14),伴奏本身也同時變好,不是拿音質換的。

**引擎端的「人聲抑制」後處理**(`engine/ktisv_engine/media/onnx_separator.py`)
在同一批歌上的取捨:

| 抑制強度 | 0% | 25% | 35% | 50% | 80% |
|---|---|---|---|---|---|
| 伴奏 SI-SDR | 11.53 | 10.25 | 9.91 | 9.35 | 8.10 |
| 人聲殘留 SIR | 33.9 | 33.7 | 35.9 | 39.9 | 52.9 |

低強度只有代價沒有收益,所以引擎預設 0%;還聽得到歌聲的人再往上拉到 50% 以上。

> 同樣的提醒:這些分數量的是「逼近 htdemucs」的程度,htdemucs 自己漏掉的和聲
> 學生也會照學。真正的判準仍是自己聽。

---

## 六之三、自動反覆訓練(`ktisv_tf.sweep`)

```bash
.venv-tf\Scripts\python -m ktisv_tf.sweep --queue data/sweep2/queue.json --folder data/sweep2 \
    --seconds 180 --patience 4 --min-gain 0.15 --deadline 19:50
```

每輪:訓練 → 匯出 → 在 5 首未見歌曲的前 180 秒上用 ONNX Runtime 整段評估 →
分數(伴奏 SI-SDR + 0.5 × 人聲殘留 SIR)超過冠軍就換,並自動複製到
`engine/models/ktisv-instrumental.onnx`(打包時帶進 `engine-bin\models\`)。
佇列檔每輪重讀,可以邊跑邊依結果調整下一輪。

**結果**(180 秒評估;第一批原本用 90 秒,冠軍已用 180 秒重評)

| 輪 | 調整 | 分數 | 伴奏 SI-SDR | 殘留 SIR | |
|---|---|---|---|---|---|
| — | 起點(懲罰 4,r6) | 27.82 | 11.38 | 32.87 | |
| s1 | 殘留懲罰 4 → 5 | 28.96 | 12.14 | 33.63 | ★ |
| s2 | 殘留懲罰 5 → 6 | **33.16** | 11.18 | **43.97** | ★ 最終採用 |
| s3 | + 音色/立體聲增強 | 27.25 | 9.92 | 34.67 | 變差 |

第一批(90 秒評估)還試過:低學習率磨細、6 秒訓練片段、少用隨機重配 —— 都沒有進步。
**唯一穩定有效的是加重殘留懲罰**(2→3→4→5→6 每一步都贏)。

s2 逐首看(伴奏 SI-SDR / 殘留 SIR),和 s1 相比每一首的殘留都減少、伴奏都略降:

| | 歌 1 | 歌 2 | 歌 3 | 歌 4 | 歌 5 |
|---|---|---|---|---|---|
| s1 | 12.12 / 25.8 | 11.36 / 36.5 | 12.29 / 42.2 | 12.60 / 35.7 | 12.34 / 28.0 |
| s2 | 11.47 / 31.2 | 10.57 / 38.1 | 11.53 / 58.5 | 11.14 / 60.0 | 11.17 / 32.0 |

所以這是一致的取捨,不是單一首歌灌水 —— 但歌 3、4 碰到 60 dB 的上限,平均值
看起來比實際更漂亮。伴奏 11.18 dB 仍高於第一版模型(10.50)。

**下一步可以試**:懲罰 7 以上(看伴奏何時開始明顯變悶);`--select` 的 SIR 權重
從 0.5 調低,避免一路往犧牲伴奏的方向走;以及最有效但最花時間的 —— 更多訓練資料。

---

## 七、測試

```bash
cd research
.venv-tf\Scripts\python -m tests.test_tf
```

涵蓋:卷積 STFT 往返無損、與 `tf.signal.stft` 的比對、輸出可加性、
片段長度對齊、遮罩值域、**換片段長度的一致性**(訓練 3 秒匯出 6 秒的前提)、
6 GB VRAM 實測、以及過擬合單一樣本。

`tests/test_tf.py` 裡 `import ktisv_tf` 排在 `import tensorflow` **前面**,
那個順序不能調動 —— 掛 CUDA DLL 路徑必須發生在 TF 載入之前,否則 TF 會
抓到系統上 CUDA 13 的 cudart,然後在第一次真的用到 GPU 時才炸。

最後那一項是這類管線最有價值的健全性檢查:「梯度沒接上」「損失算錯軸」
這類錯誤不會讓程式崩潰,只會讓訓練看起來在跑、實際什麼都沒學到。

---

## 八、和 PyTorch 版的差異

| | `ktisv_research/`(PyTorch) | `ktisv_tf/`(TensorFlow) |
|---|---|---|
| 環境 | `.venv`,Python 3.12 | `.venv-tf`,Python 3.10 |
| GPU | 原生 CUDA 13,直接可用 | 要借 torch 的 CUDA 11 DLL(見上) |
| STFT | 訓練用 `torch.stft`,匯出換卷積 | 從頭到尾都是卷積 |
| 張量版面 | channels-first `(B,C,F,T)` | channels-last `(B,T,F,C)` |
| 尺寸不整除時 | `F.interpolate` 硬湊 | 靠 `aligned_length` 避免發生 |
| ONNX 契約 | 相同 | 相同 |

TF 版少了「兩種 STFT 實作要互相驗證」這個負擔,因為它只有一種實作。
代價是環境設定麻煩得多 —— 那一整段 CUDA DLL 的折騰是 PyTorch 版沒有的。

---

## 九、已知限制

* **資料量**。34 首 / 121 分鐘,對分離模型來說很少。真正要拉高品質,
  要嘛把快取養大重跑標註,要嘛照 `DATASETS.md` 去申請 MIR-1K / MUSDB18-HQ。
* **蒸餾的天花板**。見第〇節。
* **TF 2.9 在 Windows 上的 GPU 是撐出來的**,不是官方支援的組合。
  版本一動就會垮,升級前先確認 `tests/test_tf.py` 還過得了。
* ~~C# 端還沒接上~~。已接上:引擎的 `custom` 模式讀
  `%LOCALAPPDATA%\KTISV\models\*.onnx`,前端分離引擎選「自訓模型 · 只留樂器」。
