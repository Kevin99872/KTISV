<#
.SYNOPSIS
    安裝 Virtual Audio Driver(虛擬喇叭 + 虛擬麥克風)。

.DESCRIPTION
    這是一個核心模式音訊驅動,安裝需要系統管理員權限。步驟:
      1. 把驅動的簽章憑證匯入信任的根 / 發行者存放區
      2. 用 pnputil 把驅動套件加入驅動程式存放區
      3. 用 nefconw 建立 root 列舉的裝置節點(這類虛擬裝置沒有實體硬體可以觸發安裝)
      4. 驗證裝置是否真的出現

    硬體 ID 預設從 .inf 檔解析,不寫死 —— 不同版本的驅動 ID 可能不同。

.PARAMETER DriverPath
    放驅動檔案的資料夾。預設是本腳本所在目錄。

.PARAMETER HardwareId
    覆寫硬體 ID。省略時從 INF 解析。

.PARAMETER TestSigning
    啟用 Windows 測試簽章模式。這會降低系統安全性、桌面右下角出現浮水印,
    而且需要重新開機。只有在正常安裝因簽章被拒時才用,且必須是使用者明確選擇。

.PARAMETER Uninstall
    改為移除驅動。

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File install-driver.ps1
#>

[CmdletBinding()]
param(
    # 預設值不能直接寫 $PSScriptRoot。搭配 [CmdletBinding()] 又以 -File 執行時,
    # 參數繫結發生在腳本範圍的自動變數備妥之前,$PSScriptRoot 這時還是空的,
    # $DriverPath 會被綁成空字串,後面第一個吃 -Path 的 cmdlet 就炸。
    # (以 -Command 或 dot-source 執行時反而正常 —— 所以這個 bug 只在
    #  文件裡寫的那種用法下出現。)改成事後補。
    [string]$DriverPath = "",
    [string]$HardwareId = "",
    [switch]$TestSigning,
    # 把驅動憑證加入「受信任的根憑證授權單位」。
    # 這會讓系統信任該憑證簽署的**任何**內容(包含網站憑證),
    # 是明顯的安全性降級。預設關閉,必須由使用者明確選擇。
    [switch]$TrustRootCertificate,
    [switch]$Uninstall,
    # 只解析 INF 並印出硬體 ID 就結束。不需要管理員權限,
    # 用來確認解析結果是否正確。
    [switch]$ParseOnly
)

$ErrorActionPreference = "Stop"

# 補上 $DriverPath 的預設值(見 param 區塊的說明)。
if (-not $DriverPath) {
    $DriverPath = if ($PSScriptRoot) { $PSScriptRoot } else { (Get-Location).Path }
}

# MEDIA 裝置類別 —— 音訊裝置都屬於這一類
$MediaClassGuid = "{4d36e96c-e325-11ce-bfc1-08002be10318}"
$MediaClassName = "MEDIA"

function Write-Step($text) { Write-Host "  $text" -ForegroundColor Cyan }
function Write-Ok($text)   { Write-Host "  [OK] $text" -ForegroundColor Green }
function Write-Warn($text) { Write-Host "  [!]  $text" -ForegroundColor Yellow }
function Write-Err($text)  { Write-Host "  [X]  $text" -ForegroundColor Red }

function Test-Admin {
    $id = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = New-Object Security.Principal.WindowsPrincipal($id)
    return $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

# ── 從 INF 解析硬體 ID ──────────────────────────────────────────────────
# INF 的 [Manufacturer] 指向一個 models 段,該段的每一行長這樣:
#     %DeviceName% = DeviceSection, Root\VirtualAudioDriver
# 我們要的是最後那個逗號後面的硬體 ID。
function Read-InfLines($infPath) {
    # INF 檔常見 UTF-16LE(Windows 慣例)或 UTF-8,且 [Strings] 段可能含
    # 在地化的非 ASCII 名稱。不能交給 Get-Content 用系統代碼頁猜 ——
    # 猜錯時多位元組字元會把換行吃掉,整個檔案糊成一行。
    $bytes = [System.IO.File]::ReadAllBytes($infPath)
    if ($bytes.Length -ge 2 -and $bytes[0] -eq 0xFF -and $bytes[1] -eq 0xFE) {
        $text = [System.Text.Encoding]::Unicode.GetString($bytes, 2, $bytes.Length - 2)
    } elseif ($bytes.Length -ge 2 -and $bytes[0] -eq 0xFE -and $bytes[1] -eq 0xFF) {
        $text = [System.Text.Encoding]::BigEndianUnicode.GetString($bytes, 2, $bytes.Length - 2)
    } elseif ($bytes.Length -ge 3 -and $bytes[0] -eq 0xEF -and $bytes[1] -eq 0xBB -and $bytes[2] -eq 0xBF) {
        $text = [System.Text.Encoding]::UTF8.GetString($bytes, 3, $bytes.Length - 3)
    } else {
        # 沒有 BOM:用 UTF-8 解(ASCII 是其子集,對純 ASCII 的 INF 完全正確)
        $text = [System.Text.Encoding]::UTF8.GetString($bytes)
    }
    return $text -split "`r?`n" | ForEach-Object { $_.Trim() }
}

function Get-HardwareIdFromInf($infPath) {
    $lines = Read-InfLines $infPath

    # [Manufacturer] 的每一行格式是:
    #     %Mfg% = 段名, OS修飾1, OS修飾2
    # 只有第一個 token 是 models 段名,後面都是 OS 修飾詞;
    # 真正存在的段名是「段名」或「段名.OS修飾」。
    $candidates = @()
    $inManufacturer = $false
    foreach ($line in $lines) {
        if ($line -match '^\[(.+)\]$') {
            $inManufacturer = ($Matches[1] -ieq 'Manufacturer')
            continue
        }
        if (-not $inManufacturer) { continue }
        if ($line -match '^\s*;' -or $line -eq '') { continue }
        if ($line -match '=\s*(.+)$') {
            $tokens = @($Matches[1] -split ',' |
                        ForEach-Object { ($_ -split ';')[0].Trim() } |
                        Where-Object { $_ })
            if ($tokens.Count -eq 0) { continue }
            $base = $tokens[0]
            $candidates += $base
            if ($tokens.Count -gt 1) {
                foreach ($decoration in $tokens[1..($tokens.Count - 1)]) {
                    $candidates += "$base.$decoration"
                }
            }
        }
    }

    $current = ""
    foreach ($line in $lines) {
        if ($line -match '^\[(.+)\]$') { $current = $Matches[1]; continue }
        if ($candidates -notcontains $current) { continue }
        if ($line -match '^\s*;' -or $line -eq '') { continue }
        # %Name% = InstallSection, HardwareId
        if ($line -match '=\s*[^,]+,\s*(.+)$') {
            $id = ($Matches[1] -split ';')[0].Trim()
            if ($id) { return $id }
        }
    }
    return $null
}

# ── 前置檢查 ────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "== Virtual Audio Driver $(if ($Uninstall) { '移除' } else { '安裝' }) ==" -ForegroundColor Cyan
Write-Host ""

if (-not $ParseOnly -and -not (Test-Admin)) {
    Write-Err "需要系統管理員權限。請以管理員身分執行。"
    exit 2
}

if (-not (Test-Path $DriverPath)) {
    Write-Err "找不到驅動資料夾:$DriverPath"
    exit 3
}

$inf = Get-ChildItem -Path $DriverPath -Filter *.inf -Recurse -ErrorAction SilentlyContinue |
       Select-Object -First 1
$nefcon = Get-ChildItem -Path $DriverPath -Filter nefconw.exe -Recurse -ErrorAction SilentlyContinue |
          Select-Object -First 1

if (-not $inf) {
    Write-Err "在 $DriverPath 找不到 .inf 驅動檔。"
    Write-Host ""
    Write-Host "  請先從以下網址下載 Virtual Audio Driver 的 release,解壓後把" -ForegroundColor Yellow
    Write-Host "  .inf / .sys / .cat / .cer 以及 nefconw.exe 放進這個資料夾:" -ForegroundColor Yellow
    Write-Host "    https://github.com/VirtualDrivers/Virtual-Audio-Driver/releases" -ForegroundColor Yellow
    Write-Host "  詳見同資料夾的 README.md" -ForegroundColor Yellow
    exit 4
}

Write-Ok "找到驅動:$($inf.Name)"

if (-not $HardwareId) {
    $HardwareId = Get-HardwareIdFromInf $inf.FullName
    if (-not $HardwareId) {
        Write-Err "無法從 INF 解析硬體 ID。請用 -HardwareId 明確指定。"
        exit 5
    }
    Write-Ok "硬體 ID:$HardwareId(自 INF 解析)"
} else {
    Write-Ok "硬體 ID:$HardwareId(手動指定)"
}

if ($ParseOnly) {
    Write-Host ""
    Write-Host "  (--ParseOnly:只解析,未做任何變更)" -ForegroundColor DarkGray
    Write-Output $HardwareId
    exit 0
}

if (-not $nefcon) {
    Write-Warn "找不到 nefconw.exe —— 這類虛擬裝置沒有實體硬體可以觸發安裝,"
    Write-Warn "需要它來建立裝置節點。請一併放進驅動資料夾。"
    Write-Host "    https://github.com/nefarius/nefcon/releases" -ForegroundColor Yellow
    exit 6
}
Write-Ok "找到 nefconw.exe"

# ── 移除 ────────────────────────────────────────────────────────────────
if ($Uninstall) {
    Write-Host ""
    Write-Step "移除裝置節點…"
    & $nefcon.FullName --remove-device-node --hardware-id $HardwareId --class-guid $MediaClassGuid
    if ($LASTEXITCODE -ne 0) { Write-Warn "移除裝置節點回傳 $LASTEXITCODE(可能本來就不存在)" }
    else { Write-Ok "裝置節點已移除" }

    Write-Step "從驅動存放區刪除…"
    $published = (pnputil /enum-drivers) -join "`n"
    $infOriginal = $inf.Name
    if ($published -match "(oem\d+\.inf)[^`n]*`n[^`n]*$([regex]::Escape($infOriginal))") {
        $oem = $Matches[1]
        pnputil /delete-driver $oem /uninstall /force | Out-Null
        Write-Ok "已刪除 $oem"
    } else {
        Write-Warn "在驅動存放區找不到對應項目(可能已移除)"
    }

    # 憑證也要清掉,否則移除程式後系統仍留著被降低的信任基準
    $certs = Get-ChildItem -Path $DriverPath -Filter *.cer -Recurse -ErrorAction SilentlyContinue
    if ($certs) {
        Write-Step "移除匯入的憑證…"
        foreach ($cert in $certs) {
            try {
                $thumb = (New-Object System.Security.Cryptography.X509Certificates.X509Certificate2(
                    $cert.FullName)).Thumbprint
            } catch {
                Write-Warn "無法讀取 $($cert.Name),略過"
                continue
            }
            foreach ($store in @("TrustedPublisher", "Root")) {
                $path = "Cert:\LocalMachine\$store\$thumb"
                if (Test-Path $path) {
                    Remove-Item $path -Force -ErrorAction SilentlyContinue
                    Write-Ok "已從 $store 移除 $($cert.Name)"
                }
            }
        }
    }

    Write-Host ""
    Write-Host "  移除完成。建議重新開機讓變更完全生效。" -ForegroundColor Green
    exit 0
}

# ── 測試簽章(選用,有風險)─────────────────────────────────────────────
if ($TestSigning) {
    Write-Host ""
    Write-Warn "正在啟用 Windows 測試簽章模式。"
    Write-Warn "這會降低系統安全性,且桌面右下角會出現浮水印。重開機後生效。"
    bcdedit /set testsigning on | Out-Null
    if ($LASTEXITCODE -eq 0) { Write-Ok "測試簽章已啟用(需重新開機)" }
    else { Write-Err "無法啟用測試簽章(Secure Boot 開啟時會被拒絕)" }
}

# ── 安裝 ────────────────────────────────────────────────────────────────
Write-Host ""

# 1. 憑證
#
# 兩個存放區的意義差很多,不能混為一談:
#
#   TrustedPublisher  「安裝這個發行者的驅動時不用再問我」。範圍僅限軟體/驅動
#                      的發行者信任,是驅動安裝真正需要的。
#
#   Root(受信任的根)「這是一個憑證授權單位,它簽的任何東西我都信」——
#                      包含網站的 TLS 憑證。範圍大到不該為了裝一個音訊驅動
#                      而動它。
#
# 所以預設只寫入 TrustedPublisher。若驅動的憑證鏈無法驗證(自簽或
# 非公開 CA 簽發),安裝會失敗 —— 那時使用者要自己決定是否用
# -TrustRootCertificate 降低系統的信任基準,而不是被腳本偷偷做掉。
$certs = Get-ChildItem -Path $DriverPath -Filter *.cer -Recurse -ErrorAction SilentlyContinue
if ($certs) {
    Write-Step "匯入驅動發行者憑證…"
    foreach ($cert in $certs) {
        certutil -addstore -f TrustedPublisher $cert.FullName | Out-Null
        Write-Ok "已加入受信任的發行者:$($cert.Name)"

        if ($TrustRootCertificate) {
            Write-Host ""
            Write-Warn "正在把 $($cert.Name) 加入「受信任的根憑證授權單位」。"
            Write-Warn "這代表系統將信任由該憑證簽署的任何內容 —— 包含網站憑證。"
            Write-Warn "這會降低整台電腦的安全性基準,且不會隨本程式移除而自動復原。"
            certutil -addstore -f Root $cert.FullName | Out-Null
            Write-Ok "已加入根憑證存放區(可用 -Uninstall 移除)"
            Write-Host ""
        }
    }
    if (-not $TrustRootCertificate) {
        Write-Host "    (未變更系統的根憑證信任。若驅動因憑證鏈無法驗證而安裝失敗," -ForegroundColor DarkGray
        Write-Host "     請改用已正式簽章的方案,或自行評估後加上 -TrustRootCertificate)" -ForegroundColor DarkGray
    }
} else {
    Write-Warn "沒有找到 .cer 憑證檔 —— 若驅動已由 Microsoft 簽章則不需要"
}

# 2. 加入驅動存放區
Write-Step "將驅動加入驅動程式存放區…"
$addOutput = pnputil /add-driver $inf.FullName /install 2>&1 | Out-String
Write-Host $addOutput.Trim().Split("`n")[-1].Trim() -ForegroundColor DarkGray

if ($LASTEXITCODE -ne 0) {
    Write-Err "pnputil 失敗(結束碼 $LASTEXITCODE)"
    if ($addOutput -match "0x800B0109|signature|簽章") {
        Write-Host ""
        Write-Warn "這看起來是驅動簽章被拒絕。"
        Write-Warn "此驅動的憑證不是 Microsoft 核心驅動認證,在 Secure Boot"
        Write-Warn "開啟的系統上可能無法載入。"
        Write-Warn "可以改用已認證的方案(VB-CABLE / VoiceMeeter),"
        Write-Warn "或加上 -TestSigning 參數重跑(有安全性代價)。"
    }
    exit 7
}
Write-Ok "驅動已加入存放區"

# 3. 建立裝置節點
Write-Step "建立裝置節點…"
& $nefcon.FullName --create-device-node `
    --hardware-id $HardwareId `
    --class-name $MediaClassName `
    --class-guid $MediaClassGuid
if ($LASTEXITCODE -ne 0) {
    Write-Err "建立裝置節點失敗(結束碼 $LASTEXITCODE)"
    exit 8
}
Write-Ok "裝置節點已建立"

# 4. 驗證
Write-Step "驗證裝置是否出現…"
Start-Sleep -Seconds 3

# 不能拿硬體 ID 去比對 InstanceId。
#
# nefconw 建立的是 root 列舉節點,InstanceId 長得像 ROOT\MEDIA\0001 ——
# 裡面**不含**硬體 ID 字串。舊版用 `InstanceId -like "*VirtualAudioDriver*"`
# 永遠比不中(而且 .Replace('Root\','') 區分大小寫,INF 給的是大寫 ROOT\,
# 連取代都沒發生),於是走到「找不到」那一支再落到 exit 0 ——
# 核心明明拒絕載入,卻回報安裝成功。
#
# 硬體 ID 是裝置的**屬性**,要從 DEVPKEY_Device_HardwareIds 讀。
# 只掃 root 列舉的節點。整個 MEDIA 類別在一般筆電上有四五十個裝置,
# 對每一個呼叫 Get-PnpDeviceProperty 要跑好幾分鐘(實測會直接卡住);
# 而 nefconw 建立的虛擬裝置一定掛在 ROOT\ 底下,這樣只剩個位數個候選。
$device = $null
$roots = Get-PnpDevice -Class $MediaClassName -ErrorAction SilentlyContinue |
         Where-Object { $_.InstanceId -like "ROOT\*" }
# 比對時要去掉 ROOT\ 前綴再比。裝置屬性裡列的硬體 ID **不帶**列舉器前綴 ——
# 實測 VB-CABLE 的 ROOT\MEDIA\0000 節點列出來是 `VBAudioVACWDM`,
# 而 INF 給的是 `ROOT\VBAudioVACWDM`。直接 -eq 會永遠比不中。
$wanted = ($HardwareId -replace '^ROOT\\', '')
foreach ($cand in $roots) {
    $ids = (Get-PnpDeviceProperty -InstanceId $cand.InstanceId `
                -KeyName "DEVPKEY_Device_HardwareIds" -ErrorAction SilentlyContinue).Data
    if ($ids -and ($ids | Where-Object { ($_ -replace '^ROOT\\', '') -ieq $wanted })) {
        $device = $cand; break
    }
}

if (-not $device) {
    Write-Err "驅動已加入存放區,但系統中找不到對應的裝置節點。"
    Write-Err "這不是成功 —— 沒有裝置就沒有虛擬麥克風。"
    exit 9
}

# 問題碼比 Status 精確得多:Status 只會說「Error」,問題碼會說**為什麼**。
$problem = (Get-PnpDeviceProperty -InstanceId $device.InstanceId `
                -KeyName "DEVPKEY_Device_ProblemCode" -ErrorAction SilentlyContinue).Data
$problem = if ($null -eq $problem) { 0 } else { [int]$problem }

if ($device.Status -eq "OK" -and $problem -eq 0) {
    Write-Ok "裝置運作正常:$($device.FriendlyName)"
} else {
    Write-Err "裝置已建立,但沒有啟動(狀態「$($device.Status)」,問題碼 $problem)"
    switch ($problem) {
        52 {
            Write-Err "問題碼 52 = CM_PROB_UNSIGNED_DRIVER:核心拒絕載入這個驅動。"
            Write-Err "它的簽章不在核心模式的信任鏈裡 —— pnputil 收下它只代表"
            Write-Err "catalog 是有效的 Authenticode,不代表核心願意載入。"
            Write-Err "這不是設定問題,換參數或重開機都不會改變結果。"
            Write-Err "請改用已通過 WHCP 認證的驅動(例如 VB-CABLE)。"
        }
        39 { Write-Err "問題碼 39:驅動檔案損毀或遺失。" }
        41 { Write-Err "問題碼 41:系統載入了驅動但找不到裝置。" }
        default { Write-Err "請在裝置管理員查看這個問題碼的說明。" }
    }
    exit 9
}

Write-Host ""
Write-Host "  安裝完成。請重新啟動 KTISV,它會自動偵測到虛擬裝置。" -ForegroundColor Green
Write-Host "  在 Discord 的輸入裝置選「Virtual Mic」即可。" -ForegroundColor Green
Write-Host ""
exit 0
