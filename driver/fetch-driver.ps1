<#
.SYNOPSIS
    下載 Virtual Audio Driver 與 nefconw.exe 到這個資料夾。

.DESCRIPTION
    driver\ 在版本庫裡是空的(只有 README 與腳本)—— 驅動是二進位檔,不屬於
    原始碼,而且各自有自己的授權。這個腳本負責把它們抓下來,讓建置可重現。

    抓兩樣東西:

      1. Virtual Audio Driver(MIT,VirtualDrivers/Virtual-Audio-Driver)
         驅動本體:.inf / .sys / .cat / .cer

      2. nefconw.exe(MIT,nefarius/nefcon)
         建立 root 列舉的裝置節點用。虛擬裝置沒有實體硬體可以觸發安裝,
         少了它 pnputil 把驅動放進存放區之後不會有任何裝置出現。

    抓完之後跑 install-driver.ps1 才有東西可裝。

.PARAMETER DriverVersion
    釘住 Virtual Audio Driver 的版本(例如 25.7.14)。省略時抓最新的 release。

.PARAMETER Force
    覆蓋已存在的檔案。

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File driver\fetch-driver.ps1
#>

[CmdletBinding()]
param(
    [string]$DriverVersion = "",
    [switch]$Force
)

$ErrorActionPreference = "Stop"
$here = $PSScriptRoot

# GitHub API 從 2020 起就要求帶 User-Agent,不帶會被 403 擋掉。
$headers = @{ "User-Agent" = "KTISV-fetch-driver" }

function Write-Step($t) { Write-Host "  $t" -ForegroundColor Cyan }
function Write-Ok($t)   { Write-Host "  [OK] $t" -ForegroundColor Green }
function Write-Warn($t) { Write-Host "  [!]  $t" -ForegroundColor Yellow }

# TLS 1.2:PowerShell 5.1 預設可能還在用 TLS 1.0,GitHub 早就不收了。
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

$existing = Get-ChildItem $here -Filter "*.inf" -ErrorAction SilentlyContinue
if ($existing -and -not $Force) {
    Write-Warn "driver\ 裡已經有 .inf 檔了。要重抓請加 -Force。"
    exit 0
}

$tmp = Join-Path $env:TEMP ("ktisv-driver-" + [Guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Path $tmp -Force | Out-Null

try {
    # ── 1. Virtual Audio Driver ─────────────────────────────────────────
    Write-Host "== 取得 Virtual Audio Driver ==" -ForegroundColor Cyan

    $api = if ($DriverVersion) {
        "https://api.github.com/repos/VirtualDrivers/Virtual-Audio-Driver/releases/tags/$DriverVersion"
    } else {
        "https://api.github.com/repos/VirtualDrivers/Virtual-Audio-Driver/releases/latest"
    }
    $rel = Invoke-RestMethod $api -Headers $headers
    Write-Step ("版本 " + $rel.tag_name + "  (發佈於 " + $rel.published_at + ")")

    $age = ((Get-Date) - [datetime]$rel.published_at).TotalDays
    if ($age -gt 365) {
        Write-Warn ("這個 release 已經 {0:N0} 天沒更新了。" -f $age)
        Write-Warn "在 2026/4 之後的 Windows 11 24H2+ 上很可能載不起來,見 README。"
    }

    $asset = $rel.assets | Where-Object { $_.name -like "*.zip" } | Select-Object -First 1
    if (-not $asset) { throw "這個 release 沒有 .zip 資產。" }

    $zip = Join-Path $tmp $asset.name
    Write-Step ("下載 " + $asset.name + " ({0:N0} bytes)" -f $asset.size)
    Invoke-WebRequest $asset.browser_download_url -OutFile $zip -Headers $headers

    $ext = Join-Path $tmp "vad"
    Expand-Archive $zip -DestinationPath $ext -Force

    # release 的壓縮檔內部結構換過幾次(有時多包一層資料夾),所以用遞迴找,
    # 不要假設檔案就在根目錄。
    $wanted = @("*.inf", "*.sys", "*.cat", "*.cer")
    $found = 0
    foreach ($pattern in $wanted) {
        Get-ChildItem $ext -Recurse -Filter $pattern -ErrorAction SilentlyContinue |
            ForEach-Object {
                Copy-Item $_.FullName (Join-Path $here $_.Name) -Force
                Write-Ok ("{0}  ({1:N0} bytes)" -f $_.Name, $_.Length)
                $found++
            }
    }
    if ($found -eq 0) { throw "壓縮檔裡找不到任何驅動檔。" }
    if (-not (Get-ChildItem $here -Filter "*.inf")) { throw "少了 .inf,驅動無法安裝。" }

    # ── 2. nefconw.exe ──────────────────────────────────────────────────
    Write-Host "`n== 取得 nefconw.exe ==" -ForegroundColor Cyan

    $nef = Invoke-RestMethod "https://api.github.com/repos/nefarius/nefcon/releases/latest" -Headers $headers
    Write-Step ("版本 " + $nef.tag_name)

    $nefAsset = $nef.assets | Where-Object { $_.name -like "*.zip" } | Select-Object -First 1
    if (-not $nefAsset) { throw "nefcon 的 release 沒有 .zip 資產。" }

    $nefZip = Join-Path $tmp $nefAsset.name
    Write-Step ("下載 " + $nefAsset.name + " ({0:N0} bytes)" -f $nefAsset.size)
    Invoke-WebRequest $nefAsset.browser_download_url -OutFile $nefZip -Headers $headers

    $nefExt = Join-Path $tmp "nefcon"
    Expand-Archive $nefZip -DestinationPath $nefExt -Force

    # nefcon 同時附 x86 / x64 / arm64。這個專案只發佈 x64,挑對應的那個 ——
    # 挑錯架構的話 install-driver.ps1 會在建立裝置節點那一步無聲失敗。
    $nefExe = Get-ChildItem $nefExt -Recurse -Filter "nefconw.exe" |
              Where-Object { $_.FullName -match "x64|amd64" } |
              Select-Object -First 1
    if (-not $nefExe) {
        $nefExe = Get-ChildItem $nefExt -Recurse -Filter "nefconw.exe" | Select-Object -First 1
        if ($nefExe) { Write-Warn "找不到標示 x64 的版本,改用 $($nefExe.FullName)" }
    }
    if (-not $nefExe) { throw "壓縮檔裡找不到 nefconw.exe。" }

    Copy-Item $nefExe.FullName (Join-Path $here "nefconw.exe") -Force
    Write-Ok ("nefconw.exe  ({0:N0} bytes)" -f $nefExe.Length)

    # ── 3. 確認備齊 ─────────────────────────────────────────────────────
    Write-Host "`n== driver\ 現在的內容 ==" -ForegroundColor Cyan
    Get-ChildItem $here -File | Where-Object { $_.Name -notlike "*.ps1" -and $_.Name -ne "README.md" } |
        ForEach-Object { "  {0,10:N0}  {1}" -f $_.Length, $_.Name }

    Write-Host "`n完成。接下來:" -ForegroundColor Green
    Write-Host "  驗證解析 :  powershell -File driver\install-driver.ps1 -ParseOnly"
    Write-Host "  重建安裝檔:  powershell -File installer\build-installer.ps1"
}
finally {
    Remove-Item $tmp -Recurse -Force -ErrorAction SilentlyContinue
}
