# 產生一步到位的安裝檔 KTISV-Setup.exe
#   powershell -ExecutionPolicy Bypass -File installer\build-installer.ps1
#
# 會依序:打包 Python 引擎 → 發佈 C# 前端 → 用 Inno Setup 編譯成單一安裝檔。
# 產物:installer\Output\KTISV-Setup.exe

param(
    [switch]$SkipPublish   # 已經跑過 publish.ps1 時可略過重新發佈
)

$ErrorActionPreference = "Stop"
$installerDir = $PSScriptRoot
$root = Split-Path $installerDir -Parent

Write-Host "== 產生 KTISV 安裝檔 ==" -ForegroundColor Cyan

# --- 找 Inno Setup ---
#
# 不寫死版本號與安裝位置:winget 預設裝到 %LOCALAPPDATA%\Programs,
# 手動安裝則多在 Program Files,而資料夾名還帶著主版本號(Inno Setup 6)。
# 先試常見路徑,再退回實際掃描。
$isccCandidates = @()
foreach ($base in @($env:ProgramFiles, ${env:ProgramFiles(x86)},
                    (Join-Path $env:LOCALAPPDATA "Programs"))) {
    if (-not $base -or -not (Test-Path $base)) { continue }
    $isccCandidates += Get-ChildItem -Path $base -Directory -Filter "Inno Setup*" `
                           -ErrorAction SilentlyContinue |
                       ForEach-Object { Join-Path $_.FullName "ISCC.exe" }
}
$isccCandidates += "ISCC.exe"   # 已加入 PATH 的情況

$iscc = $isccCandidates | Where-Object {
    if ($_ -eq "ISCC.exe") { [bool](Get-Command $_ -ErrorAction SilentlyContinue) }
    else { Test-Path $_ }
} | Select-Object -First 1

if (-not $iscc) {
    Write-Host ""
    Write-Host "找不到 Inno Setup 編譯器 (ISCC.exe)。" -ForegroundColor Red
    Write-Host "安裝方式:" -ForegroundColor Yellow
    Write-Host "    winget install JRSoftware.InnoSetup" -ForegroundColor Yellow
    Write-Host "或到 https://jrsoftware.org/isdl.php 下載。" -ForegroundColor Yellow
    exit 1
}
Write-Host "  找到 Inno Setup: $iscc" -ForegroundColor DarkGray

# --- 1. 發佈 App + 引擎 ---
if (-not $SkipPublish) {
    Write-Host "`n[1/2] 打包引擎並發佈前端…" -ForegroundColor Cyan
    & (Join-Path $root "publish.ps1")
    if ($LASTEXITCODE -ne 0) { Write-Host "發佈失敗。" -ForegroundColor Red; exit 1 }
} else {
    Write-Host "`n[1/2] 略過發佈(-SkipPublish)" -ForegroundColor DarkGray
}

$publishDir = Join-Path $root "KTISV\bin\Release\net10.0\win-x64\publish"
if (-not (Test-Path (Join-Path $publishDir "KTISV.exe"))) {
    Write-Host "找不到發佈產物:$publishDir\KTISV.exe" -ForegroundColor Red
    Write-Host "請先執行 publish.ps1。" -ForegroundColor Yellow
    exit 1
}

# 提醒:driver 資料夾是選用的
$driverDir = Join-Path $root "driver"
$hasDriverFiles = (Test-Path $driverDir) -and
                  (Get-ChildItem $driverDir -Filter *.inf -Recurse -ErrorAction SilentlyContinue)
if ($hasDriverFiles) {
    Write-Host "  將一併打包虛擬音訊驅動" -ForegroundColor DarkGray
} else {
    Write-Host "  driver 資料夾沒有驅動檔 —— 安裝檔不含驅動" -ForegroundColor Yellow
    Write-Host "  (使用者仍可在第一次啟動的精靈裡選擇安裝 VB-CABLE)" -ForegroundColor Yellow
}

# --- 2. 編譯安裝檔 ---
Write-Host "`n[2/2] 編譯安裝檔…" -ForegroundColor Cyan
& $iscc (Join-Path $installerDir "KTISV.iss")
if ($LASTEXITCODE -ne 0) { Write-Host "Inno Setup 編譯失敗。" -ForegroundColor Red; exit 1 }

$output = Join-Path $installerDir "Output\KTISV-Setup.exe"
if (Test-Path $output) {
    $sizeMb = [math]::Round((Get-Item $output).Length / 1MB, 1)
    Write-Host "`n完成:$output ($sizeMb MB)" -ForegroundColor Green
    Write-Host "這一個檔案就是完整的安裝程式,對方不需要另外裝 Python 或 .NET。" -ForegroundColor Green
} else {
    Write-Host "編譯結束但找不到輸出檔。" -ForegroundColor Red
    exit 1
}
