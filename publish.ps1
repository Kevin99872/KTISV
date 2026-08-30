# 產出可散布的 KTISV 整包
#   powershell -ExecutionPolicy Bypass -File publish.ps1
#   powershell -ExecutionPolicy Bypass -File publish.ps1 -SingleFile
#
# 產物:KTISV\bin\Release\net10.0\win-x64\publish\
#   ├─ KTISV.exe          自包含,目標電腦不需要裝 .NET
#   └─ engine-bin\        內建的 Python 引擎,目標電腦不需要裝 Python
#
# 為什麼預設**不用**單一檔案
# ---------------------------
# 自解壓縮的單一 exe 在執行時會把內容解到暫存目錄再載入 —— 那正是惡意軟體
# 加殼器的行為特徵,防毒軟體的啟發式偵測經常誤判,輕則警告、重則直接隔離。
#
# 而既然是用安裝檔散布,單一檔案並沒有帶來任何好處(安裝檔本來就把東西
# 打包好了),卻要付出誤判的代價。所以預設用一般的資料夾部署。
#
# 真的需要「一個檔案帶著走」時再加 -SingleFile。

param(
    [switch]$SingleFile
)

$ErrorActionPreference = "Stop"
$root = $PSScriptRoot

Write-Host "== KTISV 發佈 ==" -ForegroundColor Cyan

# 先關掉可能還開著的舊行程,避免檔案被鎖
Get-Process ktisv-engine, KTISV -ErrorAction SilentlyContinue | Stop-Process -Force
Start-Sleep -Milliseconds 500

# 1. 打包 Python 引擎
Write-Host "`n[1/2] 打包 Python 引擎…" -ForegroundColor Cyan
& (Join-Path $root "engine\build.ps1")
if ($LASTEXITCODE -ne 0) { Write-Host "引擎打包失敗。" -ForegroundColor Red; exit 1 }

# 2. 發佈前端(會自動把引擎帶進 engine-bin\)
Write-Host "`n[2/2] 發佈 Avalonia 前端…" -ForegroundColor Cyan
$publishArgs = @(
    (Join-Path $root "KTISV\KTISV.csproj"),
    "-c", "Release", "-r", "win-x64", "--self-contained", "--nologo"
)
if ($SingleFile) {
    Write-Host "  模式:單一檔案(注意:較容易被防毒誤判)" -ForegroundColor Yellow
    $publishArgs += @("-p:PublishSingleFile=true",
                      "-p:IncludeNativeLibrariesForSelfExtract=true")
} else {
    Write-Host "  模式:資料夾部署(誤判率較低)" -ForegroundColor DarkGray
}
dotnet publish @publishArgs
if ($LASTEXITCODE -ne 0) { Write-Host "前端發佈失敗。" -ForegroundColor Red; exit 1 }

$pub = Join-Path $root "KTISV\bin\Release\net10.0\win-x64\publish"
Write-Host "`n完成。整包在:" -ForegroundColor Green
Write-Host "  $pub"
Write-Host "`n把整個 publish 資料夾壓成 zip 就能散布。對方解壓後雙擊 KTISV.exe 即可,"
Write-Host "不需要另外安裝 Python 或 .NET。" -ForegroundColor Green
Write-Host "`n提醒:要送音訊給 Discord 的電腦仍需安裝 VB-CABLE。" -ForegroundColor Yellow
