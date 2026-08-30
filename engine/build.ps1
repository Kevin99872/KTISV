# 用 uv + PyInstaller 打包 KTISV 音訊引擎
#   powershell -ExecutionPolicy Bypass -File build.ps1
#
# 產物:engine/dist/ktisv-engine/  (整個資料夾就是引擎,裡面的 exe 是進入點)
# C# 前端會自動在開發樹裡找到這個路徑。

$ErrorActionPreference = "Stop"
$engine = $PSScriptRoot

Write-Host "== 打包 KTISV 引擎 ==" -ForegroundColor Cyan

# 確保有 uv
try { uv --version | Out-Null }
catch { Write-Host "找不到 uv。安裝方式見 https://docs.astral.sh/uv/" -ForegroundColor Red; exit 1 }

$venvPython = Join-Path $engine ".venv\Scripts\python.exe"

Push-Location $engine
try {
    Write-Host "`n同步相依套件(含 build 群組)…" -ForegroundColor Cyan
    uv sync --group build
    if ($LASTEXITCODE -ne 0) {
        # uv sync 失敗最常見的原因是網路抓不到套件。若現有環境已經堪用,
        # 不該因為一次網路抖動就中止打包 —— 用「能不能實際載入」當判準,
        # 而不是相信 sync 的結束碼。
        Write-Host "uv sync 失敗(常見於網路問題),改為檢查現有環境…" -ForegroundColor Yellow
        if (-not (Test-Path $venvPython)) {
            throw "uv sync 失敗,且找不到 $venvPython。請檢查網路後重試。"
        }
        & $venvPython -c "import PyInstaller, ktisv_engine" 2>$null
        if ($LASTEXITCODE -ne 0) {
            throw "uv sync 失敗,且現有環境缺少 PyInstaller 或 ktisv_engine。請檢查網路後重試。"
        }
        Write-Host "現有環境可用,繼續打包。" -ForegroundColor Green
    }

    # 先關掉可能還開著的舊引擎,否則 dist 會被鎖住
    Get-Process ktisv-engine -ErrorAction SilentlyContinue | Stop-Process -Force
    Start-Sleep -Milliseconds 500

    Write-Host "`n執行 PyInstaller…" -ForegroundColor Cyan
    # 直接用 venv 的 python 而非 `uv run` —— 後者會再觸發一次相依解析,
    # 網路不通時會重複卡在同一個地方。
    & $venvPython -m PyInstaller ktisv_engine.spec --noconfirm --distpath dist --workpath build
    if ($LASTEXITCODE -ne 0) { throw "PyInstaller 失敗" }

    $exe = Join-Path $engine "dist\ktisv-engine\ktisv-engine.exe"
    if (-not (Test-Path $exe)) { throw "找不到產物 $exe" }

    Write-Host "`n驗證產物…" -ForegroundColor Cyan
    & $exe --selftest
    $ok = $LASTEXITCODE -eq 0

    Write-Host ""
    if ($ok) {
        Write-Host "打包完成:$exe" -ForegroundColor Green
        Write-Host "從原始碼樹執行前端時會自動抓到它。"
        Write-Host "要隨前端一起發佈,把整個 dist\ktisv-engine\ 複製到前端輸出資料夾下的 engine-bin\。"
    } else {
        Write-Host "產物 selftest 有缺項,請見上方輸出。" -ForegroundColor Yellow
    }
}
finally {
    Pop-Location
}
