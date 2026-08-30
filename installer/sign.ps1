<#
.SYNOPSIS
    對建置產物做程式碼簽章。

.DESCRIPTION
    未簽章的程式在別人電腦上會觸發 SmartScreen 的「Windows 已保護您的電腦 ·
    發行者:不明」。要消除這個警告,唯一的方法是用**程式碼簽章憑證**簽署,
    沒有任何程式技巧可以繞過。

    這支腳本本身不會產生憑證 —— 它只是把簽章這一步接進建置流程,
    讓你拿到憑證後能直接用。

    憑證從哪來(2026 年的現況):
      * **OV 程式碼簽章憑證**:約 US$200-400/年。需要驗證公司/個人身分。
        簽了之後仍可能有一段時間的 SmartScreen 信譽累積期。
      * **EV 程式碼簽章憑證**:約 US$300-600/年。需要硬體金鑰(HSM/USB token)。
        **立即獲得 SmartScreen 信譽**,是唯一能馬上消除警告的選項。
      * 自簽憑證:對別人的電腦**沒有用**。除非對方主動信任你的根憑證 ——
        而那是要求別人降低系統安全性,不該這樣做。

    憑證來源可以是:
      * .pfx 檔 + 密碼(OV 憑證常見)
      * Windows 憑證存放區裡的指紋(EV 憑證用硬體金鑰時)

.PARAMETER Path
    要簽的檔案。可以給多個。

.PARAMETER PfxPath
    .pfx 憑證檔路徑。

.PARAMETER PfxPassword
    .pfx 的密碼。留空會互動詢問(不要寫進腳本或 CI 記錄)。

.PARAMETER Thumbprint
    改用憑證存放區裡的憑證,指定其指紋。EV 憑證通常走這條。

.PARAMETER TimestampUrl
    時間戳記伺服器。**一定要加時間戳記** —— 否則憑證到期後,
    已經簽好的程式也會變成無效簽章。

.EXAMPLE
    .\sign.ps1 -Path ..\KTISV\bin\Release\net10.0\win-x64\publish\KTISV.exe -PfxPath cert.pfx

.EXAMPLE
    .\sign.ps1 -Path Output\KTISV-Setup.exe -Thumbprint A1B2C3...
#>

[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string[]]$Path,

    [string]$PfxPath = "",
    [string]$PfxPassword = "",
    [string]$Thumbprint = "",
    [string]$TimestampUrl = "http://timestamp.digicert.com",
    [switch]$Verify
)

$ErrorActionPreference = "Stop"

function Write-Ok($text)   { Write-Host "  [OK] $text" -ForegroundColor Green }
function Write-Warn($text) { Write-Host "  [!]  $text" -ForegroundColor Yellow }
function Write-Err($text)  { Write-Host "  [X]  $text" -ForegroundColor Red }

# ── 找 signtool ─────────────────────────────────────────────────────────
function Find-SignTool {
    $found = Get-Command signtool.exe -ErrorAction SilentlyContinue
    if ($found) { return $found.Source }

    # Windows SDK 的預設位置,取版本號最大的
    $roots = @(
        "${env:ProgramFiles(x86)}\Windows Kits\10\bin",
        "$env:ProgramFiles\Windows Kits\10\bin"
    )
    foreach ($root in $roots) {
        if (-not (Test-Path $root)) { continue }
        $candidate = Get-ChildItem $root -Recurse -Filter signtool.exe -ErrorAction SilentlyContinue |
                     Where-Object { $_.FullName -match '\\x64\\' } |
                     Sort-Object FullName -Descending |
                     Select-Object -First 1
        if ($candidate) { return $candidate.FullName }
    }
    return $null
}

# ── 只驗證,不簽 ────────────────────────────────────────────────────────
if ($Verify) {
    Write-Host "== 簽章狀態 ==" -ForegroundColor Cyan
    foreach ($file in $Path) {
        if (-not (Test-Path $file)) { Write-Warn "找不到 $file"; continue }
        $sig = Get-AuthenticodeSignature $file
        $name = Split-Path $file -Leaf
        switch ($sig.Status) {
            "Valid" {
                Write-Ok "$name — 簽章有效"
                Write-Host "       簽署者: $($sig.SignerCertificate.Subject)" -ForegroundColor DarkGray
                Write-Host "       到期日: $($sig.SignerCertificate.NotAfter)" -ForegroundColor DarkGray
                if ($sig.TimeStamperCertificate) {
                    Write-Host "       已加時間戳記" -ForegroundColor DarkGray
                } else {
                    Write-Warn "       沒有時間戳記 —— 憑證到期後簽章會失效"
                }
            }
            "NotSigned" { Write-Warn "$name — 未簽章(在其他電腦上會觸發 SmartScreen 警告)" }
            default     { Write-Err  "$name — $($sig.Status): $($sig.StatusMessage)" }
        }
    }
    exit 0
}

# ── 簽章 ────────────────────────────────────────────────────────────────
if (-not $PfxPath -and -not $Thumbprint) {
    Write-Host ""
    Write-Err "沒有提供憑證,無法簽章。"
    Write-Host ""
    Write-Host "  未簽章的程式在其他電腦上會顯示「Windows 已保護您的電腦」。" -ForegroundColor Yellow
    Write-Host "  這**只能**用真正的程式碼簽章憑證解決,沒有程式技巧可以繞過。" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "  取得憑證後:" -ForegroundColor Cyan
    Write-Host "    .\sign.ps1 -Path <檔案> -PfxPath <憑證.pfx>"
    Write-Host "    .\sign.ps1 -Path <檔案> -Thumbprint <指紋>"
    Write-Host ""
    Write-Host "  只想檢查目前狀態:" -ForegroundColor Cyan
    Write-Host "    .\sign.ps1 -Path <檔案> -Verify"
    Write-Host ""
    exit 2
}

$signtool = Find-SignTool
if (-not $signtool) {
    Write-Err "找不到 signtool.exe。請安裝 Windows SDK:"
    Write-Host "    winget install Microsoft.WindowsSDK" -ForegroundColor Yellow
    exit 3
}
Write-Host "  signtool: $signtool" -ForegroundColor DarkGray

if ($PfxPath -and -not (Test-Path $PfxPath)) {
    Write-Err "找不到憑證檔:$PfxPath"
    exit 4
}

# 密碼不從參數帶進來時互動詢問 —— 避免出現在命令歷程或 CI 記錄裡
if ($PfxPath -and -not $PfxPassword) {
    $secure = Read-Host "請輸入 .pfx 密碼" -AsSecureString
    $PfxPassword = [Runtime.InteropServices.Marshal]::PtrToStringAuto(
        [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secure))
}

$failed = 0
foreach ($file in $Path) {
    if (-not (Test-Path $file)) { Write-Warn "找不到 $file,略過"; continue }

    $args = @("sign", "/fd", "SHA256", "/td", "SHA256", "/tr", $TimestampUrl)
    if ($Thumbprint) {
        $args += @("/sha1", $Thumbprint)
    } else {
        $args += @("/f", $PfxPath)
        if ($PfxPassword) { $args += @("/p", $PfxPassword) }
    }
    $args += $file

    & $signtool @args | Out-Null
    if ($LASTEXITCODE -eq 0) {
        Write-Ok "已簽署 $(Split-Path $file -Leaf)"
    } else {
        Write-Err "簽署失敗 $(Split-Path $file -Leaf)(結束碼 $LASTEXITCODE)"
        $failed++
    }
}

if ($failed -gt 0) { exit 5 }

Write-Host ""
Write-Host "  簽章完成。建議用 -Verify 確認結果。" -ForegroundColor Green
exit 0
