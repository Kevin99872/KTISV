; KTISV 安裝檔(Inno Setup 6)
;
; 編譯前請先執行 publish.ps1 產生發佈輸出。
; 編譯:  "C:\Program Files (x86)\Inno Setup 6\ISCC.exe" installer\KTISV.iss
; 或直接跑:  installer\build-installer.ps1
;
; 產物:installer\Output\KTISV-Setup.exe

#define AppName "KTISV"
#define AppVersion "0.1.0"
#define AppPublisher "KTISV"
#define AppExeName "KTISV.exe"
#define PublishDir "..\KTISV\bin\Release\net10.0\win-x64\publish"

[Setup]
AppId={{0BE1CED2-811A-4EA2-BE0F-AABA01FB356F}
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher={#AppPublisher}
; 裝到使用者自己的 AppData,而不是 Program Files。
;
; 這樣安裝完全不需要管理員權限 —— 不跳 UAC,也少掉一層系統層級的稽核。
; 未簽章的程式一旦要求提權,SmartScreen 與防毒的檢查會嚴格很多,
; 使用者看到的警告也更嚇人。
;
; KTISV 本來就只寫入 LocalAppData(設定、快取),沒有任何需要寫入
; Program Files 或 HKLM 的理由。
DefaultDirName={localappdata}\{#AppName}
DefaultGroupName={#AppName}
OutputDir=Output
OutputBaseFilename=KTISV-Setup
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
DisableProgramGroupPage=yes
UninstallDisplayIcon={app}\{#AppExeName}

[Languages]
Name: "chinesetrad"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "建立桌面捷徑"; GroupDescription: "附加工作:"

[Files]
; 整個發佈資料夾(自包含,不需要目標電腦裝 .NET / Python)。
;
; 刻意用資料夾部署而非單一檔案打包:自解壓縮的單一 exe 在執行時會把內容
; 解到暫存目錄再載入,那正是惡意軟體加殼器的行為特徵,防毒的啟發式偵測
; 經常誤判 —— 輕則警告、重則直接隔離,使用者會說「程式打不開」。
; 既然是用安裝檔散布,單一檔案也沒有帶來任何好處。
Source: "{#PublishDir}\*"; DestDir: "{app}"; \
    Flags: ignoreversion recursesubdirs createallsubdirs

; 虛擬音訊驅動(Virtual Audio Driver,MIT)。
;
; 只把檔案放進去,安裝檔本身**不會**去裝驅動。這是刻意的:
;
;   * 整個安裝維持 PrivilegesRequired=lowest,不跳 UAC。未簽章的安裝檔
;     一旦要求提權,SmartScreen 與防毒的檢查會嚴格很多。
;   * 裝核心驅動要動系統憑證存放區,那是使用者該自己點頭的事,
;     不該藏在「下一步、下一步」裡。
;
; 實際安裝由 App 內的設定精靈觸發,那時才單獨提權(見
; Services\DriverInstaller.cs)。它會從 exe 旁邊往上找 driver\install-driver.ps1,
; 所以放在 {app}\driver\ 剛好會被找到,精靈也才會顯示「安裝內建驅動」按鈕。
;
; skipifsourcedoesntexist:沒跑過 driver\fetch-driver.ps1 時 driver\ 裡只有
; README 與腳本,那時仍要編得出安裝檔 —— 功能退化成引導使用者自己裝
; VB-CABLE,而不是整個 build 失敗。
; Excludes fetch-driver.ps1:那是建置時用來抓驅動的下載腳本,對使用者沒有用途,
; 而且它會覆寫 driver\ 裡的檔案 —— 不該出現在安裝好的目錄裡。
Source: "..\driver\*"; DestDir: "{app}\driver"; \
    Excludes: "fetch-driver.ps1"; \
    Flags: ignoreversion recursesubdirs createallsubdirs skipifsourcedoesntexist

[Icons]
Name: "{group}\{#AppName}"; Filename: "{app}\{#AppExeName}"
Name: "{group}\移除 {#AppName}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\{#AppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#AppExeName}"; Description: "立即啟動 {#AppName}"; \
    Flags: nowait postinstall skipifsilent

; 沒有 [UninstallRun] —— 移除驅動要提權,而這個解除安裝程式是 lowest 權限。
; 改在 [Code] 裡問過使用者再單獨提權,見 CurUninstallStepChanged。

[Code]
procedure CurStepChanged(CurStep: TSetupStep);
begin
  if CurStep = ssPostInstall then
  begin
    MsgBox('安裝完成。' + #13#10 + #13#10 +
           'KTISV 現在就能使用 —— 耳機那一路完全正常。' + #13#10 + #13#10 +
           '若要把混音送進 Discord,還需要一個「虛擬麥克風」。' + #13#10 +
           '第一次開啟 KTISV 時會自動偵測,沒有的話會跳出設定精靈' + #13#10 +
           '幫你安裝內建的虛擬音效卡驅動(需要系統管理員權限)。',
           mbInformation, MB_OK);
  end;
end;

// 解除安裝時把驅動一併移除 —— 但要先問。
//
// 使用者可能是為了別的軟體才留著這張虛擬音效卡,而且移除驅動需要提權,
// 不該無聲進行。腳本不存在(沒打包驅動的版本)時整段跳過。
procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
var
  ScriptPath: String;
  ResultCode: Integer;
begin
  if CurUninstallStep = usUninstall then
  begin
    ScriptPath := ExpandConstant('{app}\driver\install-driver.ps1');
    if FileExists(ScriptPath) then
    begin
      if MsgBox('要一併移除 KTISV 安裝的虛擬音效卡驅動嗎?' + #13#10 + #13#10 +
                '若其他軟體也在用這張虛擬音效卡,請選「否」保留。' + #13#10 +
                '移除需要系統管理員權限。',
                mbConfirmation, MB_YESNO) = IDYES then
      begin
        // runas:解除安裝程式本身是 lowest 權限,這一步要單獨提權。
        ShellExec('runas', 'powershell.exe',
                  '-NoProfile -ExecutionPolicy Bypass -File "' + ScriptPath +
                  '" -DriverPath "' + ExpandConstant('{app}\driver') + '" -Uninstall',
                  '', SW_HIDE, ewWaitUntilTerminated, ResultCode);
      end;
    end;
  end;
end;
