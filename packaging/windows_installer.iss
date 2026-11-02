; Inno Setup 6 Installer Script for MicroRing Studio (Windows 10 / 11 x64)
; Installs cleanly without requiring Administrator privileges, creates Start Menu
; and optional Desktop shortcuts, and provides standard Windows Add/Remove Programs uninstall.

#ifndef MyAppVersion
  #define MyAppVersion "3.6.0"
#endif
#ifndef OutputBaseFilename
  #define OutputBaseFilename "MicroRing-Studio-v3.6.0-Windows-Setup-x64"
#endif

[Setup]
AppId={{8D4F6A2E-5C31-4B9E-A18F-90E12C4D77B3}
AppName=MicroRing Studio
AppVersion={#MyAppVersion}
AppPublisher=Zhihao Ren
AppPublisherURL=https://github.com/ZhihaoRenKevin/microring-studio
AppSupportURL=https://github.com/ZhihaoRenKevin/microring-studio/issues
DefaultDirName={autopf}\MicroRing Studio
DefaultGroupName=MicroRing Studio
DisableProgramGroupPage=yes
LicenseFile=..\LICENSE
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
OutputDir=..\dist
OutputBaseFilename={#OutputBaseFilename}
SetupIconFile=microring.ico
UninstallDisplayIcon={app}\MicroRing Studio.exe
Compression=lzma2/fast
SolidCompression=yes
WizardStyle=modern
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"

[Files]
Source: "..\dist\MicroRing Studio\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{autoprograms}\MicroRing Studio"; Filename: "{app}\MicroRing Studio.exe"
Name: "{autodesktop}\MicroRing Studio"; Filename: "{app}\MicroRing Studio.exe"; Tasks: desktopicon

[Run]
Filename: "{app}\MicroRing Studio.exe"; Description: "{cm:LaunchProgram,MicroRing Studio}"; Flags: nowait postinstall skipifsilent
