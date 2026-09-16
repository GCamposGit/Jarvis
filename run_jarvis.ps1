# PowerShell Launcher for Jarvis Assistant
param(
    [switch]$OpenBrowser = $true
)

$Host.UI.RawUI.WindowTitle = "Jarvis Assistant v0.1.0"
Set-Location "C:\dev\Jarvis"
Write-Host "Iniciando Jarvis Assistant..." -ForegroundColor Cyan
if ($OpenBrowser) {
    python "C:\dev\Jarvis\run_jarvis.py" --open
} else {
    python "C:\dev\Jarvis\run_jarvis.py"
}
