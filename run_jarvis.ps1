# PowerShell Launcher for Jarvis Assistant
param()

$Host.UI.RawUI.WindowTitle = "Jarvis Assistant v0.1.0"
Set-Location "C:\dev\jarvis"
Write-Host "Iniciando Jarvis..." -ForegroundColor Cyan
python C:\dev\jarvis\run_jarvis.py
