# VOXEL Platform Localhost Launcher
Write-Host "=====================================================================" -ForegroundColor Cyan
Write-Host " Starting VOXEL Platform Localhost Server (High Performance Mode)" -ForegroundColor Green
Write-Host "=====================================================================" -ForegroundColor Cyan

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

$StreamlitExe = Join-Path $ScriptDir ".venv\Scripts\streamlit.exe"
if (-not (Test-Path $StreamlitExe)) {
    Write-Error "Virtual environment not found at $StreamlitExe"
    exit 1
}

Write-Host "Launching VOXEL Dashboard on http://localhost:8501 ..." -ForegroundColor Yellow
Start-Process "http://localhost:8501"
& $StreamlitExe run app.py --server.port=8501 --server.address=127.0.0.1 --server.fileWatcherType=none --server.headless=false
