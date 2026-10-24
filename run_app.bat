@echo off
title VOXEL Earth Observation Platform
echo =====================================================================
echo  Starting VOXEL Platform Localhost Server (High Performance Mode)
echo =====================================================================
cd /d "%~dp0"

if not exist ".venv\Scripts\streamlit.exe" (
    echo Error: Virtual environment not found at .venv\Scripts\streamlit.exe
    pause
    exit /b 1
)

echo Starting Streamlit on http://localhost:8501 ...
start "" "http://localhost:8501"
.venv\Scripts\streamlit.exe run app.py --server.port=8501 --server.address=127.0.0.1 --server.fileWatcherType=none --server.headless=false
pause
