@echo off
REM Starts the dashboard with numpy's thread pool capped, so it stays light on memory.
cd /d "%~dp0"
set OPENBLAS_NUM_THREADS=2
set OMP_NUM_THREADS=2
".venv\Scripts\python.exe" -m streamlit run app\Home.py
