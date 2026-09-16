@echo off
title Jarvis Assistant v0.1.0
cd /d "%~dp0"
python "%~dp0run_jarvis.py" --open %*
