@echo off
chcp 65001 > nul
title 電力需給データ分析ツール

echo ========================================
echo   電力需給データ分析ツール ランチャー
echo ========================================
echo.

cd /d "%~dp0"

REM Python 3.11を優先的に探す
where py >nul 2>&1
if %errorlevel%==0 (
    py -3.11 run.py
    if %errorlevel%==0 goto :end
)

REM 通常のpythonコマンドを試す
where python >nul 2>&1
if %errorlevel%==0 (
    python run.py
    goto :end
)

echo.
echo [エラー] Pythonが見つかりません。
echo Python 3.11をインストールしてください。
echo https://www.python.org/downloads/
echo.
pause

:end
