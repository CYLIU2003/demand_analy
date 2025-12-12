#!/bin/bash
# -*- coding: utf-8 -*-
# 電力需給データ分析ツール - Linux用ランチャー
# 
# このスクリプトは以下を行います:
# 1. 仮想環境 ".demand_analy" の存在を確認
# 2. なければ自動で作成し、パッケージをインストール
# 3. あればそのまま使用
# 4. main.py を仮想環境で実行

set -e

# 設定
VENV_NAME=".demand_analy"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PATH="$HOME/$VENV_NAME"
REQUIREMENTS_FILE="$SCRIPT_DIR/requirements.txt"
MAIN_SCRIPT="$SCRIPT_DIR/main.py"
INSTALLED_FLAG="$VENV_PATH/.packages_installed"

echo "========================================"
echo "  電力需給データ分析ツール ランチャー"
echo "========================================"
echo ""

# 仮想環境が存在するかチェック
NEED_INSTALL=0
if [ -d "$VENV_PATH" ] && [ -f "$VENV_PATH/bin/python" ]; then
    echo "✅ 仮想環境 '$VENV_NAME' を検出しました（既存を使用）"
    
    # requirements.txt が更新されていたらインストールが必要
    if [ -f "$REQUIREMENTS_FILE" ]; then
        if [ ! -f "$INSTALLED_FLAG" ] || [ "$REQUIREMENTS_FILE" -nt "$INSTALLED_FLAG" ]; then
            NEED_INSTALL=1
        fi
    fi
else
    echo "⏳ 仮想環境 '$VENV_NAME' を作成中..."
    
    # Python3を探す
    if command -v python3.11 &> /dev/null; then
        PYTHON_CMD="python3.11"
    elif command -v python3 &> /dev/null; then
        PYTHON_CMD="python3"
    else
        echo "❌ Python3が見つかりません。インストールしてください。"
        exit 1
    fi
    
    echo "ℹ️  使用するPython: $PYTHON_CMD"
    $PYTHON_CMD -m venv "$VENV_PATH"
    echo "✅ 仮想環境 '$VENV_NAME' を作成しました"
    NEED_INSTALL=1
fi

# 仮想環境のPython
VENV_PYTHON="$VENV_PATH/bin/python"

# 必要ならパッケージをインストール
if [ "$NEED_INSTALL" -eq 1 ] && [ -f "$REQUIREMENTS_FILE" ]; then
    echo "⏳ 依存パッケージをインストール中..."
    "$VENV_PYTHON" -m pip install --upgrade pip -q
    "$VENV_PYTHON" -m pip install -r "$REQUIREMENTS_FILE"
    touch "$INSTALLED_FLAG"
    echo "✅ 依存パッケージのインストール完了"
fi

echo ""
echo "✅ アプリケーションを起動中..."
echo "------------------------------------------------"

# main.pyを実行
cd "$SCRIPT_DIR"
exec "$VENV_PYTHON" "$MAIN_SCRIPT"
