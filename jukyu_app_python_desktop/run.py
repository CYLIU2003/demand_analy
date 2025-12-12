# -*- coding: utf-8 -*-
"""
電力需給データ分析ツール - ランチャースクリプト

このスクリプトは以下を行います:
1. 仮想環境 ".demand_analy" の存在を確認
2. なければ自動で作成（Python 3.11推奨）
3. 必要なパッケージをインストール
4. main.py を仮想環境で実行
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

# 設定
VENV_NAME = ".demand_analy"
REQUIRED_PYTHON_VERSION = (3, 11)  # 推奨バージョン
REQUIREMENTS_FILE = "requirements.txt"


def get_user_home() -> Path:
    """ユーザーのホームディレクトリを取得"""
    return Path.home()


def get_project_root() -> Path:
    """プロジェクトルートディレクトリを取得"""
    return Path(__file__).resolve().parent


def get_venv_path() -> Path:
    """仮想環境のパスを取得（ユーザーホームディレクトリに配置）"""
    return get_user_home() / VENV_NAME


def get_venv_python() -> Path:
    """仮想環境のPython実行ファイルパスを取得"""
    venv_path = get_venv_path()
    if platform.system() == "Windows":
        return venv_path / "Scripts" / "python.exe"
    else:
        return venv_path / "bin" / "python"


def get_venv_pip() -> Path:
    """仮想環境のpip実行ファイルパスを取得"""
    venv_path = get_venv_path()
    if platform.system() == "Windows":
        return venv_path / "Scripts" / "pip.exe"
    else:
        return venv_path / "bin" / "pip"


def print_status(message: str, status: str = "INFO") -> None:
    """ステータスメッセージを表示"""
    icons = {
        "INFO": "ℹ️",
        "OK": "✅",
        "WARN": "⚠️",
        "ERROR": "❌",
        "WAIT": "⏳",
    }
    icon = icons.get(status, "•")
    print(f"{icon} {message}")


def check_python_version() -> bool:
    """現在のPythonバージョンをチェック"""
    current = sys.version_info[:2]
    print_status(f"現在のPythonバージョン: {sys.version}", "INFO")
    
    if current < REQUIRED_PYTHON_VERSION:
        print_status(
            f"Python {REQUIRED_PYTHON_VERSION[0]}.{REQUIRED_PYTHON_VERSION[1]} 以上が必要です",
            "ERROR"
        )
        return False
    return True


def find_python_311() -> str | None:
    """使用するPython実行ファイルを探す。

    可能であれば 3.11 を優先しますが、見つからない場合は
    "現在このスクリプトを実行している Python" をそのまま使います。

    これにより、Python 3.10 など 3.11 未満の環境でも
    仮想環境 .demand_analy を自動作成して実行できるようにします。
    """

    # まずは各OSで "python3.11" などを優先的に探す
    if platform.system() == "Windows":
        # Windows: py ランチャーを試す
        try:
            result = subprocess.run(
                ["py", "-3.11", "--version"],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                return "py -3.11"
        except FileNotFoundError:
            pass

        # 一般的なインストールパス
        possible_paths = [
            Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Python" / "Python311" / "python.exe",
            Path("C:/Python311/python.exe"),
            Path("C:/Program Files/Python311/python.exe"),
        ]
        for p in possible_paths:
            if p.exists():
                return str(p)
    else:
        # Linux / macOS
        for name in ["python3.11", "python3"]:
            try:
                result = subprocess.run(
                    [name, "--version"],
                    capture_output=True,
                    text=True,
                )
                if result.returncode == 0:
                    return name
            except FileNotFoundError:
                continue

    # 上記で見つからなかった場合は、実行中の Python をそのまま使う
    print_status(
        f"Python 3.11 が見つかりません。現在の Python {sys.version_info[0]}.{sys.version_info[1]} を使用します",
        "WARN",
    )
    return sys.executable


def venv_exists() -> bool:
    """仮想環境が存在するかチェック"""
    venv_python = get_venv_python()
    return venv_python.exists()


def create_venv() -> bool:
    """仮想環境を作成"""
    venv_path = get_venv_path()
    
    print_status(f"仮想環境 '{VENV_NAME}' を作成中...", "WAIT")
    
    # 適切なPythonを探す
    python_cmd = find_python_311()
    if not python_cmd:
        print_status("Python 3.11〜3.13 が見つかりません。インストールしてください。", "ERROR")
        return False
    
    try:
        # py ランチャーの場合
        if python_cmd.startswith("py "):
            cmd = python_cmd.split() + ["-m", "venv", str(venv_path)]
        else:
            cmd = [python_cmd, "-m", "venv", str(venv_path)]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print_status(f"仮想環境の作成に失敗: {result.stderr}", "ERROR")
            return False
        
        print_status(f"仮想環境 '{VENV_NAME}' を作成しました", "OK")
        return True
        
    except Exception as e:
        print_status(f"仮想環境の作成中にエラー: {e}", "ERROR")
        return False


def install_requirements() -> bool:
    """必要なパッケージをインストール"""
    requirements_path = get_project_root() / REQUIREMENTS_FILE
    
    if not requirements_path.exists():
        print_status(f"{REQUIREMENTS_FILE} が見つかりません", "WARN")
        return True
    
    print_status("依存パッケージをインストール中...", "WAIT")
    
    venv_python = get_venv_python()
    
    try:
        # まずpipをアップグレード（python -m pipを使用）
        subprocess.run(
            [str(venv_python), "-m", "pip", "install", "--upgrade", "pip"],
            capture_output=True,
            check=False  # エラーでも続行
        )
        
        # requirements.txtからインストール
        result = subprocess.run(
            [str(venv_python), "-m", "pip", "install", "-r", str(requirements_path)],
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            print_status(f"パッケージのインストールに失敗:\n{result.stderr}", "ERROR")
            return False
        
        print_status("依存パッケージのインストール完了", "OK")
        return True
        
    except subprocess.CalledProcessError as e:
        print_status(f"パッケージのインストール中にエラー: {e}", "ERROR")
        return False


def check_packages_installed() -> bool:
    """必要なパッケージがインストール済みかチェック"""
    requirements_path = get_project_root() / REQUIREMENTS_FILE
    
    if not requirements_path.exists():
        return True
    
    venv_python = get_venv_python()
    
    try:
        # pip listで確認
        result = subprocess.run(
            [str(venv_python), "-m", "pip", "list", "--format=freeze"],
            capture_output=True,
            text=True
        )
        
        installed = set()
        for line in result.stdout.strip().split("\n"):
            if "==" in line:
                pkg = line.split("==")[0].lower()
                installed.add(pkg)
        
        # requirements.txtの内容を確認
        with open(requirements_path, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    pkg = line.split("==")[0].split(">=")[0].split("<=")[0].lower()
                    if pkg not in installed:
                        return False
        
        return True
        
    except Exception:
        return False


def run_main() -> int:
    """main.pyを実行"""
    main_path = get_project_root() / "main.py"
    venv_python = get_venv_python()
    
    if not main_path.exists():
        print_status("main.py が見つかりません", "ERROR")
        return 1
    
    print_status("アプリケーションを起動中...", "OK")
    print("-" * 50)
    
    # 仮想環境のPythonでmain.pyを実行
    result = subprocess.run(
        [str(venv_python), str(main_path)],
        cwd=str(get_project_root())
    )
    
    return result.returncode


def main() -> int:
    """メイン処理"""
    print("=" * 50)
    print("🔌 電力需給データ分析ツール - ランチャー")
    print("=" * 50)
    print()
    
    project_root = get_project_root()
    print_status(f"プロジェクトディレクトリ: {project_root}", "INFO")
    
    # 仮想環境の確認
    if venv_exists():
        print_status(f"仮想環境 '{VENV_NAME}' を検出しました", "OK")
        
        # パッケージの確認
        if not check_packages_installed():
            print_status("一部のパッケージが不足しています", "WARN")
            if not install_requirements():
                return 1
    else:
        print_status(f"仮想環境 '{VENV_NAME}' が見つかりません", "WARN")
        
        # 仮想環境を作成
        if not create_venv():
            return 1
        
        # パッケージをインストール
        if not install_requirements():
            return 1
    
    print()
    
    # アプリケーション実行
    return run_main()


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n中断されました")
        sys.exit(130)
    except Exception as e:
        print_status(f"予期せぬエラー: {e}", "ERROR")
        import traceback
        traceback.print_exc()
        sys.exit(1)
