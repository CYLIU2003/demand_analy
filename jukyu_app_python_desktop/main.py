# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import re
import sys
import webbrowser
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from ml import DemandTransformerForecaster, ForecastResult, TRANSFORMER_AVAILABLE

# 日本語フォントの設定
plt.rcParams['font.sans-serif'] = ['MS Gothic', 'Yu Gothic', 'Meiryo', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

@dataclass(frozen=True)
class AreaInfo:
    # Metadata describing a supply-demand area.

    name: str
    url: str


AreaCode = str
YearMonth = str  # formatted as YYYYMM
DataFileEntry = Tuple[YearMonth, AreaCode, Path]


AREA_INFO: Dict[AreaCode, AreaInfo] = {
    "01": AreaInfo("北海道", "https://www.hepco.co.jp/network/con_service/public_document/supply_demand_results/index.html"),
    "02": AreaInfo("東北", "https://setsuden.nw.tohoku-epco.co.jp/download.html"),
    "03": AreaInfo("東京", "https://www.tepco.co.jp/forecast/html/area_jukyu-j.html"),
    "04": AreaInfo("中部", "https://powergrid.chuden.co.jp/denkiyoho/#link02"),
    "05": AreaInfo("北陸", "https://www.rikuden.co.jp/nw/denki-yoho/results_jyukyu.html"),
    "06": AreaInfo("関西", "https://www.kansai-td.co.jp/denkiyoho/area-performance/index.html"),
    "07": AreaInfo("中国", "https://www.energia.co.jp/nw/jukyuu/eria_jukyu.html"),
    "08": AreaInfo("四国", "https://www.yonden.co.jp/nw/supply_demand/data_download.html"),
    "09": AreaInfo("九州", "https://www.kyuden.co.jp/td_area_jukyu/jukyu.html"),
    "10": AreaInfo("沖縄", "https://www.okiden.co.jp/business-support/service/supply-and-demand/"),
}

# エリアコードと気象庁観測地点（代表都市）の対応
# ファイル名マッチングに使用: {city_name}_YYYYMMDD_YYYYMMDD.csv
AREA_WEATHER_MAP: Dict[AreaCode, str] = {
    "01": "sapporo",   # 北海道
    "02": "sendai",    # 東北
    "03": "tokyo",     # 東京
    "04": "nagoya",    # 中部
    "05": "toyama",    # 北陸
    "06": "osaka",     # 関西
    "07": "hiroshima", # 中国
    "08": "takamatsu", # 四国
    "09": "fukuoka",   # 九州
    "10": "naha",      # 沖縄
}

FNAME = re.compile(r"^eria_jukyu_(\d{6})_(\d{2})\.csv$")
DATA_DIR = Path(__file__).resolve().parent / "data"
WEATHER_DIR = DATA_DIR / "weather"


@dataclass
class GraphSnapshot:
    # Metadata and settings describing a saved graph preview.

    area_code: AreaCode
    area_name: str
    year_month: YearMonth
    date_label: str
    columns: List[str]
    settings: Dict[str, Any]

# 発電種別カテゴリ（比較タブ用）
GENERATION_CATEGORIES = {
    "原子力": ["原子力"],
    "火力": ["火力(LNG)", "火力(石炭)", "火力(石油)", "火力(その他)", "火力"],
    "水力": ["水力"],
    "地熱": ["地熱"],
    "バイオマス": ["バイオマス"],
    "太陽光": ["太陽光発電実績", "太陽光"],
    "風力": ["風力発電実績", "風力"],
    "揚水": ["揚水"],
}

# 需給実績カテゴリ（比較タブ用）
DEMAND_SUPPLY_CATEGORIES = {
    "エリア需要": ["エリア需要"],
    "エリア供給": ["供給力合計", "エリア供給"],
    "連系線": ["連系線"],
    "揚水動力": ["揚水動力"],
}

def scan_files() -> List[DataFileEntry]:
    # Return chronological list of CSV files present in the data directory.

    rows: List[DataFileEntry] = []
    if not DATA_DIR.exists():
        return rows
    for path in sorted(DATA_DIR.iterdir()):
        match = FNAME.match(path.name)
        if match:
            year_month, area_code = match.group(1), match.group(2)
            rows.append((year_month, area_code, path))
    return rows


def build_year_month_range(all_ym: Sequence[YearMonth]) -> Tuple[List[int], List[int]]:
    # Build inclusive year and month ranges from YYYYMM strings.

    if not all_ym:
        current_year = datetime.now().year
        return [current_year], list(range(1, 13))
    year_candidates = sorted({int(ym[:4]) for ym in all_ym})
    years = list(range(min(year_candidates), max(year_candidates) + 1))
    months = list(range(1, 13))
    return years, months


def build_availability(
    files: Sequence[DataFileEntry],
) -> Tuple[Dict[AreaCode, Dict[int, Dict[int, bool]]], List[int], List[int]]:
    """Return availability map and the discovered year, month ranges."""

    availability: Dict[AreaCode, Dict[int, Dict[int, bool]]] = {code: {} for code in AREA_INFO.keys()}
    yms = [year_month for (year_month, _area, _path) in files]
    years, months = build_year_month_range(yms)
    for area_code in availability:
        for year in years:
            availability[area_code][year] = {month: False for month in months}
    for year_month, area_code, _ in files:
        year = int(year_month[:4])
        month = int(year_month[4:6])
        if area_code in availability and year in availability[area_code] and month in availability[area_code][year]:
            availability[area_code][year][month] = True
    return availability, years, months


def read_csv(path: Path) -> Tuple[pd.DataFrame, Optional[str]]:
    # Read a CSV file using several common Japanese encodings and return (data, time-column).

    encodings = ["shift_jis", "cp932", "utf-8", "utf-8-sig"]  # try Shift_JIS variants first
    df: Optional[pd.DataFrame] = None
    for encoding in encodings:
        try:
            df = pd.read_csv(path, encoding=encoding, engine="python", skiprows=0)
            # Skip one header row when the first column stores unit metadata.
            if "単位" in str(df.columns[0]) or "MW" in str(df.columns[0]):
                df = pd.read_csv(path, encoding=encoding, engine="python", skiprows=1)
            break
        except (UnicodeDecodeError, Exception):
            continue
    if df is None:
        raise ValueError(f"ファイルを読み込めませんでした: {path}")

    date_col: Optional[str] = None
    time_col: Optional[str] = None
    for column in df.columns:
        column_upper = str(column).upper()
        if "DATE" in column_upper or "日付" in str(column):
            date_col = column
        if "TIME" in column_upper or "時刻" in str(column) or "時間" in str(column):
            time_col = column

    detected_time_column: Optional[str] = None
    if date_col and time_col:
        try:
            df["datetime"] = pd.to_datetime(
                df[date_col].astype(str) + " " + df[time_col].astype(str),
                errors="coerce",
            )
            if df["datetime"].notna().sum() > 0:
                detected_time_column = "datetime"
        except Exception:
            pass

    for column in df.columns:
        if column not in {detected_time_column, date_col, time_col}:
            try:
                df[column] = pd.to_numeric(df[column], errors="coerce")
            except Exception:
                continue

    if not detected_time_column:
        for lookup in ["datetime", "date", "time", "日時"]:
            for column in df.columns:
                if lookup.lower() in str(column).lower():
                    try:
                        parsed = pd.to_datetime(df[column], errors="coerce")
                        if parsed.notna().sum() > 0:
                            df[column] = parsed
                            detected_time_column = column
                            break
                    except Exception:
                        continue
            if detected_time_column:
                break

    if not detected_time_column:
        first_column = df.columns[0]
        try:
            parsed = pd.to_datetime(df[first_column], errors="coerce")
            if parsed.notna().sum() > 0:
                df[first_column] = parsed
                detected_time_column = first_column
        except Exception:
            pass

    return df, detected_time_column


def read_weather_csv(path: Path) -> pd.DataFrame:
    """
    気象庁の時別値CSVを読み込んで整形する
    
    フォーマット例:
    - 行1: ダウンロード時刻
    - 行2: 地名ヘッダー（東京,東京,...）
    - 行3: カラム名（年月日時,気温(℃),気温(℃),...,天気,天気,天気）
    - 行4: サブヘッダー（品質情報,均質番号,...）
    - 行5以降: データ本体
    
    天気コード（気象庁）:
    1: 快晴, 2: 晴れ, 3: 薄曇, 4: 曇, 5: 煙霧, 8: 霧, 9: 霧雨,
    10: 雨, 11: みぞれ, 12: 雪, 13: あられ, 14: ひょう, 15: 雷,
    16: しゅう雨, 17: 着氷性の雨, 18: 着氷性の霧雨, 19: しゅう雪,
    22: 霧雪, 23: 凍雨, 24: 細氷, 28: もや, 101: 降水
    """
    # 天気コードから天気名への変換テーブル
    WEATHER_CODES = {
        1: "快晴", 2: "晴れ", 3: "薄曇", 4: "曇", 5: "煙霧",
        6: "砂じん嵐", 7: "地ふぶき", 8: "霧", 9: "霧雨", 10: "雨",
        11: "みぞれ", 12: "雪", 13: "あられ", 14: "ひょう", 15: "雷",
        16: "しゅう雨", 17: "着氷性の雨", 18: "着氷性の霧雨", 19: "しゅう雪",
        22: "霧雪", 23: "凍雨", 24: "細氷", 28: "もや", 101: "降水"
    }
    
    try:
        # CP932で読み込み、最初の4行をスキップ（shift_jisより広い文字セット）
        try:
            df = pd.read_csv(path, encoding="cp932", skiprows=4, engine="python")
        except UnicodeDecodeError:
            # フォールバック: UTF-8を試す
            df = pd.read_csv(path, encoding="utf-8", skiprows=4, engine="python")
        
        # 最初の列が日時
        datetime_col = df.columns[0]
        df = df.rename(columns={datetime_col: "datetime"})
        
        # datetimeに変換
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
        
        # 必要な列を抽出（各項目の最初の値のみ取得）
        # カラム構成: 年月日時, 気温, 品質, 均質, 降水量, 現象なし, 品質, 均質, ...
        weather_data = pd.DataFrame()
        weather_data["datetime"] = df["datetime"]
        
        # 気温 (2列目, インデックス1)
        if len(df.columns) > 1:
            weather_data["temperature"] = pd.to_numeric(df.iloc[:, 1], errors="coerce")
        
        # 降水量 (5列目, インデックス4)
        if len(df.columns) > 4:
            weather_data["precipitation"] = pd.to_numeric(df.iloc[:, 4], errors="coerce")
        
        # 日照時間 (9列目, インデックス8)
        if len(df.columns) > 8:
            weather_data["sunlight"] = pd.to_numeric(df.iloc[:, 8], errors="coerce")
        
        # 風速 (13列目, インデックス12)
        if len(df.columns) > 12:
            weather_data["wind_speed"] = pd.to_numeric(df.iloc[:, 12], errors="coerce")
        
        # 風向 (15列目, インデックス14) - 文字列
        if len(df.columns) > 14:
            weather_data["wind_direction"] = df.iloc[:, 14]
        
        # 日射量 (18列目, インデックス17)
        if len(df.columns) > 17:
            weather_data["solar_radiation"] = pd.to_numeric(df.iloc[:, 17], errors="coerce")
        
        # 天気 (21列目, インデックス20) - 新規追加
        if len(df.columns) > 20:
            weather_code = pd.to_numeric(df.iloc[:, 20], errors="coerce")
            weather_data["weather_code"] = weather_code
            # コードを天気名に変換
            weather_data["weather"] = weather_code.apply(
                lambda x: WEATHER_CODES.get(int(x), "不明") if pd.notna(x) else ""
            )
        
        # NaNを含む行を削除
        weather_data = weather_data.dropna(subset=["datetime"])
        
        return weather_data
        
    except Exception as e:
        raise ValueError(f"天候データの読み込みに失敗しました: {path}\n{str(e)}")


def find_weather_file(area_code: AreaCode, year_month: YearMonth) -> Optional[Path]:
    """
    指定されたエリアと年月に対応する天候CSVファイルを検索
    
    対応するファイル名形式:
    1. 月別ファイル: {city}_YYYYMM.csv (例: tokyo_202501.csv)
    2. 期間ファイル: {city}_YYYYMMDD_YYYYMMDD.csv (例: tokyo_20250101_20250331.csv)
    """
    if not WEATHER_DIR.exists():
        print(f"[DEBUG] 天候データディレクトリが存在しません: {WEATHER_DIR}")
        return None
    
    city_name = AREA_WEATHER_MAP.get(area_code)
    if not city_name:
        print(f"[DEBUG] エリアコード {area_code} に対応する都市名が見つかりません")
        return None
    
    print(f"[DEBUG] エリア {area_code} → 都市: {city_name}, 対象年月: {year_month}")
    
    year = int(year_month[:4])
    month = int(year_month[4:6])
    
    # 該当する期間を含むファイルを探す
    available_files = list(WEATHER_DIR.glob(f"{city_name}_*.csv"))
    print(f"[DEBUG] 検索パターン: {city_name}_*.csv")
    print(f"[DEBUG] 見つかったファイル: {[f.name for f in available_files]}")
    
    for weather_file in available_files:
        try:
            filename = weather_file.stem
            parts = filename.split("_")
            print(f"[DEBUG] ファイル解析: {filename} → parts={parts}")
            
            # パターン1: 月別ファイル (city_YYYYMM.csv)
            if len(parts) == 2 and len(parts[1]) == 6:
                file_year_month = parts[1]  # YYYYMM
                if file_year_month == year_month:
                    print(f"[DEBUG] 月別ファイルがマッチしました: {weather_file.name}")
                    return weather_file
            
            # パターン2: 期間ファイル (city_YYYYMMDD_YYYYMMDD.csv)
            elif len(parts) >= 3:
                start_date_str = parts[1]  # YYYYMMDD
                end_date_str = parts[2]    # YYYYMMDD
                
                # 日付形式チェック（8桁）
                if len(start_date_str) == 8 and len(end_date_str) == 8:
                    start_year = int(start_date_str[:4])
                    start_month = int(start_date_str[4:6])
                    end_year = int(end_date_str[:4])
                    end_month = int(end_date_str[4:6])
                    
                    print(f"[DEBUG] 期間: {start_year}/{start_month:02d} ～ {end_year}/{end_month:02d}, 対象: {year}/{month:02d}")
                    
                    # 指定された年月がファイルの期間内か確認
                    if (start_year < year or (start_year == year and start_month <= month)) and \
                       (end_year > year or (end_year == year and end_month >= month)):
                        print(f"[DEBUG] 期間ファイルがマッチしました: {weather_file.name}")
                        return weather_file
        except (ValueError, IndexError) as e:
            print(f"[DEBUG] ファイル解析エラー: {filename} - {e}")
            continue
    
    print(f"[DEBUG] 該当する天候ファイルが見つかりませんでした")
    return None


def merge_weather_data(df_power: pd.DataFrame, area_code: AreaCode, year_month: YearMonth, 
                       time_column: Optional[str] = None) -> Tuple[pd.DataFrame, bool]:
    """
    電力データに天候データを結合する
    
    Returns:
        (merged_df, has_weather): 結合後のDataFrameと天候データが存在したかのフラグ
    """
    # 天候ファイルを検索
    weather_file = find_weather_file(area_code, year_month)
    
    if not weather_file:
        return df_power, False
    
    try:
        # 天候データを読み込み
        df_weather = read_weather_csv(weather_file)
        
        # 電力データに日時列がない場合はそのまま返す
        if time_column is None or time_column not in df_power.columns:
            return df_power, False
        
        # 電力データの時刻列をdatetimeに変換（まだの場合）
        if not pd.api.types.is_datetime64_any_dtype(df_power[time_column]):
            df_power[time_column] = pd.to_datetime(df_power[time_column], errors="coerce")
        
        # 日時をキーにして左外部結合
        df_merged = pd.merge(
            df_power, 
            df_weather, 
            left_on=time_column, 
            right_on="datetime", 
            how="left",
            suffixes=("", "_weather")
        )
        
        # 重複した"datetime"列を削除
        if "datetime_weather" in df_merged.columns:
            df_merged = df_merged.drop(columns=["datetime_weather"])
        
        return df_merged, True
        
    except Exception as e:
        print(f"天候データの結合に失敗: {str(e)}")
        return df_power, False


class MplCanvas(FigureCanvas):
    # Thin matplotlib canvas wrapper that exposes the Axes for plotting.

    def __init__(self, width: float = 12, height: float = 6, dpi: int = 100) -> None:
        self.fig: Figure = Figure(figsize=(width, height), dpi=dpi, facecolor="#ffffff")
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor("#f8fafc")
        self.ax.tick_params(colors="#2d3748", labelsize=10)
        self.ax.spines["bottom"].set_color("#a0d2ff")
        self.ax.spines["top"].set_color("#a0d2ff")
        self.ax.spines["left"].set_color("#a0d2ff")
        self.ax.spines["right"].set_color("#a0d2ff")
        self.fig.tight_layout(pad=2.0)
        super().__init__(self.fig)

    def update_size(self, width: float, height: float, dpi: int) -> None:
        # Adjust the canvas size and trigger redraw.

        self.fig.set_size_inches(width, height)
        self.fig.set_dpi(dpi)
        self.draw()


class GraphCard(QFrame):
    # Reusable widget that displays a saved graph alongside metadata.

    def __init__(self, parent: "MainWindow", snapshot: GraphSnapshot, remove_callback) -> None:
        super().__init__(parent)
        self.setObjectName("GraphCard")
        self.snapshot = snapshot
        self.remove_callback = remove_callback
        self.canvas = MplCanvas(
            width=snapshot.settings.get("figsize_w", 12),
            height=snapshot.settings.get("figsize_h", 6),
            dpi=snapshot.settings.get("dpi", 100),
        )
        self._build_ui()

    def _build_ui(self) -> None:
        self.setStyleSheet(
            """
            QFrame#GraphCard {
                border: 2px solid #a0d2ff;
                border-radius: 10px;
                background-color: #ffffff;
            }
            """
        )
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        header = QHBoxLayout()
        header.setSpacing(10)
        
        # 重ね合わせ用チェックボックスを追加
        self.overlay_checkbox = QCheckBox("重ね合わせに含める")
        self.overlay_checkbox.setStyleSheet("""
            QCheckBox {
                font-weight: 600;
                color: #0068B7;
                font-size: 14px;
            }
            QCheckBox::indicator {
                width: 20px;
                height: 20px;
            }
        """)
        header.addWidget(self.overlay_checkbox)
        
        header.addStretch()
        self.tag_label = QLabel("")
        self.tag_label.setStyleSheet("color: #4a5568;")
        header.addWidget(self.tag_label)
        layout.addLayout(header)

        self.title_label = QLabel("")
        self.title_label.setWordWrap(True)
        self.title_label.setStyleSheet(
            "color: #0068B7; font-weight: 700; font-size: 16px;"
        )
        layout.addWidget(self.title_label)

        self.meta_label = QLabel(self._format_meta_text())
        self.meta_label.setWordWrap(True)
        self.meta_label.setStyleSheet("color: #2d3748;")
        layout.addWidget(self.meta_label)

        layout.addWidget(self.canvas)

        button_row = QHBoxLayout()
        button_row.addStretch()
        save_btn = QPushButton("💾 このグラフを保存")
        save_btn.clicked.connect(lambda _checked=False: self.save_snapshot())
        remove_btn = QPushButton("🗑 削除")
        remove_btn.clicked.connect(lambda _checked=False: self.remove_callback(self))
        button_row.addWidget(save_btn)
        button_row.addWidget(remove_btn)
        layout.addLayout(button_row)

    def _format_meta_text(self) -> str:
        """グラフのメタデータをテキスト形式で整形"""
        meta = []
        meta.append(f"エリア: {self.snapshot.area_name}")
        meta.append(f"年月: {self.snapshot.year_month[:4]}年{self.snapshot.year_month[4:6]}月")
        if self.snapshot.date_label:
            meta.append(f"日付: {self.snapshot.date_label}")
        
        cols = self.snapshot.columns
        if len(cols) > 3:
            col_text = ", ".join(cols[:3]) + "..."
        else:
            col_text = ", ".join(cols)
        meta.append(f"項目: {col_text}")
        
        return "\n".join(meta)

    def save_snapshot(self) -> None:
        # グラフを画像として保存
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "グラフを保存",
            "",
            "PNG (*.png);;PDF (*.pdf);;SVG (*.svg)"
        )

        if filename:
            dpi = int(self.snapshot.settings.get("dpi", 100))
            self.canvas.fig.savefig(filename, dpi=dpi, bbox_inches='tight')
            QtWidgets.QMessageBox.information(self, "成功", f"グラフを保存しました:\n{filename}")

    def set_plot_title(self, title: str) -> None:
        """カード表示用のタイトルを設定する。"""
        title = title or ""
        self.title_label.setText(title)
        # Keep snapshot settings in sync for future operations.
        self.snapshot.settings["title"] = title

class MainWindow(QMainWindow):
    """メインウィンドウクラス - 需給データ分析アプリケーション"""
    
    def __init__(self) -> None:
        print("Initializing MainWindow...")
        super().__init__()
        print("super().__init__ done")
        self.setWindowTitle("電力需給データ分析ツール")
        
        # 画面サイズに合わせてウィンドウサイズを自動調整
        screen = QApplication.primaryScreen()
        if screen:
            screen_size = screen.availableGeometry()
            # 画面の80%のサイズで起動、最大1600x900
            width = min(int(screen_size.width() * 0.8), 1600)
            height = min(int(screen_size.height() * 0.85), 900)
            # 画面中央に配置
            x = (screen_size.width() - width) // 2
            y = (screen_size.height() - height) // 2
            self.setGeometry(x, y, width, height)
        else:
            self.setGeometry(100, 100, 1400, 800)
        
        # データ管理
        self.files: list[Tuple[YearMonth, AreaCode, Path]] = []
        self.avail: dict[AreaCode, dict[int, dict[int, bool]]] = {}
        self.years: list[int] = []
        self.months: list[int] = []
        self.area_data: dict[AreaCode, list[str]] = {}
        self.area_year_months: dict[AreaCode, list[YearMonth]] = {}
        
        # 現在のデータフレーム
        self.current_dataframe: Optional[pd.DataFrame] = None
        self.current_time_column: Optional[str] = None
        self.current_dataset_key: Optional[str] = None
        
        # 選択状態
        self.selected_columns: list[str] = []
        # データセットごとの列選択キャッシュ {(area_code, year_month): [cols...]}
        self.column_selection_cache: dict[tuple[str, str], list[str]] = {}
        
        # グラフ設定
        self.graph_settings = {
            "title": "",
            "xlabel": "時刻",
            "ylabel": "電力 (kW)",
            "linewidth": 2.0,
            "font_size": 12,
            "title_size": 16,
            "label_size": 12,
            "grid": True,
            "legend": True,
            "legend_loc": "best",
            "figsize_w": 12,
            "figsize_h": 6,
            "dpi": 100,
            "detailed_ticks": False,
            "show_title": True,
            "show_xlabel": True,
            "show_ylabel": True,
            "show_legend": True,
            # 研究発表用の追加オプション
            "show_weekday": False,  # 曜日を表示
            "weekday_format": "short",  # "short": (月), "full": 月曜日, "en_short": Mon
            "show_week_boundaries": False,  # 週の境界線を表示
            "week_boundary_day": "monday",  # 週の始まり: monday or sunday
            "midnight_label_format": "next_day",  # "next_day": 翌日0:00, "same_day": 当日24:00
            # 横軸日付表示のカスタマイズ
            "xaxis_date_filter": "all",  # "all", "specific_weekdays", "every_n_days", "custom"
            "xaxis_weekdays": [0, 1, 2, 3, 4, 5, 6],  # 表示する曜日 (0=月, 6=日)
            "xaxis_every_n_days": 1,  # N日おきに表示
            "xaxis_custom_dates": [],  # カスタム日付リスト (YYYY-MM-DD形式)
            "xaxis_tick_rotation": 45,  # X軸ラベルの回転角度
        }
        
        # グラフコレクション
        self.graph_snapshots: list[GraphSnapshot] = []
        self.graph_collection_widgets: list[GraphCard] = []
        
        # AI分析用データ
        self.ai_dataframe: Optional[pd.DataFrame] = None
        self.ai_time_column: Optional[str] = None
        
        # UI構築
        print("Setting up UI...")
        self._setup_ui()
        print("UI setup done")
        
        # 初期データスキャン
        print("Scanning files...")
        self.files = scan_files()
        self.avail, self.years, self.months = build_availability(self.files)
        self.refresh_area_year_months()
        self.populate_area_combos()
        self.populate_ai_controls()
        self.populate_comp_controls()
        print("Initialization complete")
    
    def _setup_ui(self) -> None:
        """UI全体を構築"""
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        # タブウィジェット
        self.tabs = QTabWidget()
        self.tabs.setTabPosition(QTabWidget.North)
        self.tabs.setStyleSheet(
            "QTabBar::tab {\n"
            "    background: #e6f2ff;\n"
            "    color: #0068B7;\n"
            "    border: 1px solid #a0d2ff;\n"
            "    padding: 10px 20px;\n"
            "    border-top-left-radius: 8px;\n"
            "    border-top-right-radius: 8px;\n"
            "    margin-right: 2px;\n"
            "}\n"
            "QTabBar::tab:selected {\n"
            "    background: #0068B7;\n"
            "    color: white;\n"
            "}\n"
        )
        
        # メインページ
        main_page = self.create_main_page()
        self.tabs.addTab(main_page, "📊 メイン")
        
        # AI分析ページ
        ai_page = self.create_ai_page()
        self.tabs.addTab(ai_page, "🤖 統計分析・予測")
        
        # データ可用性ページ
        avail_page = self.create_availability_page()
        self.tabs.addTab(avail_page, "📅 データ可用性")

        # 発電種別比較ページ
        comp_page = self.create_comparison_page()
        self.tabs.addTab(comp_page, "⚡ 発電種別比較")
        
        main_layout.addWidget(self.tabs)
        
        # スタイル適用
        self.apply_theme()

    def create_main_page(self) -> QWidget:
        """メインページ (データ選択・グラフ表示・コレクション) を構築"""
        page = QWidget()
        layout = QHBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # 左パネル: データ選択
        data_panel = self.create_data_selection_panel()
        data_panel.setMaximumWidth(400)
        
        # 中央パネル: グラフ設定
        settings_panel = self.create_graph_settings_panel()
        settings_panel.setMaximumWidth(350)
        
        # 右パネル: タブでプレビュー、グラフ、コレクションを分離
        right_tabs = QTabWidget()
        right_tabs.setStyleSheet(
            "QTabBar::tab {\n"
            "    background: #f0f9ff;\n"
            "    color: #0068B7;\n"
            "    border: 1px solid #bfdbfe;\n"
            "    padding: 8px 16px;\n"
            "    border-top-left-radius: 6px;\n"
            "    border-top-right-radius: 6px;\n"
            "}\n"
            "QTabBar::tab:selected {\n"
            "    background: #0068B7;\n"
            "    color: white;\n"
            "}\n"
        )
        
        # プレビュータブ
        preview_panel = self.create_preview_panel()
        right_tabs.addTab(preview_panel, "📋 データプレビュー")
        
        # グラフ表示タブ
        graph_panel = self.create_graph_only_panel()
        right_tabs.addTab(graph_panel, "📊 グラフ表示")
        
        # 並列比較用コレクションタブ
        collection_panel = self.create_collection_panel()
        right_tabs.addTab(collection_panel, "🗂 並列比較")
        
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(data_panel)
        splitter.addWidget(settings_panel)
        splitter.addWidget(right_tabs)
        splitter.setSizes([400, 350, 1050])
        
        layout.addWidget(splitter)
        return page
    
    def populate_area_combos(self) -> None:
        """全エリアコンボボックスにエリア一覧を入力"""
        if hasattr(self, "area_combo"):
            self.area_combo.blockSignals(True)
            self.area_combo.clear()
            for code, meta in AREA_INFO.items():
                self.area_combo.addItem(f"({code}) {meta.name}", code)
            self.area_combo.blockSignals(False)
            if self.area_combo.count() > 0:
                self.area_combo.setCurrentIndex(0)
                self.on_area_change()
        
        # 年・月の分離版コンボボックスを更新
        if hasattr(self, "year_combo") and hasattr(self, "month_combo"):
            code = self.area_combo.currentData() if hasattr(self, "area_combo") else None
            if code:
                year_months = self.area_year_months.get(code, [])
                
                # 年のリストを取得（重複を除く）
                years = sorted(set([ym[:4] for ym in year_months]))
                
                self.year_combo.blockSignals(True)
                self.year_combo.clear()
                for year in years:
                    self.year_combo.addItem(f"{year}年", year)
                self.year_combo.blockSignals(False)
                
                if self.year_combo.count() > 0:
                    self.year_combo.setCurrentIndex(0)
                    self.update_month_combo()
        
        # 後方互換性のため、ym_comboも更新（非表示だが使用される場合がある）
        if hasattr(self, "ym_combo"):
            code = self.area_combo.currentData() if hasattr(self, "area_combo") else None
            self.ym_combo.blockSignals(True)
            self.ym_combo.clear()
            if code:
                for ym in self.area_year_months.get(code, []):
                    display = f"{ym[:4]}年{ym[4:6]}月"
                    self.ym_combo.addItem(display, ym)
            self.ym_combo.blockSignals(False)
            if self.ym_combo.count() > 0:
                self.ym_combo.setCurrentIndex(0)

    def create_ai_page(self) -> QWidget:
        """統計分析・予測タブを作成"""
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)

        # ヘッダー
        header = QHBoxLayout()
        title = QLabel("📊 統計分析・予測")
        title.setStyleSheet("font-size: 20px; font-weight: 600; color: #0068B7;")
        header.addWidget(title)
        header.addStretch()
        back_btn = QPushButton("← メインに戻る")
        back_btn.setMinimumHeight(36)
        back_btn.clicked.connect(lambda: self.tabs.setCurrentIndex(0))
        header.addWidget(back_btn)
        layout.addLayout(header)

        desc = QLabel("時系列分解、統計検定、機械学習モデルによる需要予測を実行します。")
        desc.setWordWrap(True)
        layout.addWidget(desc)

        # データ選択とコントロール
        control_layout = QHBoxLayout()
        
        control_layout.addWidget(QLabel("エリア:"))
        self.ai_area_combo = QComboBox()
        self.ai_area_combo.setMinimumHeight(32)
        for code, meta in AREA_INFO.items():
            self.ai_area_combo.addItem(f"({code}) {meta.name}", code)
        self.ai_area_combo.currentIndexChanged.connect(self.on_ai_area_change)
        control_layout.addWidget(self.ai_area_combo)

        # 年選択
        control_layout.addWidget(QLabel("年:"))
        self.ai_year_combo = QComboBox()
        self.ai_year_combo.setMinimumHeight(32)
        self.ai_year_combo.currentIndexChanged.connect(self.on_ai_year_month_change)
        control_layout.addWidget(self.ai_year_combo)
        
        # 月選択
        control_layout.addWidget(QLabel("月:"))
        self.ai_month_combo = QComboBox()
        self.ai_month_combo.setMinimumHeight(32)
        self.ai_month_combo.currentIndexChanged.connect(self.on_ai_year_month_change)
        control_layout.addWidget(self.ai_month_combo)
        
        # 後方互換性のため、ai_ym_comboも保持（非表示）
        self.ai_ym_combo = QComboBox()
        self.ai_ym_combo.setVisible(False)

        control_layout.addWidget(QLabel("目的変数"))
        self.ai_column_combo = QComboBox()
        self.ai_column_combo.setMinimumHeight(32)
        control_layout.addWidget(self.ai_column_combo)
        
        control_layout.addStretch()
        layout.addLayout(control_layout)
        
        # 分析手法選択ボタン
        method_layout = QHBoxLayout()
        method_layout.addWidget(QLabel("分析手法"))
        
        self.decompose_btn = QPushButton("時系列分解 (STL)")
        self.decompose_btn.setMinimumHeight(40)
        self.decompose_btn.clicked.connect(self.run_stl_decomposition)
        method_layout.addWidget(self.decompose_btn)
        
        self.arima_btn = QPushButton("📊 ARIMA予測")
        self.arima_btn.setMinimumHeight(40)
        self.arima_btn.clicked.connect(self.run_arima_forecast)
        method_layout.addWidget(self.arima_btn)
        
        self.exp_smooth_btn = QPushButton("📈 指数平滑法")
        self.exp_smooth_btn.setMinimumHeight(40)
        self.exp_smooth_btn.clicked.connect(self.run_exponential_smoothing)
        method_layout.addWidget(self.exp_smooth_btn)
        
        if TRANSFORMER_AVAILABLE:
            self.transformer_btn = QPushButton("🤖 Transformer")
            self.transformer_btn.setMinimumHeight(40)
            self.transformer_btn.clicked.connect(self.run_transformer_forecast)
            method_layout.addWidget(self.transformer_btn)
        
        self.weather_btn = QPushButton("🌤️ 天候分析")
        self.weather_btn.setMinimumHeight(40)
        self.weather_btn.clicked.connect(self.run_weather_analysis)
        method_layout.addWidget(self.weather_btn)
        
        self.weather_forecast_btn = QPushButton("🌡️ 天候予測")
        self.weather_forecast_btn.setMinimumHeight(40)
        self.weather_forecast_btn.setToolTip("天候データを考慮した需要予測 (ARIMAX)")
        self.weather_forecast_btn.clicked.connect(self.run_weather_forecast)
        method_layout.addWidget(self.weather_forecast_btn)
        
        method_layout.addStretch()
        layout.addLayout(method_layout)
        
        # パラメータ設定
        param_layout = QHBoxLayout()
        param_layout.addWidget(QLabel("予測期間:"))
        self.ai_horizon_spin = QSpinBox()
        self.ai_horizon_spin.setRange(1, 240)
        self.ai_horizon_spin.setValue(24)
        self.ai_horizon_spin.setMinimumHeight(32)
        param_layout.addWidget(self.ai_horizon_spin)
        
        param_layout.addWidget(QLabel("訓練データ期間:"))
        self.train_ratio_spin = QDoubleSpinBox()
        self.train_ratio_spin.setRange(0.5, 0.95)
        self.train_ratio_spin.setSingleStep(0.05)
        self.train_ratio_spin.setValue(0.8)
        self.train_ratio_spin.setMinimumHeight(32)
        param_layout.addWidget(self.train_ratio_spin)
        
        param_layout.addStretch()
        layout.addLayout(param_layout)

        # タブウィジェット(分析結果)
        self.ai_tabs = QTabWidget()
        self.ai_tabs.setStyleSheet(
            "QTabWidget::pane {\n"
            "    border: 2px solid #a0d2ff;\n"
            "    border-radius: 8px;\n"
            "    background-color: white;\n"
            "}\n"
            "QTabBar::tab {\n"
            "    background: #f0f0f0;\n"
            "    padding: 8px 16px;\n"
            "    margin-right: 2px;\n"
            "}\n"
            "QTabBar::tab:selected {\n"
            "    background: #0068B7;\n"
            "    color: white;\n"
            "}\n"
        )
        
        # ログタブ
        self.ai_log_output = QPlainTextEdit()
        self.ai_log_output.setReadOnly(True)
        self.ai_log_output.setPlaceholderText("分析ログがここに表示されます...")
        self.ai_tabs.addTab(self.ai_log_output, "ログ")
        
        # 分析結果タブ
        self.ai_result_canvas = MplCanvas(width=12, height=8)
        self.ai_tabs.addTab(self.ai_result_canvas, "予測結果")
        
        # モデル評価タブ
        self.ai_eval_widget = QTextEdit()
        self.ai_eval_widget.setReadOnly(True)
        self.ai_eval_widget.setStyleSheet("font-family: 'Courier New', monospace; padding: 10px;")
        self.ai_tabs.addTab(self.ai_eval_widget, "モデル評価")
        
        # 残差分析タブ
        self.ai_residual_canvas = MplCanvas(width=12, height=6)
        self.ai_tabs.addTab(self.ai_residual_canvas, "残差分析")
        
        # 天候分析タブ
        self.ai_weather_canvas = MplCanvas(width=12, height=6)
        self.ai_tabs.addTab(self.ai_weather_canvas, "天候分析")
        
        # 統計集計タブ
        stats_widget = self._create_stats_summary_widget()
        self.ai_tabs.addTab(stats_widget, "📊 統計集計")
        
        # 需要・天候分析タブ
        demand_weather_widget = self._create_demand_weather_widget()
        self.ai_tabs.addTab(demand_weather_widget, "🌤️ 需要・天候分析")
        
        layout.addWidget(self.ai_tabs)
        
        return page
    
    def _create_stats_summary_widget(self) -> QWidget:
        """統計集計ウィジェットを作成"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)
        
        # コントロール行
        control_layout = QHBoxLayout()
        
        control_layout.addWidget(QLabel("集計期間:"))
        self.stats_period_combo = QComboBox()
        self.stats_period_combo.addItems(["時間別", "日別", "週別", "月別"])
        self.stats_period_combo.setMinimumWidth(100)
        control_layout.addWidget(self.stats_period_combo)
        
        control_layout.addWidget(QLabel("対象列:"))
        self.stats_column_combo = QComboBox()
        self.stats_column_combo.setMinimumWidth(200)
        control_layout.addWidget(self.stats_column_combo)
        
        calc_btn = QPushButton("集計実行")
        calc_btn.setMinimumHeight(32)
        calc_btn.clicked.connect(self.run_stats_summary)
        control_layout.addWidget(calc_btn)
        
        export_btn = QPushButton("📋 クリップボードにコピー")
        export_btn.setMinimumHeight(32)
        export_btn.clicked.connect(self.copy_stats_to_clipboard)
        control_layout.addWidget(export_btn)
        
        control_layout.addStretch()
        layout.addLayout(control_layout)
        
        # 結果表示テーブル
        self.stats_table = QTableWidget()
        self.stats_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.stats_table.setAlternatingRowColors(True)
        self.stats_table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self.stats_table.setStyleSheet("""
            QTableWidget {
                background-color: #ffffff;
                alternate-background-color: #f8fafc;
                border: 1px solid #e2e8f0;
                font-size: 12px;
            }
            QHeaderView::section {
                background-color: #e6f2ff;
                color: #0068B7;
                padding: 8px;
                border: 1px solid #a0d2ff;
                font-weight: 600;
            }
        """)
        layout.addWidget(self.stats_table)
        
        # サマリー表示
        self.stats_summary_label = QLabel("")
        self.stats_summary_label.setWordWrap(True)
        self.stats_summary_label.setStyleSheet("""
            QLabel {
                background-color: #f0f9ff;
                border: 1px solid #bae6fd;
                border-radius: 8px;
                padding: 15px;
                font-size: 13px;
                color: #0c4a6e;
            }
        """)
        layout.addWidget(self.stats_summary_label)
        
        return widget

    def _create_demand_weather_widget(self) -> QWidget:
        """需要・天候分析ウィジェットを作成"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)
        
        # 説明
        desc = QLabel("需要や発電量が高い日・低い日の天候情報と曜日を分析します。")
        desc.setStyleSheet("color: #666; font-size: 12px;")
        layout.addWidget(desc)
        
        # コントロール行1
        control_layout1 = QHBoxLayout()
        control_layout1.addWidget(QLabel("分析対象:"))
        self.dw_column_combo = QComboBox()
        self.dw_column_combo.setMinimumWidth(180)
        control_layout1.addWidget(self.dw_column_combo)
        
        control_layout1.addWidget(QLabel("上位/下位:"))
        self.dw_top_n_spin = QSpinBox()
        self.dw_top_n_spin.setRange(5, 50)
        self.dw_top_n_spin.setValue(10)
        self.dw_top_n_spin.setSuffix(" 日")
        control_layout1.addWidget(self.dw_top_n_spin)
        
        control_layout1.addWidget(QLabel("分析単位:"))
        self.dw_time_unit_combo = QComboBox()
        self.dw_time_unit_combo.addItems(["日別", "時別"])
        self.dw_time_unit_combo.setToolTip("日別: 1日の値で分析\n時別: 時間帯ごとの値で分析")
        control_layout1.addWidget(self.dw_time_unit_combo)
        
        control_layout1.addWidget(QLabel("集計:"))
        self.dw_agg_combo = QComboBox()
        self.dw_agg_combo.addItems(["平均", "合計", "最大"])
        control_layout1.addWidget(self.dw_agg_combo)
        
        analyze_btn = QPushButton("🔍 分析実行")
        analyze_btn.setMinimumHeight(32)
        analyze_btn.setStyleSheet("""
            QPushButton {
                background-color: #0068B7;
                color: white;
                font-weight: bold;
                border-radius: 5px;
                padding: 0 15px;
            }
            QPushButton:hover {
                background-color: #005999;
            }
        """)
        analyze_btn.clicked.connect(self.run_demand_weather_analysis)
        control_layout1.addWidget(analyze_btn)
        
        copy_btn = QPushButton("📋 コピー")
        copy_btn.setMinimumHeight(32)
        copy_btn.clicked.connect(self.copy_demand_weather_result)
        control_layout1.addWidget(copy_btn)
        
        control_layout1.addStretch()
        layout.addLayout(control_layout1)
        
        # メインスプリッター（縦分割）
        main_splitter = QSplitter(Qt.Vertical)
        
        # 上部: テーブル用スプリッター（横分割）
        table_splitter = QSplitter(Qt.Horizontal)
        
        # 左側: 高い日テーブル
        high_frame = QFrame()
        high_layout = QVBoxLayout(high_frame)
        high_layout.setContentsMargins(3, 3, 3, 3)
        high_layout.setSpacing(3)
        high_label = QLabel("📈 高い日 TOP N")
        high_label.setStyleSheet("font-weight: bold; color: #dc2626; font-size: 11px;")
        high_layout.addWidget(high_label)
        
        self.dw_high_table = QTableWidget()
        self.dw_high_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.dw_high_table.setAlternatingRowColors(True)
        self.dw_high_table.setMinimumHeight(150)
        self.dw_high_table.setStyleSheet("""
            QTableWidget {
                background-color: #fff5f5;
                alternate-background-color: #fef2f2;
                border: 1px solid #fecaca;
                font-size: 11px;
            }
            QHeaderView::section {
                background-color: #fecaca;
                color: #991b1b;
                padding: 4px;
                border: 1px solid #f87171;
                font-weight: 600;
                font-size: 10px;
            }
        """)
        high_layout.addWidget(self.dw_high_table)
        table_splitter.addWidget(high_frame)
        
        # 右側: 低い日テーブル
        low_frame = QFrame()
        low_layout = QVBoxLayout(low_frame)
        low_layout.setContentsMargins(3, 3, 3, 3)
        low_layout.setSpacing(3)
        low_label = QLabel("📉 低い日 TOP N")
        low_label.setStyleSheet("font-weight: bold; color: #2563eb; font-size: 11px;")
        low_layout.addWidget(low_label)
        
        self.dw_low_table = QTableWidget()
        self.dw_low_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.dw_low_table.setAlternatingRowColors(True)
        self.dw_low_table.setMinimumHeight(150)
        self.dw_low_table.setStyleSheet("""
            QTableWidget {
                background-color: #eff6ff;
                alternate-background-color: #dbeafe;
                border: 1px solid #bfdbfe;
                font-size: 11px;
            }
            QHeaderView::section {
                background-color: #bfdbfe;
                color: #1e40af;
                padding: 4px;
                border: 1px solid #60a5fa;
                font-weight: 600;
                font-size: 10px;
            }
        """)
        low_layout.addWidget(self.dw_low_table)
        table_splitter.addWidget(low_frame)
        
        # テーブルスプリッターを等分割
        table_splitter.setSizes([500, 500])
        main_splitter.addWidget(table_splitter)
        
        # 下部: サマリー（スクロール可能）
        summary_scroll = QtWidgets.QScrollArea()
        summary_scroll.setWidgetResizable(True)
        summary_scroll.setMinimumHeight(100)
        summary_scroll.setMaximumHeight(200)
        
        self.dw_summary_label = QLabel("分析を実行してください")
        self.dw_summary_label.setWordWrap(True)
        self.dw_summary_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.dw_summary_label.setStyleSheet("""
            QLabel {
                background-color: #fefce8;
                border: none;
                padding: 10px;
                font-size: 11px;
                color: #713f12;
            }
        """)
        summary_scroll.setWidget(self.dw_summary_label)
        summary_scroll.setStyleSheet("""
            QScrollArea {
                border: 1px solid #fde047;
                border-radius: 6px;
                background-color: #fefce8;
            }
        """)
        main_splitter.addWidget(summary_scroll)
        
        # スプリッターの初期サイズ設定
        main_splitter.setSizes([400, 150])
        
        layout.addWidget(main_splitter, stretch=1)
        
        return widget

    def create_availability_page(self) -> QWidget:
        """データ可用性確認ページを作成"""
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)

        # ヘッダー
        header = QHBoxLayout()
        title = QLabel("📊 データ可用性マップ")
        title.setStyleSheet("font-size: 20px; font-weight: 600; color: #0068B7;")
        header.addWidget(title)
        header.addStretch()
        
        # 年選択
        header.addWidget(QLabel("表示年:"))
        self.avail_year_combo = QComboBox()
        self.avail_year_combo.setMinimumWidth(100)
        # 年リストを作成 (2016年から現在+1年まで)
        current_year = datetime.now().year
        years = sorted(list(range(2016, current_year + 2)), reverse=True)
        for y in years:
            self.avail_year_combo.addItem(f"{y}年", y)
        self.avail_year_combo.currentIndexChanged.connect(self.refresh_availability_table)
        header.addWidget(self.avail_year_combo)

        # 更新ボタン
        refresh_btn = QPushButton("🔄 更新")
        refresh_btn.setMinimumHeight(36)
        refresh_btn.clicked.connect(self.on_area_change)
        header.addWidget(refresh_btn)
        
        layout.addLayout(header)

        # 説明
        desc = QLabel("各電力会社の需給実績データの保有状況を確認できます。")
        layout.addWidget(desc)

        # テーブル
        self.avail_table = QTableWidget()
        self.avail_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.avail_table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self.avail_table.verticalHeader().setVisible(False)
        self.avail_table.setStyleSheet("""
            QTableWidget {
                border: 2px solid #a0d2ff;
                border-radius: 10px;
                background-color: #ffffff;
            }
            QHeaderView::section {
                background-color: #e6f2ff;
                color: #0068B7;
                border: 1px solid #a0d2ff;
                padding: 8px;
                font-weight: 600;
            }
        """)
        layout.addWidget(self.avail_table, stretch=1)
        
        # 初期表示
        self.refresh_availability_table()
        
        return page

    def refresh_availability_table(self) -> None:
        """データ可用性テーブルを更新 (企業 x 月)"""
        if not hasattr(self, "avail_table") or not hasattr(self, "avail_year_combo"):
            return
            
        self.avail_table.clear()
        
        selected_year = self.avail_year_combo.currentData()
        if selected_year is None:
            selected_year = datetime.now().year

        # カラム設定: エリア, リンク, 1月...12月
        headers = ["エリア", "データ元"] + [f"{m}月" for m in range(1, 13)]
        self.avail_table.setColumnCount(len(headers))
        self.avail_table.setHorizontalHeaderLabels(headers)
        self.avail_table.setRowCount(len(AREA_INFO))

        # 列幅の調整
        header = self.avail_table.horizontalHeader()
        header.setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeToContents) # エリア名
        header.setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeToContents) # リンク
        for i in range(2, len(headers)):
            header.setSectionResizeMode(i, QtWidgets.QHeaderView.Stretch)

        for r, (code, info) in enumerate(AREA_INFO.items()):
            # エリア名
            item_name = QTableWidgetItem(f"({code}) {info.name}")
            item_name.setFont(QtGui.QFont("", 10, QtGui.QFont.Bold))
            self.avail_table.setItem(r, 0, item_name)
            
            # リンクボタン
            btn = QPushButton("Webページを開く")
            btn.setCursor(Qt.PointingHandCursor)
            btn.setStyleSheet("""
                QPushButton {
                    background-color: #0068B7;
                    color: white;
                    border-radius: 4px;
                    padding: 4px 8px;
                    font-size: 11px;
                }
                QPushButton:hover {
                    background-color: #005090;
                }
            """)
            # クロージャで変数をキャプチャ
            btn.clicked.connect(lambda checked=False, url=info.url: QtGui.QDesktopServices.openUrl(QtCore.QUrl(url)))
            
            # セルにウィジェットを配置するためのコンテナ
            widget = QWidget()
            layout = QHBoxLayout(widget)
            layout.setContentsMargins(4, 2, 4, 2)
            layout.addWidget(btn)
            layout.setAlignment(Qt.AlignCenter)
            self.avail_table.setCellWidget(r, 1, widget)
            
            # 各月のデータ状況
            year_data = self.avail.get(code, {}).get(selected_year, {})
            
            for m in range(1, 13):
                has_data = year_data.get(m, False)
                
                if has_data:
                    text = "◯"
                    bg_color = "#10b981" # 緑
                    fg_color = "#ffffff"
                else:
                    text = "—"
                    bg_color = "#f3f4f6" # グレー
                    fg_color = "#9ca3af"
                
                item = QTableWidgetItem(text)
                item.setTextAlignment(Qt.AlignCenter)
                item.setBackground(QtGui.QColor(bg_color))
                item.setForeground(QtGui.QBrush(QtGui.QColor(fg_color)))
                if has_data:
                    item.setFont(QtGui.QFont("", 10, QtGui.QFont.Bold))
                
                self.avail_table.setItem(r, m + 1, item)

    def refresh_area_year_months(self) -> None:
        # Recompute available year-month combinations per area.

        for code in AREA_INFO.keys():
            self.area_year_months[code] = []
        for year_month, code, _ in self.files:
            self.area_year_months.setdefault(code, []).append(year_month)
        for code, values in self.area_year_months.items():
            unique_sorted = sorted(set(values))
            self.area_year_months[code] = unique_sorted

    def populate_ai_controls(self) -> None:
        # Fill AI tab combos based on scanned files.

        if not hasattr(self, "ai_area_combo"):
            return
        current_code = self.ai_area_combo.currentData()
        self.ai_area_combo.blockSignals(True)
        self.ai_area_combo.clear()
        for code, meta in AREA_INFO.items():
            self.ai_area_combo.addItem(f"({code}) {meta.name}", code)
        self.ai_area_combo.blockSignals(False)
        if current_code:
            idx = self.ai_area_combo.findData(current_code)
            if idx >= 0:
                self.ai_area_combo.setCurrentIndex(idx)
        self.on_ai_area_change()
    
    def on_stats_area_change(self) -> None:
        # 統計分析タブ エリア変更時
        if not hasattr(self, "stats_ym_combo"):
            return
        code = self.stats_area_combo.currentData()
        self.stats_ym_combo.blockSignals(True)
        self.stats_ym_combo.clear()
        for ym in self.area_year_months.get(code, []):
            display = f"{ym[:4]}年{ym[4:6]}月"
            self.stats_ym_combo.addItem(display, ym)
        self.stats_ym_combo.blockSignals(False)
        if self.stats_ym_combo.count() > 0:
            self.stats_ym_combo.setCurrentIndex(0)
    
    def on_stats_ym_change(self) -> None:
        # 統計分析タブ 年月変更時
        pass
    
    def run_statistical_analysis(self) -> None:
        # 統計分析を実行
        code = self.stats_area_combo.currentData()
        ym = self.stats_ym_combo.currentData()
        
        if not code or not ym:
            QtWidgets.QMessageBox.warning(self, "警告", "エリアと年月を選択してください。")
            return
        
        path = DATA_DIR / f"eria_jukyu_{ym}_{code}.csv"
        if not path.exists():
            QtWidgets.QMessageBox.warning(self, "エラー", f"ファイルが見つかりません: {path.name}")
            return
        
        try:
            df, time_col = read_csv(path)
            
            # 数値蛻励ｒ蜿門ｾ・
            numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
            
            if len(numeric_cols) == 0:
                QtWidgets.QMessageBox.warning(self, "エラー", "数値データが見つかりません")
                return
            
            # 基本統計量を計算
            self.display_basic_statistics(df, numeric_cols, time_col)
            
            # 統計分析プロットを描画
            self.plot_timeseries_analysis(df, numeric_cols, time_col)
            
            # 分布分析を描画
            self.plot_distribution_analysis(df, numeric_cols)
            
            # 相関分析を描画
            self.plot_correlation_analysis(df, numeric_cols)
            
            # 時間帯別分析を描画
            self.plot_hourly_analysis(df, numeric_cols, time_col)
            
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "エラー", f"分析中にエラーが発生しました:\n{str(e)}")
    
    def display_basic_statistics(self, df: pd.DataFrame, numeric_cols: list, time_col: Optional[str]) -> None:
        # 基本統計量を表示
        import io
        from scipy import stats as scipy_stats
        
        output = io.StringIO()
        output.write("=" * 80 + "\n")
        output.write("基本統計量レポート\n")
        output.write("=" * 80 + "\n\n")
        
        output.write(f"データ期間: {len(df)}行\n")
        if time_col and time_col in df.columns:
            try:
                time_series = pd.to_datetime(df[time_col])
                output.write(f"開始時刻: {time_series.min()}\n")
                output.write(f"終了時刻: {time_series.max()}\n")
            except:
                pass
        output.write("\n")
        
        # 主要な列の統計量
        key_columns = [col for col in ["発電実績(万kW)", "需要実績(万kW)", "揚水発電実績(万kW)"] if col in numeric_cols]
        if not key_columns:
            key_columns = numeric_cols[:3]  # 最初の3列
        
        for col in key_columns:
            output.write(f"\n【{col}】\n")
            output.write("-" * 60 + "\n")
            
            data = df[col].dropna()
            if len(data) == 0:
                continue
            
            output.write(f"  サンプル数:     {len(data):>12,}\n")
            output.write(f"  平均値:         {data.mean():>12,.2f}\n")
            output.write(f"  中央値:         {data.median():>12,.2f}\n")
            output.write(f"  標準偏差:       {data.std():>12,.2f}\n")
            output.write(f"  最小値:         {data.min():>12,.2f}\n")
            output.write(f"  最大値:         {data.max():>12,.2f}\n")
            output.write(f"  範囲:           {data.max() - data.min():>12,.2f}\n")
            output.write(f"  25%分位点:      {data.quantile(0.25):>12,.2f}\n")
            output.write(f"  75%分位点:      {data.quantile(0.75):>12,.2f}\n")
            output.write(f"  四分位範囲:     {data.quantile(0.75) - data.quantile(0.25):>12,.2f}\n")
            
            # 歪度と尖度
            try:
                skewness = scipy_stats.skew(data)
                kurtosis = scipy_stats.kurtosis(data)
                output.write(f"  歪度:           {skewness:>12,.4f}  (0=対称, >0=右裾, <0=左裾)\n")
                output.write(f"  尖度:           {kurtosis:>12,.4f}  (0=正規分布, >0=尖鋭, <0=平坦)\n")
            except:
                pass
            
            # 変動係数
            cv = (data.std() / data.mean() * 100) if data.mean() != 0 else 0
            output.write(f"  変動係数 (CV):  {cv:>12,.2f}%\n")
        
        output.write("\n" + "=" * 80 + "\n")
        
        self.stats_summary_widget.setPlainText(output.getvalue())
    
    def plot_timeseries_analysis(self, df: pd.DataFrame, numeric_cols: list, time_col: Optional[str]) -> None:
        # Plot the key numeric metrics against the time axis.
        self.stats_timeseries_canvas.ax.clear()
        
        # 譎る俣蛻励ｒ蜿門ｾ・
        if time_col and time_col in df.columns:
            try:
                x = pd.to_datetime(df[time_col])
            except:
                x = range(len(df))
        else:
            x = range(len(df))
        
        # 主要な列をプロット
        key_columns = [col for col in ["需要実績(万kW)", "発電実績(万kW)"] if col in numeric_cols]
        if not key_columns:
            key_columns = numeric_cols[:2]
        
        for col in key_columns:
            self.stats_timeseries_canvas.ax.plot(x, df[col], label=col, linewidth=1.5, alpha=0.8)
        
        self.stats_timeseries_canvas.ax.set_xlabel("時刻", fontsize=11)
        self.stats_timeseries_canvas.ax.set_ylabel("電力(万kW)", fontsize=11)
        self.stats_timeseries_canvas.ax.set_title("時系列トレンド分析", fontsize=14, fontweight='bold', color='#0068B7')
        self.stats_timeseries_canvas.ax.legend(loc='best', framealpha=0.95)
        self.stats_timeseries_canvas.ax.grid(True, alpha=0.3)
        self.stats_timeseries_canvas.fig.tight_layout()
        self.stats_timeseries_canvas.draw()
    
    def plot_distribution_analysis(self, df: pd.DataFrame, numeric_cols: list) -> None:
        # Visualize numeric distributions (histogram + box plot).
        self.stats_distribution_canvas.fig.clear()
        
        key_columns = [col for col in ["需要実績(万kW)", "発電実績(万kW)"] if col in numeric_cols]
        if not key_columns:
            key_columns = numeric_cols[:2]
        
        n_cols = len(key_columns)
        
        for i, col in enumerate(key_columns):
            # ヒストグラム
            ax1 = self.stats_distribution_canvas.fig.add_subplot(2, n_cols, i + 1)
            data = df[col].dropna()
            ax1.hist(data, bins=50, alpha=0.7, color='#0068B7', edgecolor='black')
            ax1.axvline(data.mean(), color='red', linestyle='--', linewidth=2, label=f'平均 {data.mean():.1f}')
            ax1.axvline(data.median(), color='green', linestyle='--', linewidth=2, label=f'中央値: {data.median():.1f}')
            ax1.set_xlabel(col, fontsize=9)
            ax1.set_ylabel("頻度", fontsize=9)
            ax1.set_title(f"{col} - 分布", fontsize=10)
            ax1.legend(fontsize=8)
            ax1.grid(True, alpha=0.3)
            
            # ボックスプロット
            ax2 = self.stats_distribution_canvas.fig.add_subplot(2, n_cols, n_cols + i + 1)
            ax2.boxplot(data, vert=True, patch_artist=True,
                       boxprops=dict(facecolor='#a0d2ff'),
                       medianprops=dict(color='red', linewidth=2))
            ax2.set_ylabel(col, fontsize=9)
            ax2.set_title(f"{col} - 箱ひげ図", fontsize=10)
            ax2.grid(True, alpha=0.3, axis='y')
        
        self.stats_distribution_canvas.fig.suptitle("分布性質分析", fontsize=14, fontweight='bold', color='#0068B7')
        self.stats_distribution_canvas.fig.tight_layout()
        self.stats_distribution_canvas.draw()
    
    def plot_correlation_analysis(self, df: pd.DataFrame, numeric_cols: list) -> None:
        # Render a correlation heatmap for the numeric features.
        self.stats_correlation_canvas.ax.clear()
        
        # 相関行列を計算
        corr_matrix = df[numeric_cols].corr()
        
        # ヒートマップを描画
        im = self.stats_correlation_canvas.ax.imshow(corr_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
        
        # 軸ラベルを設定
        self.stats_correlation_canvas.ax.set_xticks(range(len(numeric_cols)))
        self.stats_correlation_canvas.ax.set_yticks(range(len(numeric_cols)))
        self.stats_correlation_canvas.ax.set_xticklabels(numeric_cols, rotation=45, ha='right', fontsize=8)
        self.stats_correlation_canvas.ax.set_yticklabels(numeric_cols, fontsize=8)
        
        # 相関係数を表示
        for i in range(len(numeric_cols)):
            for j in range(len(numeric_cols)):
                text = self.stats_correlation_canvas.ax.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                                           ha="center", va="center", color="black", fontsize=7)
        
        self.stats_correlation_canvas.ax.set_title("発電方式間の相関分析", fontsize=14, fontweight='bold', color='#0068B7')
        self.stats_correlation_canvas.fig.colorbar(im, ax=self.stats_correlation_canvas.ax, label='相関係数')
        self.stats_correlation_canvas.fig.tight_layout()
        self.stats_correlation_canvas.draw()
    
    def plot_hourly_analysis(self, df: pd.DataFrame, numeric_cols: list, time_col: Optional[str]) -> None:
        # Summarize hourly averages and variability.
        self.stats_hourly_canvas.ax.clear()
        
        # 時刻情報を抽出
        if time_col and time_col in df.columns:
            try:
                df['hour'] = pd.to_datetime(df[time_col]).dt.hour
            except:
                QtWidgets.QMessageBox.warning(self, "警告", "時刻情報を抽出できませんでした。")
                return
        else:
            QtWidgets.QMessageBox.warning(self, "警告", "時刻列が見つかりません。")
            return
        
        # 主要列
        key_col = "需要実績(万kW)" if "需要実績(万kW)" in numeric_cols else numeric_cols[0]
        
        # 時間帯別の統計量を計算
        hourly_stats = df.groupby('hour')[key_col].agg(['mean', 'std', 'min', 'max'])
        
        hours = hourly_stats.index
        means = hourly_stats['mean']
        stds = hourly_stats['std']
        
        # 平均値と信頼区間をプロット
        self.stats_hourly_canvas.ax.plot(hours, means, 'o-', linewidth=2, markersize=6, label='平均､', color='#0068B7')
        self.stats_hourly_canvas.ax.fill_between(hours, means - stds, means + stds, alpha=0.3, label='ﾂｱ1標準偏差')
        
        self.stats_hourly_canvas.ax.set_xlabel("時刻", fontsize=11)
        self.stats_hourly_canvas.ax.set_ylabel(key_col, fontsize=11)
        self.stats_hourly_canvas.ax.set_title(f"時間帯別需要パターン分析 ({key_col})", fontsize=14, fontweight='bold', color='#0068B7')
        self.stats_hourly_canvas.ax.set_xticks(range(0, 24, 2))
        self.stats_hourly_canvas.ax.legend(loc='best')
        self.stats_hourly_canvas.ax.grid(True, alpha=0.3)
        self.stats_hourly_canvas.fig.tight_layout()
        self.stats_hourly_canvas.draw()

    def on_ai_area_change(self) -> None:
        # Populate the year-month combo when the area changes.

        code = self.ai_area_combo.currentData()
        
        # 年・月の分離版コンボボックスを更新
        if hasattr(self, "ai_year_combo") and hasattr(self, "ai_month_combo"):
            year_months = self.area_year_months.get(code, [])
            
            # 年のリストを取得（重複を除く）
            years = sorted(set([ym[:4] for ym in year_months]))
            
            self.ai_year_combo.blockSignals(True)
            self.ai_year_combo.clear()
            for year in years:
                self.ai_year_combo.addItem(f"{year}年", year)
            self.ai_year_combo.blockSignals(False)
            
            if self.ai_year_combo.count() > 0:
                self.ai_year_combo.setCurrentIndex(0)
                self.update_ai_month_combo()
        
        # 後方互換性のため、ai_ym_comboも更新
        if hasattr(self, "ai_ym_combo"):
            self.ai_ym_combo.blockSignals(True)
            self.ai_ym_combo.clear()
            for ym in self.area_year_months.get(code, []):
                display = f"{ym[:4]}年{ym[4:6]}月"
                self.ai_ym_combo.addItem(display, ym)
            self.ai_ym_combo.blockSignals(False)
            if self.ai_ym_combo.count() > 0:
                self.ai_ym_combo.setCurrentIndex(0)
        
        self.ai_dataframe = None
        self.ai_time_column = None
        self.ai_target_series = None
        self.ai_training_index = None
        self.ai_column_combo.clear()
        
        if (hasattr(self, "ai_year_combo") and self.ai_year_combo.count() == 0):
            self.append_ai_log("選択したエリアのCSVが見つかりません。data/にファイルを追加してください。")

    def update_ai_month_combo(self):
        """AI分析タブ: 選択された年に対応する月のリストを更新"""
        if not hasattr(self, "ai_year_combo") or not hasattr(self, "ai_month_combo"):
            return
        
        code = self.ai_area_combo.currentData()
        year = self.ai_year_combo.currentData()
        
        if not code or not year:
            return
        
        # 選択された年に対応する月を取得
        year_months = self.area_year_months.get(code, [])
        months = sorted([ym[4:6] for ym in year_months if ym.startswith(year)])
        
        self.ai_month_combo.blockSignals(True)
        self.ai_month_combo.clear()
        for month in months:
            self.ai_month_combo.addItem(f"{int(month)}月", month)
        self.ai_month_combo.blockSignals(False)
        
        if self.ai_month_combo.count() > 0:
            self.ai_month_combo.setCurrentIndex(0)
            self.on_ai_year_month_change()

    def on_ai_year_month_change(self):
        """AI分析タブ: 年または月が変更された時の処理"""
        if not hasattr(self, "ai_year_combo") or not hasattr(self, "ai_month_combo"):
            return
        
        year = self.ai_year_combo.currentData()
        month = self.ai_month_combo.currentData()
        
        if not year or not month:
            return
        
        # 年月を結合してYYYYMM形式にする
        ym = f"{year}{month}"
        
        # ai_ym_comboも更新（後方互換性）
        if hasattr(self, "ai_ym_combo"):
            for i in range(self.ai_ym_combo.count()):
                if self.ai_ym_combo.itemData(i) == ym:
                    self.ai_ym_combo.blockSignals(True)
                    self.ai_ym_combo.setCurrentIndex(i)
                    self.ai_ym_combo.blockSignals(False)
                    break
        
        # 既存のon_ai_ym_change処理を実行
        self.on_ai_ym_change()

    def on_ai_ym_change(self) -> None:
        # Load the selected dataset and populate the column combo.

        self.load_ai_dataset()

    def load_ai_dataset(self) -> None:
        code = self.ai_area_combo.currentData()
        ym = self.ai_ym_combo.currentData()
        if not code or not ym:
            return
        path = DATA_DIR / f"eria_jukyu_{ym}_{code}.csv"
        if not path.exists():
            self.append_ai_log(f"CSVの瑚ｦ九▽かりせ帙ｓ: {path.name}")
            self.ai_dataframe = None
            self.ai_column_combo.clear()
            return
        try:
            df, time_col = read_csv(path)
            
            # 天候データの結合を試みる
            if time_col:
                df, has_weather = merge_weather_data(df, code, ym, time_col)
                if has_weather:
                    self.append_ai_log("天候データを結合しました。")

            numeric_columns = [
                str(col)
                for col in df.columns
                if pd.api.types.is_numeric_dtype(df[col])
            ]
            self.ai_column_combo.blockSignals(True)
            self.ai_column_combo.clear()
            for col in numeric_columns:
                self.ai_column_combo.addItem(col)
            self.ai_column_combo.blockSignals(False)
            
            # 需要・天候分析の列コンボを更新（積算列も含む）
            self._update_dw_columns()
            
            self.ai_dataframe = df
            self.ai_time_column = time_col
            self.ai_target_series = None
            self.ai_training_index = None
            if numeric_columns:
                self.ai_column_combo.setCurrentIndex(0)
            self.append_ai_log(
                f"{path.name} を読み込みました (行数: {len(df):,})."
            )
            if not numeric_columns:
                self.append_ai_log("数値カラムが見つかりませんでした。")
        except Exception as exc:
            self.append_ai_log(f"CSV読込に失敗しました: {exc}")
            self.ai_dataframe = None
            self.ai_column_combo.clear()

    def append_ai_log(self, message: str) -> None:
        if not hasattr(self, "ai_log_output") or self.ai_log_output is None:
            return
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.ai_log_output.appendPlainText(f"[{timestamp}] {message}")
    
    def get_ai_data_series(self, interpolate: bool = False):
        # AIタブで選択した系列を取得する。
        if self.ai_dataframe is None:
            self.load_ai_dataset()
        
        if self.ai_dataframe is None:
            QtWidgets.QMessageBox.warning(self, "警告", "データが読み込まれていません。")
            return None, None
        
        target_column = (self.ai_column_combo.currentText() or "").strip()
        if not target_column:
            QtWidgets.QMessageBox.warning(self, "警告", "目的列を選択してください。")
            return None, None
        
        series = pd.to_numeric(self.ai_dataframe[target_column], errors="coerce")
        
        missing_count = series.isna().sum()
        
        if interpolate and missing_count > 0:
            series_clean = series.interpolate(method='linear', limit_direction='both')
            series_clean = series_clean.ffill().bfill()
            self.append_ai_log(
                f"データ件数: {len(series_clean)} 行 (欠損 {missing_count} 件を補完)"
            )
        else:
            series_clean = series.dropna()
            if missing_count > 0:
                self.append_ai_log(
                    f"欠損 {missing_count} 件を除外 (残り {len(series_clean)} 行)"
                )
            else:
                self.append_ai_log(f"データ件数: {len(series_clean)} 行 (欠損なし)")
        
        if len(series_clean) < 10:
            QtWidgets.QMessageBox.warning(
                self,
                "警告",
                "系列のデータ数が少なすぎます (10 行以上必要)。",
            )
            return None, None
        
        return series_clean, target_column
    
    def run_stl_decomposition(self) -> None:
        # STL時系列分解を実行
        from statsmodels.tsa.seasonal import STL
        
        self.append_ai_log("=" * 50)
        self.append_ai_log("STL時系列分解を開始します...")
        
        # 欠損値を補完して時系列データを取得
        series, col_name = self.get_ai_data_series(interpolate=True)
        if series is None:
            return
        
        try:
            # STL分解（周期性を自動調整）
            period = 24  # 24時間周期
            
            # データの長さに応じて周期を調整
            if len(series) < period * 2:
                period = max(7, len(series) // 3)
                if period % 2 == 0:  # 奇数にする
                    period += 1
            
            if len(series) < 2 * period:
                QtWidgets.QMessageBox.warning(self, "警告", f"データ数が不足しているため分解できません。\n最低でも{2*period}サンプル必要です（現在: {len(series)}）。")
                return
            
            self.append_ai_log(f"STL分解を実行中（周期: {period}）...")
            # periodパラメータの周期を調整、seasonal引数は季節性成分のスムージング幅（奇数）
            seasonal_length = 7  # 季節性成分のスムージング幅（奇数である必要がある）
            stl = STL(series.values, period=period, seasonal=seasonal_length, robust=True)
            result = stl.fit()
            
            # プロット
            self.ai_result_canvas.fig.clear()
            
            ax1 = self.ai_result_canvas.fig.add_subplot(4, 1, 1)
            ax1.plot(series.values, label='元データ', color='#0068B7', linewidth=1)
            ax1.set_ylabel('観測値', fontsize=10)
            ax1.legend(loc='upper right', fontsize=9)
            ax1.grid(True, alpha=0.3)
            ax1.set_title(f'STL時系列分解: {col_name}', fontsize=12, fontweight='bold', color='#0068B7')
            
            ax2 = self.ai_result_canvas.fig.add_subplot(4, 1, 2)
            ax2.plot(result.trend, label='トレンド', color='#10b981', linewidth=1.5)
            ax2.set_ylabel('トレンド', fontsize=10)
            ax2.legend(loc='upper right', fontsize=9)
            ax2.grid(True, alpha=0.3)
            
            ax3 = self.ai_result_canvas.fig.add_subplot(4, 1, 3)
            ax3.plot(result.seasonal, label='季節性', color='#f59e0b', linewidth=1)
            ax3.set_ylabel('季節性', fontsize=10)
            ax3.legend(loc='upper right', fontsize=9)
            ax3.grid(True, alpha=0.3)
            
            ax4 = self.ai_result_canvas.fig.add_subplot(4, 1, 4)
            ax4.plot(result.resid, label='残差', color='#ef4444', linewidth=0.8, alpha=0.7)
            ax4.set_ylabel('残差', fontsize=10)
            ax4.set_xlabel('時間インデックス', fontsize=10)
            ax4.legend(loc='upper right', fontsize=9)
            ax4.grid(True, alpha=0.3)
            
            self.ai_result_canvas.fig.tight_layout()
            self.ai_result_canvas.draw()
            
            # 評価指標を更新
            self.ai_eval_widget.setPlainText(
                f"STL分解評価指標\n{'='*60}\n\n"
                f"トレンド成分:\n"
                f"  平均: {result.trend.mean():.2f}\n"
                f"  標準偏差: {result.trend.std():.2f}\n"
                f"  範囲: [{result.trend.min():.2f}, {result.trend.max():.2f}]\n\n"
                f"季節性成分:\n"
                f"  振幅: {(result.seasonal.max() - result.seasonal.min()):.2f}\n"
                f"  周期: {period}\n\n"
                f"残差成分:\n"
                f"  平均: {result.resid.mean():.4f}\n"
                f"  標準偏差: {result.resid.std():.2f}\n"
                f"  ホワイトノイズ性の簡易チェック\n"
            )
            
            self.append_ai_log("STL分解が完了しました。")
            self.ai_tabs.setCurrentIndex(1) # 結果タブに切り替え
            
        except Exception as e:
            self.append_ai_log(f"エラー: {str(e)}")
            QtWidgets.QMessageBox.critical(self, "エラー", f"STL分解の実行に失敗しました:\n{str(e)}")
    
    def run_arima_forecast(self) -> None:
        # ARIMAモデルの実行
        try:
            from statsmodels.tsa.arima.model import ARIMA
            from sklearn.metrics import mean_absolute_error, mean_squared_error
        except ImportError as e:
            self.append_ai_log(f"ライブラリのインポートエラー: {str(e)}")
            QtWidgets.QMessageBox.critical(self, "エラー", f"必要なライブラリがインストールされていない可能性があります:\n{str(e)}\n\npip install statsmodels scikit-learn")
            return
        
        self.append_ai_log("=" * 50)
        self.append_ai_log("ARIMA予測を開始します...")
        
        series, col_name = self.get_ai_data_series()
        if series is None:
            return
        
        try:
            # 訓練/テストデータ分割
            train_ratio = self.train_ratio_spin.value()
            train_size = int(len(series) * train_ratio)
            train, test = series[:train_size], series[train_size:]
            
            if len(test) == 0:
                QtWidgets.QMessageBox.warning(self, "警告", "テストデータがありません。訓練データの比率を調整してください。")
                return
            
            self.append_ai_log(f"訓練データ: {len(train)}サンプル, テストデータ: {len(test)}サンプル")
            
            # ARIMAモデル (p=5, d=1, q=0) - パラメータは調整が必要
            model = ARIMA(train, order=(5, 1, 0))
            fitted = model.fit()
            
            # 予測
            forecast_steps = min(self.ai_horizon_spin.value(), len(test))
            forecast = fitted.forecast(steps=forecast_steps)
            
            # 評価
            actual = test[:forecast_steps]
            mae = mean_absolute_error(actual, forecast)
            rmse = np.sqrt(mean_squared_error(actual, forecast))
            
            eval_text = (
                f"ARIMA(5,1,0) モデル評価 (予測期間: {forecast_steps}ステップ):\n"
                f"  - 平均絶対誤差 (MAE): {mae:.4f}\n"
                f"  - 二乗平均平方根誤差 (RMSE): {rmse:.4f}\n\n"
                f"予測値:\n{forecast.to_string(float_format='{:.2f}'.format)}"
            )
            self.ai_eval_widget.setText(eval_text)
            self.append_ai_log("ARIMAモデルの評価が完了しました。")
            
            # グラフ描画
            self.ai_result_canvas.fig.clear()
            ax = self.ai_result_canvas.fig.add_subplot(111)
            ax.plot(series.index, series, label='実績値', color='blue')
            ax.plot(actual.index, actual, label='テスト実績', color='green')
            ax.plot(actual.index, forecast, label='予測値 (ARIMA)', color='red', linestyle='--')
            ax.set_title(f'{col_name} - ARIMA予測')
            ax.set_xlabel('日付')
            ax.set_ylabel('値')
            ax.legend()
            ax.grid(True)
            self.ai_result_canvas.draw()
            self.append_ai_log("予測グラフを更新しました。")
            self.ai_tabs.setCurrentIndex(1) # 予測タブに切り替え

        except Exception as e:
            self.append_ai_log(f"ARIMA予測中にエラーが発生しました: {str(e)}")
            QtWidgets.QMessageBox.critical(self, "エラー", f"ARIMA予測の実行中にエラーが発生しました:\n{str(e)}")
    
    def run_exponential_smoothing(self) -> None:
        # 指数平滑法で予測
        try:
            from statsmodels.tsa.holtwinters import ExponentialSmoothing
            from sklearn.metrics import mean_absolute_error, mean_squared_error
        except ImportError as e:
            self.append_ai_log(f"ライブラリのインポートエラー: {str(e)}")
            QtWidgets.QMessageBox.critical(self, "エラー", f"必要なライブラリがインストールされていない可能性があります:\n{str(e)}\n\npip install statsmodels scikit-learn")
            return
        
        self.append_ai_log("=" * 50)
        self.append_ai_log("指数平滑法による予測を開始します...")
        
        series, col_name = self.get_ai_data_series()
        if series is None:
            return
        
        try:
            # 訓練/テストデータ分割
            train_ratio = self.train_ratio_spin.value()
            train_size = int(len(series) * train_ratio)
            train, test = series[:train_size], series[train_size:]
            
            if len(test) == 0:
                QtWidgets.QMessageBox.warning(self, "警告", "テストデータがありません。")
                return
            
            self.append_ai_log(f"訓練データ: {len(train)}サンプル, テストデータ: {len(test)}サンプル")
            
            # Holt-Winters法（加法モデル、季節性24時間）
            seasonal_periods = min(24, len(train) // 2)
            if seasonal_periods < 2:
                # 季節性なしモデル
                model = ExponentialSmoothing(train, trend='add', seasonal=None)
                self.append_ai_log("季節性なしモデルを適用")
            else:
                model = ExponentialSmoothing(train, trend='add', seasonal='add', seasonal_periods=seasonal_periods)
                self.append_ai_log(f"季節性ありモデルを適用（周期: {seasonal_periods}）")
            
            fitted = model.fit()
            
            # 予測
            forecast_steps = min(self.ai_horizon_spin.value(), len(test))
            forecast = fitted.forecast(steps=forecast_steps)
            
            # 評価
            actual = test[:forecast_steps]
            mae = mean_absolute_error(actual, forecast)
            rmse = np.sqrt(mean_squared_error(actual, forecast))
            mape = np.mean(np.abs((actual - forecast) / actual)) * 100
            
            # プロット
            self.ai_result_canvas.fig.clear()
            ax = self.ai_result_canvas.fig.add_subplot(1, 1, 1)
            
            ax.plot(range(len(train)), train.values, label='訓練データ', color='#0068B7', linewidth=1.5, alpha=0.8)
            test_idx = range(len(train), len(train) + len(actual))
            ax.plot(test_idx, actual.values, label='実績値', color='#10b981', linewidth=1.5)
            ax.plot(test_idx, forecast, label='指数平滑予測', color='#f59e0b', linewidth=2, linestyle='--')
            
            ax.set_xlabel('時間インデックス', fontsize=11)
            ax.set_ylabel(col_name, fontsize=11)
            ax.set_title('Holt-Winters指数平滑予測', fontsize=14, fontweight='bold', color='#0068B7')
            ax.legend(loc='best', fontsize=10)
            ax.grid(True, alpha=0.3)
            
            self.ai_result_canvas.fig.tight_layout()
            self.ai_result_canvas.draw()
            
            # 評価結果
            self.ai_eval_widget.setPlainText(
                f"指数平滑予測モデル評価\n{'='*60}\n\n"
                f"モデル: Holt-Winters (加法モデル)\n"
                f"訓練サンプル数: {len(train)}\n"
                f"予測期間: {forecast_steps}ステップ\n\n"
                f"評価指標:\n"
                f"  MAE  (平均絶対誤差):     {mae:.4f}\n"
                f"  RMSE (二乗平均平方根誤差): {rmse:.4f}\n"
                f"  MAPE (平均絶対パーセント誤差): {mape:.2f}%\n\n"
                f"パラメータ:\n"
                f"  Alpha (レベル平滑化): {fitted.params['smoothing_level']:.4f}\n"
                f"  Beta  (トレンド平滑化): {fitted.params.get('smoothing_trend', 0):.4f}\n"
                f"  Gamma (季節性平滑化): {fitted.params.get('smoothing_seasonal', 0):.4f}\n"
            )
            
            # 残差分析
            residuals = train - fitted.fittedvalues
            self.plot_residual_analysis(residuals)
            
            self.append_ai_log(f"指数平滑予測完了 - MAE: {mae:.2f}, RMSE: {rmse:.2f}, MAPE: {mape:.2f}%")
            self.ai_tabs.setCurrentIndex(1)
            
        except Exception as e:
            self.append_ai_log(f"エラー: {str(e)}")
            QtWidgets.QMessageBox.critical(self, "エラー", f"指数平滑予測の実行に失敗しました:\n{str(e)}")
    
    def run_transformer_forecast(self) -> None:
        # Transformerモデルの実行
        if not TRANSFORMER_AVAILABLE:
            QtWidgets.QMessageBox.warning(self, "警告", "PyTorchがインストールされていない可能性があります。\n\npip install torch")
            return
        
        try:
            from sklearn.metrics import mean_absolute_error, mean_squared_error
        except ImportError as e:
            self.append_ai_log(f"ライブラリのインポートエラー: {str(e)}")
            QtWidgets.QMessageBox.critical(self, "エラー", f"scikit-learnが必要です。\n{str(e)}")
            return
        
        self.append_ai_log("=" * 50)
        self.append_ai_log("Transformer予測を開始します...")
        
        series, col_name = self.get_ai_data_series()
        if series is None:
            return
        
        try:
            # 訓練/テストデータ分割
            train_ratio = self.train_ratio_spin.value()
            train_size = int(len(series) * train_ratio)
            train, test = series[:train_size], series[train_size:]
            
            if len(test) == 0:
                QtWidgets.QMessageBox.warning(self, "警告", "テストデータがありません。訓練データの比率を調整してください。")
                return
            
            self.append_ai_log(f"訓練データ: {len(train)}サンプル, テストデータ: {len(test)}サンプル")
            
            # Transformerモデルのパラメータ
            prediction_length = min(self.ai_horizon_spin.value(), len(test))
            context_length = min(48, len(train) // 2)  # 48時間(2日分)をコンテキストとして使用
            
            if context_length < 24:
                QtWidgets.QMessageBox.warning(self, "警告", "訓練データが少なすぎます。最低24時間必要です。")
                return
            
            self.append_ai_log(f"コンテキスト長: {context_length}, 予測期間: {prediction_length}")
            self.append_ai_log("モデルを学習中...")
            
            # Transformerモデルの作成と学習
            forecaster = DemandTransformerForecaster(
                context_length=context_length,
                prediction_length=prediction_length,
                d_model=128,
                nhead=4,
                num_layers=2,
                dropout=0.1,
                feedforward_dim=256,
                learning_rate=5e-4,
                batch_size=32,
                epochs=30,  # エポック数
            )
            
            # 学習
            def progress_callback(epoch, total_epochs, train_loss, val_loss):
                QtWidgets.QApplication.processEvents()
                val_str = f", Val: {val_loss:.6f}" if val_loss is not None else ""
                self.append_ai_log(f"  Epoch {epoch:2d}/{total_epochs}: Train: {train_loss:.6f}{val_str}")

            training_log = forecaster.fit(train.values, validation_split=0.2, verbose=False, callback=progress_callback)
            
            # 予測
            result = forecaster.predict(train.values)
            forecast = result.prediction[:prediction_length]
            
            # 評価指標
            actual = test[:prediction_length]
            mae = mean_absolute_error(actual, forecast)
            rmse = np.sqrt(mean_squared_error(actual, forecast))
            mape = np.mean(np.abs((actual - forecast) / actual)) * 100
            
            # プロット
            self.ai_result_canvas.fig.clear()
            ax = self.ai_result_canvas.fig.add_subplot(1, 1, 1)
            
            # 訓練データ
            ax.plot(range(len(train)), train.values, label='訓練データ', color='#0068B7', linewidth=1.5, alpha=0.8)
            
            # テストデータ
            test_idx = range(len(train), len(train) + len(actual))
            ax.plot(test_idx, actual.values, label='実績値', color='#10b981', linewidth=1.5)
            ax.plot(test_idx, forecast, label='Transformer予測', color='#ef4444', linewidth=2, linestyle='--')
            
            ax.set_xlabel('時間インデックス', fontsize=11)
            ax.set_ylabel(col_name, fontsize=11)
            ax.set_title(f'Transformer予測結果', fontsize=14, fontweight='bold', color='#0068B7')
            ax.legend(loc='best', fontsize=10)
            ax.grid(True, alpha=0.3)
            
            self.ai_result_canvas.fig.tight_layout()
            self.ai_result_canvas.draw()
            
            # 学習曲線のプロット
            self.plot_training_curves(training_log)
            
            # 最終検証損失の文字列を作成
            val_loss_str = f"{training_log.val_loss[-1]:.6f}" if training_log.val_loss[-1] is not None else "N/A"
            
            # 評価結果
            self.ai_eval_widget.setPlainText(
                f"Transformer予測モデル評価\n{'='*60}\n\n"
                f"モデル: Transformer Encoder\n"
                f"訓練サンプル数: {len(train)}\n"
                f"コンテキスト長: {context_length}時間\n"
                f"予測期間: {prediction_length}ステップ\n"
                f"エポック数: 30\n\n"
                f"モデルアーキテクチャ:\n"
                f"  d_model: 128\n"
                f"  num_heads: 4\n"
                f"  num_layers: 2\n"
                f"  feedforward_dim: 256\n"
                f"  dropout: 0.1\n\n"
                f"評価指標:\n"
                f"  MAE  (平均絶対誤差):     {mae:.4f}\n"
                f"  RMSE (二乗平均平方根誤差): {rmse:.4f}\n"
                f"  MAPE (平均絶対パーセント誤差): {mape:.2f}%\n\n"
                f"最終学習損失: {training_log.train_loss[-1]:.6f}\n"
                f"最終検証損失: {val_loss_str}\n"
            )
            
            # 残差分析
            residuals = actual.values - forecast
            self.plot_residual_analysis(pd.Series(residuals))
            
            self.append_ai_log(f"Transformer予測完了 - MAE: {mae:.2f}, RMSE: {rmse:.2f}, MAPE: {mape:.2f}%")
            self.ai_tabs.setCurrentIndex(1)
            
        except Exception as e:
            import traceback
            self.append_ai_log(f"エラー: {str(e)}")
            self.append_ai_log(traceback.format_exc())
            QtWidgets.QMessageBox.critical(self, "エラー", f"Transformer予測の実行に失敗しました:\n{str(e)}")
    
    def plot_residual_analysis(self, residuals) -> None:
        # 残差分析プロット
        from scipy import stats as scipy_stats
        
        self.ai_residual_canvas.fig.clear()
        
        # 残差の時系列プロット
        ax1 = self.ai_residual_canvas.fig.add_subplot(2, 2, 1)
        ax1.plot(residuals, color='#ef4444', linewidth=0.8)
        ax1.axhline(y=0, color='black', linestyle='--', linewidth=1)
        ax1.set_title('残差の時系列', fontsize=10, fontweight='bold')
        ax1.set_xlabel('時間インデックス', fontsize=9)
        ax1.set_ylabel('残差', fontsize=9)
        ax1.grid(True, alpha=0.3)
        
        # 残差のヒストグラム
        ax2 = self.ai_residual_canvas.fig.add_subplot(2, 2, 2)
        ax2.hist(residuals, bins=30, color='#0068B7', alpha=0.7, edgecolor='black')
        ax2.set_title('残差の分布', fontsize=10, fontweight='bold')
        ax2.set_xlabel('残差', fontsize=9)
        ax2.set_ylabel('頻度', fontsize=9)
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Q-Qプロット
        ax3 = self.ai_residual_canvas.fig.add_subplot(2, 2, 3)
       
        scipy_stats.probplot(residuals, dist="norm", plot=ax3)
        ax3.set_title('Q-Qプロット（正規性確認）', fontsize=10, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # ACF（自己相関）
        ax4 = self.ai_residual_canvas.fig.add_subplot(2, 2, 4)
        from statsmodels.graphics.tsaplots import plot_acf
        plot_acf(residuals, ax=ax4, lags=min(40, len(residuals)//2), alpha=0.05)
        ax4.set_title('自己相関（ACF）', fontsize=10, fontweight='bold')
        ax4.set_xlabel('ラグ', fontsize=9)
        
        self.ai_residual_canvas.fig.tight_layout()
        self.ai_residual_canvas.draw()

    def plot_training_curves(self, training_log) -> None:
        # 学習曲線をプロット (Transformer用) 残差分析キャンバスの4番目のサブプロットに追加
        # 残差分析の4番目のプロット(ACF等)を学習曲線に置き換える
        # ただし、学習のプロットはそのままにして情報のログに出力
        epochs = range(1, len(training_log.train_loss) + 1)
        
        # ログに学習の進捗を出力
        self.append_ai_log("学習曲線")
        for i, (train_loss, val_loss) in enumerate(zip(training_log.train_loss, training_log.val_loss), 1):
            if i % 5 == 0 or i == len(training_log.train_loss):  # 5エポックごとに表示
                val_str = f", Val: {val_loss:.6f}" if val_loss is not None else ""
                self.append_ai_log(f"  Epoch {i:2d}: Train: {train_loss:.6f}{val_str}")

    def run_weather_analysis(self) -> None:
        """天候データと発電量の関係を分析"""
        self.append_ai_log("=" * 50)
        self.append_ai_log("天候データ分析を開始します...")
        
        # エリアと年月を取得
        code = self.ai_area_combo.currentData()
        ym = self.ai_ym_combo.currentData()
        
        if not code or not ym:
            QtWidgets.QMessageBox.warning(self, "警告", "エリアと年月を選択してください。")
            return
        
        # 電力データを読み込み
        path = DATA_DIR / f"eria_jukyu_{ym}_{code}.csv"
        if not path.exists():
            QtWidgets.QMessageBox.warning(self, "エラー", f"ファイルが見つかりません: {path.name}")
            return
        
        try:
            df_power, time_col = read_csv(path)
            self.append_ai_log(f"{path.name} を読み込みました (行数: {len(df_power):,}).")
            
            # 天候データと結合
            df_merged, has_weather = merge_weather_data(df_power, code, ym, time_col)
            
            if not has_weather:
                city_name = AREA_WEATHER_MAP.get(code, "不明")
                area_name = AREA_INFO.get(code, AreaInfo("不明", "")).name
                
                # 利用可能な天候ファイルをリストアップ
                available_files = []
                if WEATHER_DIR.exists():
                    available_files = [f.name for f in WEATHER_DIR.glob("*.csv")]
                
                msg = (
                    f"エリア: {code} ({area_name})\n"
                    f"対象年月: {ym[:4]}年{ym[4:6]}月\n"
                    f"期待ファイル: weather/{city_name}_YYYYMMDD_YYYYMMDD.csv\n\n"
                )
                
                if available_files:
                    msg += f"利用可能な天候ファイル:\n" + "\n".join([f"  - {f}" for f in available_files[:5]])
                    if len(available_files) > 5:
                        msg += f"\n  ... 他{len(available_files)-5}件"
                else:
                    msg += "天候データフォルダにファイルがありません。"
                
                QtWidgets.QMessageBox.warning(self, "天候データなし", msg)
                self.append_ai_log(f"天候データが見つかりませんでした: {city_name}_*.csv (対象: {ym})")
                return
            
            self.append_ai_log(f"天候データを結合しました。")
            
            # 発電種別と天候の列を抽出
            generation_cols = []
            for category, col_names in GENERATION_CATEGORIES.items():
                for col_name in col_names:
                    if col_name in df_merged.columns:
                        generation_cols.append((category, col_name))
                        break
            
            weather_cols = {
                "temperature": "気温(℃)",
                "solar_radiation": "日射量(MJ/㎡)",
                "sunlight": "日照時間(h)",
                "wind_speed": "風速(m/s)",
                "precipitation": "降水量(mm)"
            }
            
            # 利用可能な天候列を確認
            available_weather = {k: v for k, v in weather_cols.items() if k in df_merged.columns}
            
            if not available_weather:
                QtWidgets.QMessageBox.warning(self, "エラー", "天候データ列が見つかりません。")
                return
            
            self.append_ai_log(f"発電種別: {len(generation_cols)}種類")
            self.append_ai_log(f"天候指標: {', '.join(available_weather.values())}")
            
            # プロット
            self.ai_weather_canvas.fig.clear()
            
            # サブプロット数を計算
            n_weather = len(available_weather)
            n_generation = min(3, len(generation_cols))  # 最大3種類の発電方式
            
            if n_generation == 0:
                QtWidgets.QMessageBox.warning(self, "エラー", "発電データが見つかりません。")
                return
            
            # 重要な発電方式を選択（太陽光、風力、需要実績を優先）
            priority_categories = ["太陽光", "風力", "火力", "水力", "原子力"]
            selected_generation = []
            
            for cat in priority_categories:
                for gen_cat, gen_col in generation_cols:
                    if gen_cat == cat and len(selected_generation) < n_generation:
                        selected_generation.append((gen_cat, gen_col))
                        break
                if len(selected_generation) >= n_generation:
                    break
            
            # 不足分を補充
            if len(selected_generation) < n_generation:
                for gen_cat, gen_col in generation_cols:
                    if (gen_cat, gen_col) not in selected_generation:
                        selected_generation.append((gen_cat, gen_col))
                        if len(selected_generation) >= n_generation:
                            break
            
            # グリッドレイアウト
            n_rows = len(selected_generation)
            n_cols = min(2, n_weather)  # 最大2列
            
            plot_idx = 1
            correlations = []
            
            for i, (gen_cat, gen_col) in enumerate(selected_generation):
                # 主要な天候指標（気温と日射量）を優先
                primary_weather = []
                if "temperature" in available_weather:
                    primary_weather.append(("temperature", available_weather["temperature"]))
                if "solar_radiation" in available_weather:
                    primary_weather.append(("solar_radiation", available_weather["solar_radiation"]))
                elif "sunlight" in available_weather:
                    primary_weather.append(("sunlight", available_weather["sunlight"]))
                
                for j, (weather_key, weather_label) in enumerate(primary_weather[:n_cols]):
                    ax = self.ai_weather_canvas.fig.add_subplot(n_rows, n_cols, plot_idx)
                    plot_idx += 1
                    
                    # データを準備（NaNを除去）
                    mask = df_merged[gen_col].notna() & df_merged[weather_key].notna()
                    x_data = df_merged.loc[mask, weather_key]
                    y_data = df_merged.loc[mask, gen_col]
                    
                    if len(x_data) < 2:
                        ax.text(0.5, 0.5, "データ不足", ha='center', va='center', transform=ax.transAxes)
                        continue
                    
                    # 散布図
                    ax.scatter(x_data, y_data, alpha=0.3, s=20, color='#0068B7')
                    
                    # 回帰直線
                    try:
                        from scipy.stats import linregress
                        slope, intercept, r_value, p_value, std_err = linregress(x_data, y_data)
                        line_x = np.array([x_data.min(), x_data.max()])
                        line_y = slope * line_x + intercept
                        ax.plot(line_x, line_y, 'r--', linewidth=2, label=f'R²={r_value**2:.3f}')
                        correlations.append((gen_cat, weather_label, r_value, p_value))
                    except Exception:
                        pass
                    
                    ax.set_xlabel(weather_label, fontsize=9)
                    ax.set_ylabel(f'{gen_cat} (万kW)', fontsize=9)
                    ax.set_title(f'{gen_cat} vs {weather_label}', fontsize=10, fontweight='bold')
                    ax.legend(fontsize=8)
                    ax.grid(True, alpha=0.3)
            
            self.ai_weather_canvas.fig.tight_layout()
            self.ai_weather_canvas.draw()
            
            # 相関係数レポート
            report = f"天候データと発電量の相関分析\n{'='*60}\n\n"
            report += f"エリア: {AREA_INFO[code].name}\n"
            report += f"期間: {ym[:4]}年{ym[4:6]}月\n"
            report += f"データ数: {len(df_merged):,}件\n\n"
            report += "相関分析結果:\n"
            report += "-" * 60 + "\n"
            
            for gen_cat, weather_label, r_value, p_value in correlations:
                significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "n.s."
                report += f"{gen_cat} × {weather_label}:\n"
                report += f"  相関係数 (r): {r_value:>8.4f}\n"
                report += f"  決定係数 (R²): {r_value**2:>8.4f}\n"
                report += f"  p値:        {p_value:>8.6f} {significance}\n\n"
            
            report += "\n注: ***p<0.001, **p<0.01, *p<0.05, n.s.=有意差なし\n"
            
            self.ai_eval_widget.setPlainText(report)
            
            # ログ出力
            self.append_ai_log("相関分析完了:")
            for gen_cat, weather_label, r_value, p_value in correlations:
                self.append_ai_log(f"  {gen_cat} × {weather_label}: r={r_value:.3f}, p={p_value:.4f}")
            
            self.ai_tabs.setCurrentIndex(4)  # 天候分析タブに切り替え
            
        except Exception as e:
            import traceback
            self.append_ai_log(f"エラー: {str(e)}")
            self.append_ai_log(traceback.format_exc())
            QtWidgets.QMessageBox.critical(self, "エラー", f"天候分析に失敗しました:\n{str(e)}")

    def run_weather_forecast(self) -> None:
        """天候データを考慮した需要予測（ARIMAX）"""
        try:
            from statsmodels.tsa.statespace.sarimax import SARIMAX
            from sklearn.metrics import mean_absolute_error, mean_squared_error
        except ImportError as e:
            self.append_ai_log(f"ライブラリのインポートエラー: {str(e)}")
            QtWidgets.QMessageBox.critical(self, "エラー", f"必要なライブラリがインストールされていません:\n{str(e)}\n\npip install statsmodels scikit-learn")
            return
        
        self.append_ai_log("=" * 50)
        self.append_ai_log("天候考慮予測（ARIMAX）を開始します...")
        
        # エリアと年月を取得
        code = self.ai_area_combo.currentData()
        ym = self.ai_ym_combo.currentData()
        col_name = self.ai_column_combo.currentText()
        
        if not code or not ym or not col_name:
            QtWidgets.QMessageBox.warning(self, "警告", "エリア、年月、目的変数を選択してください。")
            return
        
        # 電力データを読み込み
        path = DATA_DIR / f"eria_jukyu_{ym}_{code}.csv"
        if not path.exists():
            QtWidgets.QMessageBox.warning(self, "エラー", f"ファイルが見つかりません: {path.name}")
            return
        
        try:
            df_power, time_col = read_csv(path)
            self.append_ai_log(f"{path.name} を読み込みました (行数: {len(df_power):,}).")
            
            # 天候データと結合
            df_merged, has_weather = merge_weather_data(df_power, code, ym, time_col)
            
            if not has_weather:
                city_name = AREA_WEATHER_MAP.get(code, "不明")
                area_name = AREA_INFO.get(code, AreaInfo("不明", "")).name
                
                # 利用可能な天候ファイルをリストアップ
                available_files = []
                if WEATHER_DIR.exists():
                    available_files = [f.name for f in WEATHER_DIR.glob("*.csv")]
                
                msg = (
                    f"エリア: {code} ({area_name})\n"
                    f"対象年月: {ym[:4]}年{ym[4:6]}月\n"
                    f"期待ファイル: weather/{city_name}_YYYYMMDD_YYYYMMDD.csv\n\n"
                    "天候予測には天候データが必要です。\n\n"
                )
                
                if available_files:
                    msg += f"利用可能な天候ファイル:\n" + "\n".join([f"  - {f}" for f in available_files[:5]])
                else:
                    msg += "天候データフォルダにファイルがありません。"
                
                QtWidgets.QMessageBox.warning(self, "天候データなし", msg)
                self.append_ai_log(f"天候データが見つかりませんでした: {city_name}_*.csv (対象: {ym})")
                return
            
            self.append_ai_log(f"天候データを結合しました。")
            
            # 目的変数と外部変数を準備
            if col_name not in df_merged.columns:
                QtWidgets.QMessageBox.warning(self, "エラー", f"目的変数 '{col_name}' が見つかりません。")
                return
            
            # 目的変数（需要実績など）
            y = pd.to_numeric(df_merged[col_name], errors="coerce")
            
            # 外部変数（天候データ）を選択
            exog_cols = []
            if "temperature" in df_merged.columns:
                exog_cols.append("temperature")
            if "solar_radiation" in df_merged.columns:
                exog_cols.append("solar_radiation")
            elif "sunlight" in df_merged.columns:
                exog_cols.append("sunlight")
            if "wind_speed" in df_merged.columns:
                exog_cols.append("wind_speed")
            
            if not exog_cols:
                QtWidgets.QMessageBox.warning(self, "エラー", "利用可能な天候データがありません。")
                return
            
            self.append_ai_log(f"外部変数: {', '.join(exog_cols)}")
            
            # 外部変数のDataFrameを作成
            exog = df_merged[exog_cols].copy()
            for col in exog_cols:
                exog[col] = pd.to_numeric(exog[col], errors="coerce")
            
            # 欠損値を処理（線形補間）
            y = y.interpolate(method='linear', limit_direction='both').fillna(method='ffill').fillna(method='bfill')
            exog = exog.interpolate(method='linear', limit_direction='both').fillna(method='ffill').fillna(method='bfill')
            
            # 訓練/テストデータ分割
            train_ratio = self.train_ratio_spin.value()
            train_size = int(len(y) * train_ratio)
            
            y_train, y_test = y[:train_size], y[train_size:]
            exog_train, exog_test = exog[:train_size], exog[train_size:]
            
            if len(y_test) == 0:
                QtWidgets.QMessageBox.warning(self, "警告", "テストデータがありません。訓練データ比率を調整してください。")
                return
            
            self.append_ai_log(f"訓練データ: {len(y_train)}サンプル, テストデータ: {len(y_test)}サンプル")
            self.append_ai_log("SARIMAXモデルを学習中...")
            
            # SARIMAXモデル（外部変数あり）
            # order=(1,1,1): p=1, d=1, q=1 - パラメータは調整可能
            model = SARIMAX(y_train, exog=exog_train, order=(1, 1, 1), enforce_stationarity=False, enforce_invertibility=False)
            fitted = model.fit(disp=False, maxiter=50)
            
            self.append_ai_log("モデル学習完了。予測を実行中...")
            
            # 予測
            forecast_steps = min(self.ai_horizon_spin.value(), len(y_test))
            forecast = fitted.forecast(steps=forecast_steps, exog=exog_test[:forecast_steps])
            
            # 評価指標
            actual = y_test[:forecast_steps]
            mae = mean_absolute_error(actual, forecast)
            rmse = np.sqrt(mean_squared_error(actual, forecast))
            mape = np.mean(np.abs((actual - forecast) / actual)) * 100
            
            # プロット
            self.ai_result_canvas.fig.clear()
            ax = self.ai_result_canvas.fig.add_subplot(1, 1, 1)
            
            # 訓練データ
            ax.plot(range(len(y_train)), y_train.values, label='訓練データ', color='#0068B7', linewidth=1.5, alpha=0.8)
            
            # テストデータ
            test_idx = range(len(y_train), len(y_train) + len(actual))
            ax.plot(test_idx, actual.values, label='実測値', color='#10b981', linewidth=1.5)
            
            # 予測値
            ax.plot(test_idx, forecast.values, label='ARIMAX予測 (天候考慮)', color='#ef4444', linewidth=2, linestyle='--')
            
            ax.set_xlabel('時刻インデックス', fontsize=11)
            ax.set_ylabel(col_name, fontsize=11)
            ax.set_title(f'ARIMAX予測結果（天候データ考慮）', fontsize=14, fontweight='bold', color='#0068B7')
            ax.legend(loc='best', fontsize=10)
            ax.grid(True, alpha=0.3)
            
            self.ai_result_canvas.fig.tight_layout()
            self.ai_result_canvas.draw()
            
            # 評価結果
            self.ai_eval_widget.setPlainText(
                f"ARIMAX予測モデル評価（天候データ考慮）\n{'='*60}\n\n"
                f"モデル: SARIMAX(1, 1, 1)\n"
                f"訓練サンプル数: {len(y_train)}\n"
                f"予測期間: {forecast_steps}ステップ\n\n"
                f"外部変数（天候データ）:\n"
                f"  {', '.join([f'{col}' for col in exog_cols])}\n\n"
                f"評価指標:\n"
                f"  MAE  (平均絶対誤差):     {mae:.4f}\n"
                f"  RMSE (二乗平均平方根誤差): {rmse:.4f}\n"
                f"  MAPE (平均絶対パーセント誤差): {mape:.2f}%\n\n"
                f"モデル要約:\n{fitted.summary().as_text()}"
            )
            
            # 残差分析
            residuals = actual.values - forecast.values
            self.plot_residual_analysis(pd.Series(residuals))
            
            self.append_ai_log(f"ARIMAX予測完了 - MAE: {mae:.2f}, RMSE: {rmse:.2f}, MAPE: {mape:.2f}%")
            self.ai_tabs.setCurrentIndex(1)
            
        except Exception as e:
            import traceback
            self.append_ai_log(f"エラー: {str(e)}")
            self.append_ai_log(traceback.format_exc())
            QtWidgets.QMessageBox.critical(self, "エラー", f"天候予測に失敗しました:\n{str(e)}")

    def prepare_ai_dataset(self) -> None:
        if self.ai_dataframe is None:
            self.load_ai_dataset()
        if self.ai_dataframe is None:
            QtWidgets.QMessageBox.warning(self, "警告", "データが読み込まれていません。")
            return
        target_column = self.ai_column_combo.currentText()
        if not target_column:
            QtWidgets.QMessageBox.warning(self, "警告", "目的列を選択してください。")
            return
        series = pd.to_numeric(self.ai_dataframe[target_column], errors="coerce")
        valid = series.dropna()
        self.ai_target_series = series
        self.ai_training_index = valid.index
        self.append_ai_log(
            f"列'{target_column}' を読み込みました。利用可能なサンプル: {len(valid):,}件 / 欠損 {series.isna().sum():,}件"
        )
        context_len = min(self.ai_context_spin.value(), len(valid))
        if context_len == 0:
            self.append_ai_log("有効なデータがありません。CSVを確認してください。")
            return

        history_values = valid.iloc[-context_len:]
        timestamps = None
        if self.ai_time_column and self.ai_time_column in self.ai_dataframe.columns:
            try:
                time_series = pd.to_datetime(
                    self.ai_dataframe[self.ai_time_column], errors="coerce"
                )
                timestamps = time_series.reindex(history_values.index)
            except Exception:
                timestamps = None
        self.display_history_preview(history_values, timestamps)

    def display_history_preview(self, values: pd.Series, timestamps: Optional[pd.Series]) -> None:
        # Show the latest context window in the result table.

        self.ai_result_table.clear()
        self.ai_result_table.setColumnCount(3)
        self.ai_result_table.setHorizontalHeaderLabels(["種別", "タイムスタンプ", "値"])
        self.ai_result_table.setRowCount(len(values))
        for row_idx, (idx, value) in enumerate(values.items()):
            ts_text = ""
            if timestamps is not None:
                try:
                    ts = timestamps.loc[idx]
                    if pd.notna(ts):
                        ts_text = pd.to_datetime(ts).strftime("%Y-%m-%d %H:%M")
                except Exception:
                    ts_text = str(idx)
            else:
                ts_text = str(idx)
            self.ai_result_table.setItem(row_idx, 0, QTableWidgetItem("螻･豁ｴ"))
            self.ai_result_table.setItemItem(row_idx, 1, QTableWidgetItem(ts_text))
            self.ai_result_table.setItem(row_idx, 2, QTableWidgetItem(f"{float(value):,.2f}"))

    def train_transformer_model(self) -> None:
        if self.ai_dataframe is None:
            self.load_ai_dataset()
        if self.ai_dataframe is None:
            QtWidgets.QMessageBox.warning(self, "警告", "データが読み込まれていません。")
            return
        target_column = self.ai_column_combo.currentText()
        if not target_column:
            QtWidgets.QMessageBox.warning(self, "警告", "目的列を選択してください。")
            return
        if self.ai_target_series is None:
            self.prepare_ai_dataset()
        if self.ai_target_series is None:
            return
        series = pd.to_numeric(self.ai_target_series, errors="coerce")
        series_interpolated = (
            series.interpolate(limit_direction="both")
            .bfill()
            .ffill()
        )
        context_length = self.ai_context_spin.value()
        prediction_length = self.ai_horizon_spin.value()
        if len(series_interpolated.dropna()) < context_length + prediction_length:
            QtWidgets.QMessageBox.warning(
                self,
                "警告",
                "学習に有効なデータがありません。コンテキスト長や分析ステップ数を調整してください。",
            )
            return

        epochs = self.ai_epoch_spin.value()
        batch_size = self.ai_batch_spin.value()
        learning_rate = self.ai_lr_spin.value()

        self.append_ai_log(
            f"Transformerを初期化しました(context={context_length}, horizon={prediction_length}, epochs={epochs})."
        )
        self.ai_forecaster = DemandTransformerForecaster(
            context_length=context_length,
            prediction_length=prediction_length,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
        )
        try:
            log = self.ai_forecaster.fit(series_interpolated.to_numpy(), validation_split=0.2)
        except Exception as exc:
            self.append_ai_log(f"学習中にエラーが発生しました: {exc}")
            QtWidgets.QMessageBox.critical(self, "エラー", f"学習中にエラーが発生しました: {exc}")
            return

        final_train = log.train_loss[-1] if log.train_loss else float("nan")
        final_val = log.val_loss[-1] if log.val_loss and log.val_loss[-1] is not None else None
        summary = f"学習完了 - train_loss={final_train:.6f}"
        if final_val is not None:
            summary += f", val_loss={final_val:.6f}"
        self.append_ai_log(summary)

        try:
            result = self.ai_forecaster.predict(series_interpolated.to_numpy())
        except Exception as exc:
            self.append_ai_log(f"予測に失敗しました: {exc}")
            QtWidgets.QMessageBox.critical(self, "エラー", str(exc))
            return

        self.display_forecast_result(result, series_interpolated.index)

    def display_forecast_result(self, result: ForecastResult, index: pd.Index) -> None:
        # Render forecast results in the AI result table.

        history_len = len(result.history)
        prediction_len = len(result.prediction)
        self.ai_result_table.clear()
        self.ai_result_table.setColumnCount(3)
        self.ai_result_table.setHorizontalHeaderLabels(["種別", "タイムスタンプ", "値"])
        self.ai_result_table.setRowCount(history_len + prediction_len)

        history_index = index[-history_len:] if history_len <= len(index) else pd.RangeIndex(history_len)
        if self.ai_time_column and self.ai_dataframe is not None and self.ai_time_column in self.ai_dataframe.columns:
            try:
                time_series = pd.to_datetime(self.ai_dataframe[self.ai_time_column], errors="coerce")
                history_times = time_series.iloc[-history_len:].reset_index(drop=True)
            except Exception:
                history_times = pd.Series([None] * history_len)
        else:
            history_times = pd.Series([None] * history_len)

        for row, (idx, value) in enumerate(zip(history_index, result.history)):
            ts_text = ""
            if row < len(history_times) and history_times.iloc[row] is not None and pd.notna(history_times.iloc[row]):
                ts_text = pd.to_datetime(history_times.iloc[row]).strftime("%Y-%m-%d %H:%M")
            else:
                ts_text = str(idx)
            self.ai_result_table.setItem(row, 0, QTableWidgetItem("螻･豁ｴ"))
            self.ai_result_table.setItem(row, 1, QTableWidgetItem(ts_text))
            self.ai_result_table.setItem(row, 2, QTableWidgetItem(f"{float(value):,.2f}"))

        future_times: List[str] = []
        if self.ai_time_column and self.ai_dataframe is not None and self.ai_time_column in self.ai_dataframe.columns:
            try:
                time_series = pd.to_datetime(self.ai_dataframe[self.ai_time_column], errors="coerce")
                valid_times = time_series.dropna()
                if len(valid_times) >= 2:
                    inferred_freq = valid_times.iloc[-1] - valid_times.iloc[-2]
                    if inferred_freq == pd.Timedelta(0):
                        inferred_freq = None
                else:
                    inferred_freq = None
                last_time = valid_times.iloc[-1] if len(valid_times) else None
                if inferred_freq is not None and last_time is not None:
                    future_times = [
                        (last_time + inferred_freq * (i + 1)).strftime("%Y-%m-%d %H:%M")
                        for i in range(prediction_len)
                    ]
            except Exception:
                future_times = []

        for i, value in enumerate(result.prediction):
            row = history_len + i
            ts_text = future_times[i] if i < len(future_times) else f"t+{i + 1}"
            self.ai_result_table.setItem(row, 0, QTableWidgetItem("予測"))
            self.ai_result_table.setItem(row, 1, QTableWidgetItem(ts_text))
            self.ai_result_table.setItem(row, 2, QTableWidgetItem(f"{float(value):,.2f}"))
    
    def create_data_selection_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(12)

        title = QLabel("データ選択 / フィルター")
        title.setStyleSheet("font-size: 16px; font-weight: 600; color: #0068B7;")
        layout.addWidget(title)

        # Area selector
        area_group = QGroupBox("対象エリア")
        area_layout = QVBoxLayout()
        self.area_combo = QComboBox()
        self.area_combo.setMinimumHeight(36)
        self.area_combo.currentIndexChanged.connect(self.on_area_change)
        area_layout.addWidget(self.area_combo)
        area_group.setLayout(area_layout)
        layout.addWidget(area_group)

        # Year-month selector (分離版)
        ym_group = QGroupBox("対象の年月")
        ym_layout = QGridLayout()
        
        # 年選択
        ym_layout.addWidget(QLabel("年:"), 0, 0)
        self.year_combo = QComboBox()
        self.year_combo.setMinimumHeight(36)
        self.year_combo.currentIndexChanged.connect(self.on_year_month_change)
        ym_layout.addWidget(self.year_combo, 0, 1)
        
        # 月選択
        ym_layout.addWidget(QLabel("月:"), 1, 0)
        self.month_combo = QComboBox()
        self.month_combo.setMinimumHeight(36)
        self.month_combo.currentIndexChanged.connect(self.on_year_month_change)
        ym_layout.addWidget(self.month_combo, 1, 1)
        
        ym_group.setLayout(ym_layout)
        layout.addWidget(ym_group)
        
        # 後方互換性のため、ym_comboも保持（非表示）
        self.ym_combo = QComboBox()
        self.ym_combo.setVisible(False)

        # Date filter selector
        date_group = QGroupBox("日付範囲")
        date_layout = QVBoxLayout()
        self.date_combo = QComboBox()
        self.date_combo.setMinimumHeight(36)
        self.date_combo.addItem("全期間", "all")
        self.date_combo.currentIndexChanged.connect(self.on_date_change)
        date_layout.addWidget(self.date_combo)
        date_group.setLayout(date_layout)
        layout.addWidget(date_group)

        # Column selection area
        column_group = QGroupBox("表示する列")
        column_layout = QVBoxLayout()

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMinimumHeight(300)
        scroll_widget = QWidget()
        self.column_checkbox_layout = QVBoxLayout(scroll_widget)
        self.column_checkbox_layout.setAlignment(Qt.AlignTop)
        self.column_checkboxes: dict[str, QCheckBox] = {}
        scroll.setWidget(scroll_widget)
        column_layout.addWidget(scroll)

        btn_row = QHBoxLayout()
        select_all_btn = QPushButton("全選択")
        select_all_btn.clicked.connect(self.select_all_columns)
        deselect_all_btn = QPushButton("全解除")
        deselect_all_btn.clicked.connect(self.deselect_all_columns)
        btn_row.addWidget(select_all_btn)
        btn_row.addWidget(deselect_all_btn)
        column_layout.addLayout(btn_row)

        column_group.setLayout(column_layout)
        layout.addWidget(column_group)

        # Action buttons
        self.view_btn = QPushButton("📊 グラフ更新")
        self.view_btn.setMinimumHeight(44)
        self.view_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                          stop:0 #0068B7, stop:1 #005291);
                color: white;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                          stop:0 #0080e0, stop:1 #0068B7);
            }
        """)
        self.view_btn.clicked.connect(self.render_view)
        layout.addWidget(self.view_btn)

        self.add_to_collection_btn = QPushButton("📌 コレクションに追加")
        self.add_to_collection_btn.setMinimumHeight(40)
        self.add_to_collection_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                          stop:0 #10b981, stop:1 #059669);
                color: white;
                font-weight: 600;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                          stop:0 #34d399, stop:1 #10b981);
            }
        """)
        self.add_to_collection_btn.clicked.connect(self.add_graph_to_collection)
        layout.addWidget(self.add_to_collection_btn)

        # 統計分析タブへデータを移行するボタン
        self.transfer_to_analysis_btn = QPushButton("📈 統計分析タブで分析")
        self.transfer_to_analysis_btn.setMinimumHeight(40)
        self.transfer_to_analysis_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                          stop:0 #8b5cf6, stop:1 #7c3aed);
                color: white;
                font-weight: 600;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                          stop:0 #a78bfa, stop:1 #8b5cf6);
            }
        """)
        self.transfer_to_analysis_btn.clicked.connect(self.transfer_to_analysis_tab)
        layout.addWidget(self.transfer_to_analysis_btn)

        layout.addStretch()
        return panel
    
    def create_graph_settings_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(12)

        title = QLabel("グラフ設定")
        title.setStyleSheet("font-size: 16px; font-weight: 600; color: #0068B7;")
        layout.addWidget(title)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_widget = QWidget()
        settings_layout = QVBoxLayout(scroll_widget)
        settings_layout.setSpacing(12)

        title_group = QGroupBox("タイトル")
        title_layout = QVBoxLayout()
        self.title_input = QLineEdit(self.graph_settings["title"])
        self.title_input.setPlaceholderText("グラフタイトルを入力")
        self.title_input.textChanged.connect(lambda: self.update_setting("title", self.title_input.text()))
        title_layout.addWidget(self.title_input)
        title_group.setLayout(title_layout)
        settings_layout.addWidget(title_group)

        label_group = QGroupBox("軸ラベル")
        label_layout = QGridLayout()
        self.xlabel_input = QLineEdit(self.graph_settings["xlabel"])
        self.xlabel_input.textChanged.connect(lambda: self.update_setting("xlabel", self.xlabel_input.text()))
        self.ylabel_input = QLineEdit(self.graph_settings["ylabel"])
        self.ylabel_input.textChanged.connect(lambda: self.update_setting("ylabel", self.ylabel_input.text()))
        label_layout.addWidget(QLabel("X軸:"), 0, 0)
        label_layout.addWidget(self.xlabel_input, 0, 1)
        label_layout.addWidget(QLabel("Y軸:"), 1, 0)
        label_layout.addWidget(self.ylabel_input, 1, 1)
        label_group.setLayout(label_layout)
        settings_layout.addWidget(label_group)

        line_group = QGroupBox("線の太さ")
        line_layout = QHBoxLayout()
        self.linewidth_spin = QDoubleSpinBox()
        self.linewidth_spin.setRange(0.5, 10.0)
        self.linewidth_spin.setSingleStep(0.5)
        self.linewidth_spin.setValue(self.graph_settings["linewidth"])
        self.linewidth_spin.valueChanged.connect(lambda: self.update_setting("linewidth", self.linewidth_spin.value()))
        line_layout.addWidget(self.linewidth_spin)
        line_group.setLayout(line_layout)
        settings_layout.addWidget(line_group)

        size_group = QGroupBox("図のサイズ (インチ)")
        size_layout = QGridLayout()
        self.width_spin = QSpinBox()
        self.width_spin.setRange(4, 20)
        self.width_spin.setValue(self.graph_settings["figsize_w"])
        self.width_spin.valueChanged.connect(lambda: self.update_setting("figsize_w", self.width_spin.value()))
        self.height_spin = QSpinBox()
        self.height_spin.setRange(3, 15)
        self.height_spin.setValue(self.graph_settings["figsize_h"])
        self.height_spin.valueChanged.connect(lambda: self.update_setting("figsize_h", self.height_spin.value()))
        size_layout.addWidget(QLabel("幅:"), 0, 0)
        size_layout.addWidget(self.width_spin, 0, 1)
        size_layout.addWidget(QLabel("高さ:"), 1, 0)
        size_layout.addWidget(self.height_spin, 1, 1)
        size_group.setLayout(size_layout)
        settings_layout.addWidget(size_group)

        dpi_group = QGroupBox("DPI (解像度)")
        dpi_layout = QHBoxLayout()
        self.dpi_spin = QSpinBox()
        self.dpi_spin.setRange(50, 600)
        self.dpi_spin.setSingleStep(10)
        self.dpi_spin.setValue(self.graph_settings["dpi"])
        self.dpi_spin.valueChanged.connect(lambda: self.update_setting("dpi", self.dpi_spin.value()))
        dpi_layout.addWidget(self.dpi_spin)
        dpi_group.setLayout(dpi_layout)
        settings_layout.addWidget(dpi_group)

        font_group = QGroupBox("フォントサイズ")
        font_layout = QGridLayout()
        self.font_size_spin = QSpinBox()
        self.font_size_spin.setRange(6, 24)
        self.font_size_spin.setValue(self.graph_settings["font_size"])
        self.font_size_spin.valueChanged.connect(lambda: self.update_setting("font_size", self.font_size_spin.value()))
        self.title_size_spin = QSpinBox()
        self.title_size_spin.setRange(8, 36)
        self.title_size_spin.setValue(self.graph_settings["title_size"])
        self.title_size_spin.valueChanged.connect(lambda: self.update_setting("title_size", self.title_size_spin.value()))
        self.label_size_spin = QSpinBox()
        self.label_size_spin.setRange(6, 24)
        self.label_size_spin.setValue(self.graph_settings.get("label_size", 12))
        self.label_size_spin.valueChanged.connect(lambda: self.update_setting("label_size", self.label_size_spin.value()))
        font_layout.addWidget(QLabel("凡例/目盛:"), 0, 0)
        font_layout.addWidget(self.font_size_spin, 0, 1)
        font_layout.addWidget(QLabel("軸ラベル:"), 1, 0)
        font_layout.addWidget(self.label_size_spin, 1, 1)
        font_layout.addWidget(QLabel("タイトル:"), 2, 0)
        font_layout.addWidget(self.title_size_spin, 2, 1)
        font_group.setLayout(font_layout)
        settings_layout.addWidget(font_group)

        options_group = QGroupBox("表示オプション")
        options_layout = QVBoxLayout()
        
        # タイトル表示
        self.show_title_check = QCheckBox("タイトルを表示")
        self.show_title_check.setChecked(self.graph_settings.get("show_title", True))
        self.show_title_check.toggled.connect(lambda: self.update_setting("show_title", self.show_title_check.isChecked()))
        options_layout.addWidget(self.show_title_check)
        
        # X軸ラベル表示
        self.show_xlabel_check = QCheckBox("X軸ラベルを表示")
        self.show_xlabel_check.setChecked(self.graph_settings.get("show_xlabel", True))
        self.show_xlabel_check.toggled.connect(lambda: self.update_setting("show_xlabel", self.show_xlabel_check.isChecked()))
        options_layout.addWidget(self.show_xlabel_check)
        
        # Y軸ラベル表示
        self.show_ylabel_check = QCheckBox("Y軸ラベルを表示")
        self.show_ylabel_check.setChecked(self.graph_settings.get("show_ylabel", True))
        self.show_ylabel_check.toggled.connect(lambda: self.update_setting("show_ylabel", self.show_ylabel_check.isChecked()))
        options_layout.addWidget(self.show_ylabel_check)
        
        # グリッド表示
        self.grid_check = QCheckBox("グリッドを表示")
        self.grid_check.setChecked(self.graph_settings["grid"])
        self.grid_check.toggled.connect(lambda: self.update_setting("grid", self.grid_check.isChecked()))
        options_layout.addWidget(self.grid_check)
        
        # 凡例表示
        self.show_legend_check = QCheckBox("凡例を表示")
        self.show_legend_check.setChecked(self.graph_settings.get("show_legend", True))
        self.show_legend_check.toggled.connect(lambda: self.update_setting("show_legend", self.show_legend_check.isChecked()))
        options_layout.addWidget(self.show_legend_check)
        self.legend_check = self.show_legend_check  # 後方互換
        
        # 詳細目盛り
        self.detailed_ticks_check = QCheckBox("軸の目盛りを細かく表示")
        self.detailed_ticks_check.setChecked(self.graph_settings.get("detailed_ticks", False))
        self.detailed_ticks_check.setToolTip("拡大表示時に軸の目盛りをより細かく表示します")
        self.detailed_ticks_check.toggled.connect(lambda: self.update_setting("detailed_ticks", self.detailed_ticks_check.isChecked()))
        legend_row = QHBoxLayout()
        legend_row.addWidget(QLabel("凡例の位置:"))
        self.legend_loc_combo = QComboBox()
        self.legend_loc_combo.addItems([
            "best",
            "upper right",
            "upper left",
            "lower left",
            "lower right",
            "right",
            "center left",
            "center right",
            "lower center",
            "upper center",
            "center",
        ])
        self.legend_loc_combo.setCurrentText(self.graph_settings["legend_loc"])
        self.legend_loc_combo.currentTextChanged.connect(lambda value: self.update_setting("legend_loc", value))
        legend_row.addWidget(self.legend_loc_combo)
        legend_row.addStretch()
        options_layout.addLayout(legend_row)
        options_group.setLayout(options_layout)
        settings_layout.addWidget(options_group)

        # 研究発表用設定グループ
        research_group = QGroupBox("📊 研究発表用設定")
        research_group.setStyleSheet("""
            QGroupBox {
                font-weight: 600;
                color: #0068B7;
                border: 2px solid #a0d2ff;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
        """)
        research_layout = QVBoxLayout()
        research_layout.setSpacing(8)
        
        # 曜日表示オプション
        self.show_weekday_check = QCheckBox("日付に曜日を併記する")
        self.show_weekday_check.setChecked(self.graph_settings.get("show_weekday", False))
        self.show_weekday_check.setToolTip("X軸の日付ラベルに曜日を表示します")
        self.show_weekday_check.toggled.connect(lambda: self.update_setting("show_weekday", self.show_weekday_check.isChecked()))
        research_layout.addWidget(self.show_weekday_check)
        
        # 曜日フォーマット選択
        weekday_format_row = QHBoxLayout()
        weekday_format_row.addWidget(QLabel("曜日の形式:"))
        self.weekday_format_combo = QComboBox()
        self.weekday_format_combo.addItems([
            "short",     # (月)
            "full",      # 月曜日
            "en_short",  # Mon
            "en_full",   # Monday
        ])
        self.weekday_format_combo.setCurrentText(self.graph_settings.get("weekday_format", "short"))
        self.weekday_format_combo.currentTextChanged.connect(lambda value: self.update_setting("weekday_format", value))
        weekday_format_row.addWidget(self.weekday_format_combo)
        weekday_format_row.addStretch()
        research_layout.addLayout(weekday_format_row)
        
        # 週の境界線表示
        self.show_week_boundaries_check = QCheckBox("週の境界に縦線を表示")
        self.show_week_boundaries_check.setChecked(self.graph_settings.get("show_week_boundaries", False))
        self.show_week_boundaries_check.setToolTip("週の始まり/終わりに縦線を描画します")
        self.show_week_boundaries_check.toggled.connect(lambda: self.update_setting("show_week_boundaries", self.show_week_boundaries_check.isChecked()))
        research_layout.addWidget(self.show_week_boundaries_check)
        
        # 週の始まりの曜日
        week_start_row = QHBoxLayout()
        week_start_row.addWidget(QLabel("週の始まり:"))
        self.week_boundary_day_combo = QComboBox()
        self.week_boundary_day_combo.addItems(["monday", "sunday"])
        self.week_boundary_day_combo.setCurrentText(self.graph_settings.get("week_boundary_day", "monday"))
        self.week_boundary_day_combo.currentTextChanged.connect(lambda value: self.update_setting("week_boundary_day", value))
        week_start_row.addWidget(self.week_boundary_day_combo)
        week_start_row.addStretch()
        research_layout.addLayout(week_start_row)
        
        # 深夜0時のラベル形式
        midnight_row = QHBoxLayout()
        midnight_row.addWidget(QLabel("24:00/0:00の表記:"))
        self.midnight_label_combo = QComboBox()
        self.midnight_label_combo.addItem("翌日の0:00 (例: 2日0:00)", "next_day")
        self.midnight_label_combo.addItem("当日の24:00 (例: 1日24:00)", "same_day")
        idx = self.midnight_label_combo.findData(self.graph_settings.get("midnight_label_format", "next_day"))
        if idx >= 0:
            self.midnight_label_combo.setCurrentIndex(idx)
        self.midnight_label_combo.currentIndexChanged.connect(
            lambda: self.update_setting("midnight_label_format", self.midnight_label_combo.currentData())
        )
        midnight_row.addWidget(self.midnight_label_combo)
        midnight_row.addStretch()
        research_layout.addLayout(midnight_row)
        
        # セパレーター
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setFrameShadow(QFrame.Sunken)
        separator.setStyleSheet("background-color: #bfdbfe;")
        research_layout.addWidget(separator)
        
        # 横軸日付フィルター
        xaxis_filter_label = QLabel("📅 横軸に表示する日付の制御")
        xaxis_filter_label.setStyleSheet("font-weight: 600; color: #0068B7; margin-top: 8px;")
        research_layout.addWidget(xaxis_filter_label)
        
        # フィルタータイプ選択
        xaxis_filter_type_row = QHBoxLayout()
        xaxis_filter_type_row.addWidget(QLabel("表示モード:"))
        self.xaxis_date_filter_combo = QComboBox()
        self.xaxis_date_filter_combo.addItem("すべての日付", "all")
        self.xaxis_date_filter_combo.addItem("特定の曜日のみ", "specific_weekdays")
        self.xaxis_date_filter_combo.addItem("N日おきに表示", "every_n_days")
        self.xaxis_date_filter_combo.setToolTip("横軸に表示する日付の範囲を制御します")
        idx = self.xaxis_date_filter_combo.findData(self.graph_settings.get("xaxis_date_filter", "all"))
        if idx >= 0:
            self.xaxis_date_filter_combo.setCurrentIndex(idx)
        self.xaxis_date_filter_combo.currentIndexChanged.connect(
            lambda: self.on_xaxis_filter_mode_changed()
        )
        xaxis_filter_type_row.addWidget(self.xaxis_date_filter_combo)
        xaxis_filter_type_row.addStretch()
        research_layout.addLayout(xaxis_filter_type_row)
        
        # 曜日選択（特定の曜日モード用）
        self.xaxis_weekday_frame = QFrame()
        xaxis_weekday_layout = QVBoxLayout(self.xaxis_weekday_frame)
        xaxis_weekday_layout.setContentsMargins(15, 5, 0, 5)
        xaxis_weekday_layout.setSpacing(5)
        
        weekday_label = QLabel("表示する曜日:")
        weekday_label.setStyleSheet("color: #6b7280; font-size: 11px;")
        xaxis_weekday_layout.addWidget(weekday_label)
        
        self.xaxis_weekday_checks = {}
        weekday_names = ["月曜", "火曜", "水曜", "木曜", "金曜", "土曜", "日曜"]
        weekday_grid = QGridLayout()
        weekday_grid.setSpacing(5)
        for i, name in enumerate(weekday_names):
            check = QCheckBox(name)
            check.setChecked(i in self.graph_settings.get("xaxis_weekdays", [0, 1, 2, 3, 4, 5, 6]))
            check.toggled.connect(lambda checked, idx=i: self.on_xaxis_weekday_changed())
            self.xaxis_weekday_checks[i] = check
            weekday_grid.addWidget(check, i // 4, i % 4)
        xaxis_weekday_layout.addLayout(weekday_grid)
        research_layout.addWidget(self.xaxis_weekday_frame)
        
        # N日おき設定（every_n_daysモード用）
        self.xaxis_every_n_frame = QFrame()
        xaxis_every_n_layout = QHBoxLayout(self.xaxis_every_n_frame)
        xaxis_every_n_layout.setContentsMargins(15, 5, 0, 5)
        xaxis_every_n_layout.addWidget(QLabel("間隔:"))
        self.xaxis_every_n_spin = QSpinBox()
        self.xaxis_every_n_spin.setRange(1, 30)
        self.xaxis_every_n_spin.setValue(self.graph_settings.get("xaxis_every_n_days", 1))
        self.xaxis_every_n_spin.setSuffix(" 日おき")
        self.xaxis_every_n_spin.setToolTip("何日おきに日付ラベルを表示するか")
        self.xaxis_every_n_spin.valueChanged.connect(
            lambda: self.update_setting("xaxis_every_n_days", self.xaxis_every_n_spin.value())
        )
        xaxis_every_n_layout.addWidget(self.xaxis_every_n_spin)
        xaxis_every_n_layout.addStretch()
        research_layout.addWidget(self.xaxis_every_n_frame)
        
        # X軸ラベル回転角度
        rotation_row = QHBoxLayout()
        rotation_row.addWidget(QLabel("ラベル回転角度:"))
        self.xaxis_rotation_spin = QSpinBox()
        self.xaxis_rotation_spin.setRange(0, 90)
        self.xaxis_rotation_spin.setValue(self.graph_settings.get("xaxis_tick_rotation", 45))
        self.xaxis_rotation_spin.setSuffix(" °")
        self.xaxis_rotation_spin.setToolTip("X軸ラベルの回転角度（見やすさ調整用）")
        self.xaxis_rotation_spin.valueChanged.connect(
            lambda: self.update_setting("xaxis_tick_rotation", self.xaxis_rotation_spin.value())
        )
        rotation_row.addWidget(self.xaxis_rotation_spin)
        rotation_row.addStretch()
        research_layout.addLayout(rotation_row)
        
        # 初期表示状態を設定
        self.on_xaxis_filter_mode_changed()
        
        research_group.setLayout(research_layout)
        settings_layout.addWidget(research_group)

        save_btn = QPushButton("💾 グラフを保存")
        save_btn.setMinimumHeight(36)
        save_btn.clicked.connect(self.save_graph)
        settings_layout.addWidget(save_btn)
        settings_layout.addStretch()

        scroll.setWidget(scroll_widget)
        layout.addWidget(scroll)
        return panel
    
    def create_preview_panel(self) -> QWidget:
        """データプレビュー用パネルを作成"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(12)

        frame_style = """
            QFrame {
                border: 2px solid #a0d2ff;
                border-radius: 10px;
                background-color: #ffffff;
                padding: 10px;
            }
        """

        table_container = QFrame()
        table_container.setStyleSheet(frame_style)
        table_layout = QVBoxLayout(table_container)
        table_layout.setContentsMargins(5, 5, 5, 5)
        table_layout.setSpacing(6)

        self.preview_info_label = QLabel("データを読み込むとプレビューを表示します")
        self.preview_info_label.setStyleSheet("font-weight: 600; color: #0068B7; font-size: 14px;")
        table_layout.addWidget(self.preview_info_label)

        self.preview_table = QTableWidget()
        self.preview_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.preview_table.setAlternatingRowColors(True)
        self.preview_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.preview_table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.preview_table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self.preview_table.verticalHeader().setVisible(False)
        self.preview_table.setStyleSheet(
            """
            QTableWidget {
                background-color: #ffffff;
                alternate-background-color: #f8fafc;
                border: none;
                color: #2d3748;
                font-size: 12px;
            }
            """
        )
        table_layout.addWidget(self.preview_table)
        layout.addWidget(table_container)
        return panel

    def create_graph_only_panel(self) -> QWidget:
        """グラフ表示専用パネルを作成"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(12)

        frame_style = """
            QFrame {
                border: 2px solid #a0d2ff;
                border-radius: 10px;
                background-color: #ffffff;
                padding: 10px;
            }
        """

        canvas_container = QFrame()
        canvas_container.setStyleSheet(frame_style)
        canvas_layout = QVBoxLayout(canvas_container)
        canvas_layout.setContentsMargins(5, 5, 5, 5)

        self.canvas = MplCanvas(
            width=self.graph_settings["figsize_w"],
            height=self.graph_settings["figsize_h"],
            dpi=self.graph_settings["dpi"],
        )
        
        # NavigationToolbarを追加（ズーム・パン機能）
        self.toolbar = NavigationToolbar(self.canvas, panel)
        self.toolbar.setStyleSheet("""
            QToolBar {
                background-color: #f0f9ff;
                border: 1px solid #bfdbfe;
                border-radius: 5px;
                padding: 3px;
            }
            QToolButton {
                background-color: transparent;
                border: none;
                padding: 5px;
            }
            QToolButton:hover {
                background-color: #dbeafe;
                border-radius: 3px;
            }
        """)
        canvas_layout.addWidget(self.toolbar)
        canvas_layout.addWidget(self.canvas, stretch=1)
        layout.addWidget(canvas_container, stretch=1)
        
        return panel

    def create_collection_panel(self) -> QWidget:
        """並列比較用コレクションパネルを作成"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(12)

        # コレクションヘッダー
        collection_header = QHBoxLayout()
        collection_header.setSpacing(10)
        collection_title = QLabel("🗂 並列比較用グラフコレクション")
        collection_title.setStyleSheet("font-size: 16px; font-weight: 600; color: #0068B7;")
        collection_header.addWidget(collection_title)
        collection_header.addStretch()
        self.collection_info_label = QLabel("0件を保存中")
        self.collection_info_label.setStyleSheet("color: #4a5568; font-size: 13px;")
        collection_header.addWidget(self.collection_info_label)
        
        # 重ね合わせ表示ボタン
        overlay_btn = QPushButton("📊 選択したグラフを重ね合わせ")
        overlay_btn.setMinimumHeight(36)
        overlay_btn.setMaximumWidth(200)
        overlay_btn.setStyleSheet("""
            QPushButton {
                background: #10b981;
                color: white;
                font-weight: 600;
                border-radius: 5px;
                padding: 0 15px;
            }
            QPushButton:hover {
                background: #059669;
            }
        """)
        overlay_btn.clicked.connect(self.overlay_selected_graphs)
        collection_header.addWidget(overlay_btn)
        
        clear_btn = QPushButton("🧹 全削除")
        clear_btn.setMinimumHeight(36)
        clear_btn.setMaximumWidth(120)
        clear_btn.setStyleSheet("""
            QPushButton {
                background: #ef4444;
                color: white;
                font-weight: 600;
                border-radius: 5px;
                padding: 0 15px;
            }
            QPushButton:hover {
                background: #dc2626;
            }
        """)
        clear_btn.clicked.connect(self.clear_graph_collection)
        collection_header.addWidget(clear_btn)
        layout.addLayout(collection_header)

        # 重ね合わせモード選択
        overlay_mode_frame = QFrame()
        overlay_mode_frame.setStyleSheet("""
            QFrame {
                background-color: #f0f9ff;
                border: 1px solid #bfdbfe;
                border-radius: 5px;
                padding: 8px;
            }
        """)
        overlay_mode_layout = QHBoxLayout(overlay_mode_frame)
        overlay_mode_layout.setContentsMargins(10, 8, 10, 8)
        
        overlay_mode_label = QLabel("重ね合わせモード:")
        overlay_mode_label.setStyleSheet("font-weight: 600; color: #0068B7;")
        overlay_mode_layout.addWidget(overlay_mode_label)
        
        self.overlay_mode_group = QtWidgets.QButtonGroup(panel)
        self.overlay_time_radio = QtWidgets.QRadioButton("時系列順（時刻で結合）")
        self.overlay_time_radio.setToolTip("時刻データを基準に、同じ時刻のデータを重ね合わせます")
        self.overlay_time_radio.setChecked(True)
        self.overlay_index_radio = QtWidgets.QRadioButton("X軸共有（値の単純比較）")
        self.overlay_index_radio.setToolTip("X軸のインデックスを共有し、データポイントの順序で比較します")
        
        self.overlay_mode_group.addButton(self.overlay_time_radio)
        self.overlay_mode_group.addButton(self.overlay_index_radio)
        
        overlay_mode_layout.addWidget(self.overlay_time_radio)
        overlay_mode_layout.addWidget(self.overlay_index_radio)
        overlay_mode_layout.addStretch()
        
        layout.addWidget(overlay_mode_frame)
        
        # 説明文
        info_label = QLabel("「📌 コレクションに追加」ボタンで現在のグラフを保存できます。チェックボックスで複数選択し、「📊 選択したグラフを重ね合わせ」で同一グラフに表示できます。")
        info_label.setStyleSheet("color: #6b7280; font-size: 12px; padding: 5px 10px;")
        info_label.setWordWrap(True)
        layout.addWidget(info_label)

        # コレクションスクロールエリア
        self.graph_collection_scroll = QScrollArea()
        self.graph_collection_scroll.setWidgetResizable(True)
        self.graph_collection_scroll.setStyleSheet("""
            QScrollArea {
                border: 2px solid #a0d2ff;
                border-radius: 10px;
                background-color: #f9fafb;
            }
        """)
        self.graph_collection_container = QWidget()
        self.graph_collection_layout = QVBoxLayout(self.graph_collection_container)
        self.graph_collection_layout.setContentsMargins(10, 10, 10, 10)
        self.graph_collection_layout.setSpacing(15)
        self.graph_collection_layout.setAlignment(Qt.AlignTop)
        self.graph_collection_scroll.setWidget(self.graph_collection_container)
        layout.addWidget(self.graph_collection_scroll, stretch=1)
        
        return panel

    def create_graph_panel(self) -> QWidget:
        """後方互換性のために残すダミー関数（グラフ+コレクション統合版）"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(12)

        frame_style = """
            QFrame {
                border: 2px solid #a0d2ff;
                border-radius: 10px;
                background-color: #ffffff;
                padding: 10px;
            }
        """

        canvas_container = QFrame()
        canvas_container.setStyleSheet(frame_style)
        canvas_layout = QVBoxLayout(canvas_container)
        canvas_layout.setContentsMargins(5, 5, 5, 5)

        self.canvas = MplCanvas(
            width=self.graph_settings["figsize_w"],
            height=self.graph_settings["figsize_h"],
            dpi=self.graph_settings["dpi"],
        )
        canvas_layout.addWidget(self.canvas, stretch=1)
        layout.addWidget(canvas_container, stretch=1)

        # コレクション部分を追加
        collection_header = QHBoxLayout()
        collection_header.setSpacing(10)
        collection_title = QLabel("🗂 並列比較用グラフコレクション")
        collection_title.setStyleSheet("font-size: 15px; font-weight: 600; color: #0068B7;")
        collection_header.addWidget(collection_title)
        collection_header.addStretch()
        self.collection_info_label = QLabel("0件を保存中")
        self.collection_info_label.setStyleSheet("color: #4a5568;")
        collection_header.addWidget(self.collection_info_label)
        clear_btn = QPushButton("🧹 全削除")
        clear_btn.setMaximumWidth(120)
        clear_btn.clicked.connect(self.clear_graph_collection)
        collection_header.addWidget(clear_btn)
        layout.addLayout(collection_header)

        self.graph_collection_scroll = QScrollArea()
        self.graph_collection_scroll.setWidgetResizable(True)
        self.graph_collection_scroll.setMinimumHeight(240)
        self.graph_collection_container = QWidget()
        self.graph_collection_layout = QVBoxLayout(self.graph_collection_container)
        self.graph_collection_layout.setContentsMargins(5, 5, 5, 5)
        self.graph_collection_layout.setSpacing(12)
        self.graph_collection_layout.setAlignment(Qt.AlignTop)
        self.graph_collection_scroll.setWidget(self.graph_collection_container)
        layout.addWidget(self.graph_collection_scroll)
        
        return panel

    def create_graph_display_panel(self) -> QWidget:
        """後方互換性のために残すダミー関数"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(12)

        title = QLabel("プレビュー & グラフ表示")
        title.setStyleSheet("font-size: 16px; font-weight: 600; color: #0068B7;")
        layout.addWidget(title)

        splitter = QSplitter(Qt.Vertical)
        splitter.setChildrenCollapsible(False)

        frame_style = """
            QFrame {
                border: 2px solid #a0d2ff;
                border-radius: 10px;
                background-color: #ffffff;
                padding: 10px;
            }
        """

        table_container = QFrame()
        table_container.setStyleSheet(frame_style)
        table_layout = QVBoxLayout(table_container)
        table_layout.setContentsMargins(5, 5, 5, 5)
        table_layout.setSpacing(6)

        self.preview_info_label = QLabel("データを読み込むとプレビューを表示します")
        self.preview_info_label.setStyleSheet("font-weight: 600; color: #0068B7;")
        table_layout.addWidget(self.preview_info_label)

        self.preview_table = QTableWidget()
        self.preview_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.preview_table.setAlternatingRowColors(True)
        self.preview_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.preview_table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.preview_table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self.preview_table.verticalHeader().setVisible(False)
        self.preview_table.setStyleSheet(
            """
            QTableWidget {
                background-color: #ffffff;
                alternate-background-color: #f8fafc;
                border: none;
                color: #2d3748;
            }
            """
        )
        table_layout.addWidget(self.preview_table)

        canvas_container = QFrame()
        canvas_container.setStyleSheet(frame_style)
        canvas_layout = QVBoxLayout(canvas_container)
        canvas_layout.setContentsMargins(5, 5, 5, 5)

        self.canvas = MplCanvas(
            width=self.graph_settings["figsize_w"],
            height=self.graph_settings["figsize_h"],
            dpi=self.graph_settings["dpi"],
        )
        canvas_layout.addWidget(self.canvas, stretch=1)

        splitter.addWidget(table_container)
        splitter.addWidget(canvas_container)
        splitter.setSizes([280, 520])
        layout.addWidget(splitter, stretch=1)
        return panel
    
    def update_setting(self, key, value):
        # 設定を更新
        self.graph_settings[key] = value
    
    def on_xaxis_filter_mode_changed(self):
        """横軸日付フィルターモード変更時の処理"""
        mode = self.xaxis_date_filter_combo.currentData()
        self.update_setting("xaxis_date_filter", mode)
        
        # モードに応じてUIの表示/非表示を切り替え
        if hasattr(self, "xaxis_weekday_frame"):
            self.xaxis_weekday_frame.setVisible(mode == "specific_weekdays")
        if hasattr(self, "xaxis_every_n_frame"):
            self.xaxis_every_n_frame.setVisible(mode == "every_n_days")
    
    def on_xaxis_weekday_changed(self):
        """横軸表示曜日変更時の処理"""
        selected_weekdays = [
            idx for idx, check in self.xaxis_weekday_checks.items() if check.isChecked()
        ]
        self.update_setting("xaxis_weekdays", selected_weekdays)

    def populate_preview_table(self, df: pd.DataFrame | None):
        # プレビュー表を更新
        if not hasattr(self, "preview_table") or self.preview_table is None:
            return

        if df is None or df.empty:
            self.preview_table.clear()
            self.preview_table.setRowCount(0)
            self.preview_table.setColumnCount(0)
            self.preview_info_label.setText("表示できるデータがありません")
            return

        display_df = df.copy()
        max_rows = 200
        total_rows = len(display_df)
        if total_rows > max_rows:
            display_df = display_df.head(max_rows)

        column_names = [str(c) for c in display_df.columns]
        numeric_flags = {name: pd.api.types.is_numeric_dtype(display_df[name]) for name in display_df.columns}

        self.preview_table.setColumnCount(len(column_names))
        self.preview_table.setHorizontalHeaderLabels(column_names)
        self.preview_table.setRowCount(len(display_df))

        for row_idx, (_, row) in enumerate(display_df.iterrows()):
            for col_idx, name in enumerate(display_df.columns):
                value = row[name]
                if pd.isna(value):
                    text = ""
                elif numeric_flags[name]:
                    if isinstance(value, (int, np.integer)) or (isinstance(value, float) and value.is_integer()):
                        text = f"{int(round(float(value))):,}"
                    else:
                        text = f"{float(value):,.2f}"
                elif isinstance(value, (datetime, pd.Timestamp)):
                    text = pd.to_datetime(value).strftime("%Y-%m-%d %H:%M")
                else:
                    text = str(value)
                item = QTableWidgetItem(text)
                if numeric_flags[name]:
                    item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                else:
                    item.setTextAlignment(Qt.AlignLeft | Qt.AlignVCenter)
                self.preview_table.setItem(row_idx, col_idx, item)

        displayed_rows = len(display_df)
        if total_rows > max_rows:
            self.preview_info_label.setText(
                f"{total_rows:,}行のうち {displayed_rows:,}行を表示 (最大 {max_rows:,} 行)"
            )
        else:
            self.preview_info_label.setText(f"{total_rows:,}行を表示中")
    
    def update_column_checkboxes(self):
        # 列チェックボックスを更新
        prev_key = getattr(self, "current_dataset_key", None)
        if prev_key and isinstance(prev_key, tuple):
            # 現在の選択状態をキャッシュ
            self.column_selection_cache[prev_key] = [
                col for col, cb in self.column_checkboxes.items() if cb.isChecked()
            ]
            # 最後に使用した選択パターンも保存（エリア間で共有）
            self._last_column_selection = [
                col for col, cb in self.column_checkboxes.items() if cb.isChecked()
            ]

        # 既存のチェックボックスをクリア
        for checkbox in self.column_checkboxes.values():
            checkbox.deleteLater()
        self.column_checkboxes.clear()
        self.selected_columns = []
        self.current_dataframe = None
        self.current_time_column = None
        self.current_dataset_key = None
        
        code = self.area_combo.currentData()
        ym = self.ym_combo.currentData()
        if not ym or not code:
            self.populate_preview_table(None)
            return
        
        path = DATA_DIR / f"eria_jukyu_{ym}_{code}.csv"
        if not path.exists():
            self.populate_preview_table(None)
            return
        
        try:
            df, tcol = read_csv(path)
            self.current_dataframe = df
            self.current_time_column = tcol
            self.current_dataset_key = (code, ym)
            
            # 選択状態の復元: 1. 特定データセットのキャッシュ, 2. 最後の選択パターン, 3. 全選択
            saved_selection = self.column_selection_cache.get(self.current_dataset_key)
            last_selection = getattr(self, "_last_column_selection", None)
            
            # 数値カラムを取得
            for c in df.columns:
                    col_str = str(c).lower()
                    if not any(keyword in col_str for keyword in ['date', 'time', '時刻', '日時', '日付']):
                        checkbox = QCheckBox(str(c))
                        if saved_selection is not None:
                            # このデータセット固有のキャッシュがある場合
                            checkbox.setChecked(str(c) in saved_selection)
                        elif last_selection is not None and str(c) in last_selection:
                            # 最後に使用した選択パターンを適用（同じ列名のみ）
                            checkbox.setChecked(True)
                        elif last_selection is not None:
                            # 最後に選択パターンがあり、この列が含まれていない場合
                            checkbox.setChecked(False)
                        else:
                            # 初回は全選択
                            checkbox.setChecked(True)
                        checkbox.toggled.connect(self.on_column_selection_changed)
                        self.column_checkbox_layout.addWidget(checkbox)
                        self.column_checkboxes[str(c)] = checkbox

            # チェック状態に基づき選択リストを更新
            self.on_column_selection_changed()
            self.on_date_change()
        except Exception as e:
            print(f"Error loading columns: {e}")
            self.populate_preview_table(None)
    
    def on_column_selection_changed(self):
        # 列選択が変更された時
        self.selected_columns = [col for col, cb in self.column_checkboxes.items() if cb.isChecked()]
        if getattr(self, "current_dataset_key", None):
            self.column_selection_cache[self.current_dataset_key] = list(self.selected_columns)
    
    def select_all_columns(self):
        """列チェックボックスを全選択する"""
        for checkbox in self.column_checkboxes.values():
            checkbox.setChecked(True)
    
    def deselect_all_columns(self):
        # 列チェックボックスを全て解除
        for checkbox in self.column_checkboxes.values():
            checkbox.setChecked(False)
    
    def save_graph(self):
        # グラフを画像として保存
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "グラフを保存",
            "",
            "PNG (*.png);;PDF (*.pdf);;SVG (*.svg)"
        )

        if filename:
            self.canvas.fig.savefig(filename, dpi=self.graph_settings['dpi'], bbox_inches='tight')
            QtWidgets.QMessageBox.information(self, "成功", f"グラフを保存しました:\n{filename}")

    def add_graph_to_collection(self):
        # 現在の設定でグラフカードを追加
        if self.graph_collection_layout is None:
            return

        code = self.area_combo.currentData()
        ym = self.ym_combo.currentData()
        if not code or not ym:
            QtWidgets.QMessageBox.warning(self, "警告", "エリアと年月を選択してください")
            return

        if self.current_dataframe is None:
            QtWidgets.QMessageBox.warning(self, "警告", "グラフに利用できるデータがありません")
            return

        df_filtered, selected_date = self.get_current_filtered_dataframe()
        if df_filtered is None:
            QtWidgets.QMessageBox.warning(self, "警告", "選択された条件で表示できるデータがありません")
            return

        if not self.selected_columns:
            self.selected_columns = [col for col, cb in self.column_checkboxes.items() if cb.isChecked()]

        if not self.selected_columns:
            QtWidgets.QMessageBox.warning(self, "警告", "表示する発電方式を選択してください")
            return

        date_label = self.date_combo.currentText() or ""
        if not selected_date or selected_date == "all":
            date_label = "全期間"

        settings_snapshot = self.graph_settings.copy()
        snapshot = GraphSnapshot(
            area_code=code,
            area_name=AREA_INFO[code].name,
            year_month=ym,
            date_label=date_label,
            columns=list(self.selected_columns),
            settings=settings_snapshot,
        )

        card = GraphCard(self, snapshot, self.remove_graph_card)
        self.graph_collection_widgets.append(card)
        self.graph_collection_layout.addWidget(card)

        title_text = self.draw_graph_on_canvas(
            card.canvas,
            df_filtered,
            self.current_time_column,
            snapshot.columns,
            code,
            ym,
            selected_date,
            settings_snapshot,
        )
        card.set_plot_title(title_text)
        self.update_graph_collection_info()

    def remove_graph_card(self, card: GraphCard):
        # コレクションから指定カードを削除
        if not isinstance(card, GraphCard):
            return
        if card in self.graph_collection_widgets:
            self.graph_collection_widgets.remove(card)
        if self.graph_collection_layout is not None:
            self.graph_collection_layout.removeWidget(card)
        card.setParent(None)
        card.deleteLater()
        self.update_graph_collection_info()

    def clear_graph_collection(self):
        # 保存済みグラフを全て削除
        for card in list(self.graph_collection_widgets):
            self.remove_graph_card(card)

    def overlay_selected_graphs(self):
        """選択されたグラフを重ね合わせて表示"""
        selected_cards = [card for card in self.graph_collection_widgets if card.overlay_checkbox.isChecked()]
        
        if len(selected_cards) < 2:
            QtWidgets.QMessageBox.warning(
                self, 
                "警告", 
                "重ね合わせるには2つ以上のグラフを選択してください。"
            )
            return
        
        # 選択されたモードを確認
        use_time_mode = self.overlay_time_radio.isChecked()
        
        # メインキャンバスをクリアして重ね合わせ表示
        self.canvas.fig.clear()
        ax = self.canvas.fig.add_subplot(111)
        ax.set_facecolor("#f8fafc")
        
        # カラーパレット
        colors = ['#0068B7', '#10b981', '#ef4444', '#f59e0b', '#8b5cf6', '#ec4899', '#06b6d4', '#84cc16']
        
        if use_time_mode:
            # モード1: 時系列順（時刻で結合）
            # 各カードから元のDataFrameデータを取得して時刻で結合
            for idx, card in enumerate(selected_cards):
                color = colors[idx % len(colors)]
                
                # スナップショットから元データを読み込み
                code = card.snapshot.area_code
                ym = card.snapshot.year_month
                path = DATA_DIR / f"eria_jukyu_{ym}_{code}.csv"
                
                if not path.exists():
                    continue
                
                try:
                    # CSVファイルを読み込み
                    df = pd.read_csv(path, encoding="shift_jis", skiprows=1)
                    
                    # 時刻列を探す
                    time_col = None
                    for col in df.columns:
                        col_lower = str(col).lower()
                        if any(kw in col_lower for kw in ["date", "time", "日時", "時刻", "年月日"]):
                            time_col = col
                            break
                    
                    if time_col:
                        # 時刻列をDatetime型に変換
                        df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
                        df = df.dropna(subset=[time_col])
                        
                        # 選択された列をプロット
                        for column in card.snapshot.columns:
                            if column in df.columns:
                                valid = df[column].notna()
                                label = f"{card.snapshot.area_name} - {column}"
                                ax.plot(
                                    df[time_col][valid],
                                    df[column][valid],
                                    color=color,
                                    linewidth=2,
                                    alpha=0.8,
                                    label=label,
                                    marker='o',
                                    markersize=3
                                )
                        
                        # X軸の日時フォーマットを調整
                        self.canvas.fig.autofmt_xdate(rotation=45)
                except Exception as e:
                    print(f"データ読み込みエラー ({card.snapshot.area_name}): {e}")
                    continue
            
            ax.set_xlabel('時刻（時系列順）', fontsize=12, fontweight='bold')
            ax.set_title('選択グラフの重ね合わせ表示（時系列順）', fontsize=16, fontweight='bold', color='#0068B7', pad=20)
        
        else:
            # モード2: X軸共有（値の単純比較）
            # インデックスベースでプロット（日数として表示）
            for idx, card in enumerate(selected_cards):
                color = colors[idx % len(colors)]
                
                # カードのキャンバスから線データを取得して再描画
                for line in card.canvas.ax.get_lines():
                    xdata = line.get_xdata()
                    ydata = line.get_ydata()
                    label = line.get_label()
                    
                    # データポイントを日数として表示（1日目から開始）
                    x_days = [i + 1 for i in range(len(ydata))]
                    
                    # ラベルにエリア情報を追加
                    if not label.startswith('_'):
                        enhanced_label = f"{card.snapshot.area_name} - {label}"
                    else:
                        enhanced_label = f"{card.snapshot.area_name}"
                    
                    ax.plot(x_days, ydata, color=color, linewidth=2, alpha=0.8, label=enhanced_label, marker='o', markersize=3)
            
            ax.set_xlabel('日数', fontsize=12, fontweight='bold')
            ax.set_title('選択グラフの重ね合わせ表示（X軸共有）', fontsize=16, fontweight='bold', color='#0068B7', pad=20)
            
            # X軸を整数で表示（日数）
            ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        
        # 共通の軸ラベルとスタイル
        ax.set_ylabel('電力量 (MWh)', fontsize=12, fontweight='bold')
        
        # 凡例
        ax.legend(loc='best', fontsize=9, framealpha=0.9, ncol=1)
        
        # グリッド
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # 軸の装飾
        ax.tick_params(colors="#2d3748", labelsize=10)
        ax.spines["bottom"].set_color("#a0d2ff")
        ax.spines["top"].set_color("#a0d2ff")
        ax.spines["left"].set_color("#a0d2ff")
        ax.spines["right"].set_color("#a0d2ff")
        
        # Y軸のフォーマット
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _pos: f"{int(x):,}"))
        
        self.canvas.fig.tight_layout()
        self.canvas.draw()
        
        # グラフ表示タブに切り替え
        if hasattr(self, 'tabs'):
            # メインページのタブを取得
            main_page_widget = self.tabs.widget(0)
            if main_page_widget and hasattr(main_page_widget, 'findChild'):
                right_tabs = main_page_widget.findChild(QTabWidget)
                if right_tabs:
                    # グラフ表示タブ（インデックス1）に切り替え
                    right_tabs.setCurrentIndex(1)
        
        mode_text = "時系列順" if use_time_mode else "X軸共有"
        QtWidgets.QMessageBox.information(
            self,
            "完了",
            f"{len(selected_cards)}個のグラフを重ね合わせて表示しました。\nモード: {mode_text}"
        )

    def update_graph_collection_info(self):
        # カード数のテキストを更新
        if self.collection_info_label is None:
            return
        count = len(self.graph_collection_widgets)
        self.collection_info_label.setText(f"{count}件を保存中")

    def apply_modern_palette(self):
        pal = QtGui.QPalette()
        # 東京都市大学の統計カラーテーマ
        pal.setColor(QtGui.QPalette.Window, QtGui.QColor("#f0f4f8"))
        pal.setColor(QtGui.QPalette.WindowText, QtGui.QColor("#1a202c"))
        pal.setColor(QtGui.QPalette.Base, QtGui.QColor("#ffffff"))
        pal.setColor(QtGui.QPalette.AlternateBase, QtGui.QColor("#e6f2ff"))
        pal.setColor(QtGui.QPalette.Text, QtGui.QColor("#2d3748"))
        pal.setColor(QtGui.QPalette.Button, QtGui.QColor("#ffffff"))
        pal.setColor(QtGui.QPalette.ButtonText, QtGui.QColor("#0068B7"))
        pal.setColor(QtGui.QPalette.Highlight, QtGui.QColor("#0068B7"))
        pal.setColor(QtGui.QPalette.HighlightedText, QtGui.QColor("#ffffff"))
        self.setPalette(pal)
        
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f0f4f8;
            }
            QTableWidget {
                gridline-color: #cbd5e0;
                color: #2d3748;
                font-size: 13px;
                selection-background-color: #0068B7;
                background-color: #ffffff;
            }
            QHeaderView::section {
                background-color: #e6f2ff;
                color: #0068B7;
                border: 1px solid #a0d2ff;
                padding: 8px;
                font-weight: 600;
                font-size: 12px;
            }
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                          stop:0 #0068B7, stop:1 #005291);
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 8px;
                font-weight: 600;
                font-size: 13px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                          stop:0 #0080e0, stop:1 #0068B7);
            }
            QPushButton:pressed {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                          stop:0 #005291, stop:1 #004170);
            }
            QLabel {
                color: #2d3748;
            }
            QComboBox {
                background-color: #ffffff;
                border: 2px solid #a0d2ff;
                border-radius: 8px;
                padding: 6px 10px;
                color: #2d3748;
                font-size: 13px;
                min-height: 24px;
            }
            QComboBox:hover {
                border: 2px solid #0068B7;
            }
            QComboBox::drop-down {
                border: none;
                padding-right: 8px;
            }
            QComboBox::down-arrow {
                image: none;
                border-left: 5px solid transparent;
                border-right: 5px solid transparent;
                border-top: 5px solid #0068B7;
                margin-right: 5px;
            }
            QComboBox QAbstractItemView {
                background-color: #ffffff;
                color: #2d3748;
                selection-background-color: #0068B7;
                border: 2px solid #a0d2ff;
                border-radius: 6px;
            }
            QSplitter::handle {
                background-color: #cbd5e0;
                width: 2px;
            }
            QSplitter::handle:hover {
                background-color: #0068B7;
            }
            QGroupBox {
                border: 2px solid #a0d2ff;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 10px;
                font-weight: 600;
                color: #0068B7;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
            QCheckBox {
                color: #2d3748;
                font-size: 13px;
            }
            QCheckBox::indicator {
                width: 18px;
                height: 18px;
                border: 2px solid #a0d2ff;
                border-radius: 4px;
            }
            QCheckBox::indicator:checked {
                background-color: #0068B7;
                border-color: #0068B7;
            }
            QSpinBox, QDoubleSpinBox, QLineEdit {
                background-color: #ffffff;
                border: 2px solid #a0d2ff;
                border-radius: 6px;
                padding: 4px 8px;
                color: #2d3748;
                font-size: 13px;
            }
            QSpinBox:focus, QDoubleSpinBox:focus, QLineEdit:focus {
                border-color: #0068B7;
            }
        """)

    def apply_theme(self):
        """アプリケーション全体のテーマを適用"""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f5f7fa;
            }
            QLabel {
                color: #333333;
                font-family: "Meiryo", "Yu Gothic", sans-serif;
            }
            QGroupBox {
                font-weight: bold;
                border: 1px solid #d1d5db;
                border-radius: 6px;
                margin-top: 12px;
                background-color: white;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                left: 10px;
                padding: 0 5px;
                color: #0068B7;
            }
            QPushButton {
                background-color: #ffffff;
                border: 1px solid #d1d5db;
                border-radius: 4px;
                padding: 6px 12px;
                color: #333333;
            }
            QPushButton:hover {
                background-color: #f3f4f6;
                border-color: #9ca3af;
            }
            QPushButton:pressed {
                background-color: #e5e7eb;
            }
            QComboBox, QSpinBox, QDoubleSpinBox, QDateEdit {
                border: 1px solid #d1d5db;
                border-radius: 4px;
                padding: 4px;
                background-color: white;
                selection-background-color: #0068B7;
            }
            QTableWidget {
                border: 1px solid #d1d5db;
                gridline-color: #e5e7eb;
                background-color: white;
                alternate-background-color: #f9fafb;
            }
            QHeaderView::section {
                background-color: #f3f4f6;
                padding: 4px;
                border: 1px solid #e5e7eb;
                font-weight: bold;
                color: #4b5563;
            }
        """)

    def open_official(self) -> None:
        # Open the official data portal for the currently selected area.

        code = self.area_combo.currentData()
        if code:
            webbrowser.open(AREA_INFO[code].url)

    def open_folder(self):
        path = str(DATA_DIR)
        if os.name == "nt":
            os.startfile(path)
        elif sys.platform == "darwin":
            os.system(f'open "{path}"')
        else:
            os.system(f'xdg-open "{path}"')

    def on_area_change(self):
        # メインページ: エリア変更時にヒートマップを更新
        self.files = scan_files()
        self.avail, self.years, self.months = build_availability(self.files)
        self.refresh_area_year_months()
        self.refresh_availability_table()

    def update_month_combo(self):
        """選択された年に対応する月のリストを更新"""
        if not hasattr(self, "year_combo") or not hasattr(self, "month_combo"):
            return
        
        code = self.area_combo.currentData() if hasattr(self, "area_combo") else None
        year = self.year_combo.currentData()
        
        if not code or not year:
            return
        
        # 選択された年に対応する月を取得
        year_months = self.area_year_months.get(code, [])
        months = sorted([ym[4:6] for ym in year_months if ym.startswith(year)])
        
        self.month_combo.blockSignals(True)
        self.month_combo.clear()
        for month in months:
            self.month_combo.addItem(f"{int(month)}月", month)
        self.month_combo.blockSignals(False)
        
        if self.month_combo.count() > 0:
            self.month_combo.setCurrentIndex(0)
            self.on_year_month_change()

    def on_year_month_change(self):
        """年または月が変更された時の処理"""
        if not hasattr(self, "year_combo") or not hasattr(self, "month_combo"):
            return
        
        year = self.year_combo.currentData()
        month = self.month_combo.currentData()
        
        if not year or not month:
            return
        
        # 年月を結合してYYYYMM形式にする
        ym = f"{year}{month}"
        
        # ym_comboも更新（後方互換性）
        if hasattr(self, "ym_combo"):
            for i in range(self.ym_combo.count()):
                if self.ym_combo.itemData(i) == ym:
                    self.ym_combo.blockSignals(True)
                    self.ym_combo.setCurrentIndex(i)
                    self.ym_combo.blockSignals(False)
                    break
        
        # 既存のon_ym_change処理を実行
        self.on_ym_change()

    def on_ym_change(self):
        # 年月が変更された時に日付リストを更新
        code = self.area_combo.currentData()
        ym = self.ym_combo.currentData()
        if not ym or not code:
            return
        
        path = DATA_DIR / f"eria_jukyu_{ym}_{code}.csv"
        if not path.exists():
            self.current_dataframe = None
            self.current_time_column = None
            self.current_dataset_key = None
            self.populate_preview_table(None)
            return
        
        try:
            df, tcol = read_csv(path)
            self.date_combo.blockSignals(True)
            self.date_combo.clear()
            self.date_combo.addItem("全期間", "all")
            
            if tcol and tcol in df.columns:
                dates = pd.to_datetime(df[tcol]).dt.date.unique()
                dates = sorted(dates)
                for date in dates:
                    date_str = date.strftime("%Y年%m月d日")
                    self.date_combo.addItem(date_str, str(date))
            self.date_combo.blockSignals(False)

            # 列チェックボックスを更新
            self.update_column_checkboxes()
        except Exception:
            self.date_combo.blockSignals(False)
            self.populate_preview_table(None)

    def on_date_change(self):
        # 日付選択が変わった際にプレビューを更新
        if self.current_dataframe is None:
            self.populate_preview_table(None)
            return

        df_filtered, _selected_date = self.get_current_filtered_dataframe()
        if df_filtered is None:
            self.populate_preview_table(None)
        else:
            self.populate_preview_table(df_filtered)

    def get_current_filtered_dataframe(self) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
        # 現在の選択条件でフィルタ済みデータを取得
        if self.current_dataframe is None:
            return None, None

        df_filtered = self.current_dataframe.copy()
        tcol = self.current_time_column
        selected_date = self.date_combo.currentData()

        if selected_date and selected_date != "all" and tcol and tcol in df_filtered.columns:
            try:
                df_filtered["_date"] = pd.to_datetime(df_filtered[tcol]).dt.date
                filter_date = pd.to_datetime(selected_date).date()
                df_filtered = df_filtered[df_filtered["_date"] == filter_date].copy()
                df_filtered = df_filtered.drop(columns=["_date"])
            except Exception:
                pass

        if len(df_filtered) == 0:
            return None, selected_date

        return df_filtered, selected_date

    def draw_graph_on_canvas(
        self,
        canvas: MplCanvas,
        df_filtered: pd.DataFrame,
        time_column: Optional[str],
        columns: Sequence[str],
        area_code: AreaCode,
        year_month: YearMonth,
        selected_date: Optional[str],
        settings: Dict[str, Any],
    ) -> str:
        """選択された項目でグラフを描画してタイトルを返す"""
        canvas.ax.clear()
        canvas.ax.set_facecolor("#f8fafc")
        canvas.ax.tick_params(colors="#2d3748", labelsize=settings["font_size"])
        for spine in canvas.ax.spines.values():
            spine.set_color("#a0d2ff")

        colors = ['#0068B7', '#00A0E9', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899', '#14b8a6']

        # 曜日フォーマット設定
        weekday_names_jp = ["月", "火", "水", "木", "金", "土", "日"]
        weekday_names_jp_full = ["月曜日", "火曜日", "水曜日", "木曜日", "金曜日", "土曜日", "日曜日"]
        weekday_names_en = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        weekday_names_en_full = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        
        def get_weekday_str(dt, fmt):
            """曜日文字列を取得"""
            if pd.isna(dt):
                return ""
            wd = dt.weekday()
            if fmt == "short":
                return f"({weekday_names_jp[wd]})"
            elif fmt == "full":
                return weekday_names_jp_full[wd]
            elif fmt == "en_short":
                return weekday_names_en[wd]
            elif fmt == "en_full":
                return weekday_names_en_full[wd]
            return f"({weekday_names_jp[wd]})"

        if time_column and time_column in df_filtered.columns:
            x_data = df_filtered[time_column]
            
            # 時間データをdatetimeに変換
            x_datetime = pd.to_datetime(x_data, errors="coerce")
            
            for idx, column in enumerate(columns):
                if column in df_filtered.columns:
                    valid = df_filtered[column].notna()
                    canvas.ax.plot(
                        x_datetime[valid],
                        df_filtered[column][valid],
                        label=str(column),
                        color=colors[idx % len(colors)],
                        linewidth=settings["linewidth"],
                        alpha=0.9,
                    )
            
            # X軸の目盛り位置を制御（日付フィルター機能）
            import matplotlib.dates as mdates
            xaxis_filter = settings.get("xaxis_date_filter", "all")
            
            if xaxis_filter != "all":
                # フィルタリングされた日付のみ表示
                unique_dates = x_datetime.dt.normalize().unique()
                unique_dates = pd.Series(unique_dates).sort_values()
                
                filtered_dates = []
                
                if xaxis_filter == "specific_weekdays":
                    # 特定の曜日のみ表示
                    selected_weekdays = settings.get("xaxis_weekdays", [0, 1, 2, 3, 4, 5, 6])
                    for dt in unique_dates:
                        if pd.notna(dt) and dt.weekday() in selected_weekdays:
                            filtered_dates.append(dt)
                
                elif xaxis_filter == "every_n_days":
                    # N日おきに表示
                    n = settings.get("xaxis_every_n_days", 1)
                    for i, dt in enumerate(unique_dates):
                        if pd.notna(dt) and i % n == 0:
                            filtered_dates.append(dt)
                
                # フィルタリングされた日付を目盛り位置に設定
                if filtered_dates:
                    canvas.ax.set_xticks(filtered_dates)
            
            # X軸ラベルのカスタマイズ（曜日表示対応）
            if settings.get("show_weekday", False):
                weekday_fmt = settings.get("weekday_format", "short")
                midnight_fmt = settings.get("midnight_label_format", "next_day")
                
                def custom_date_formatter(x, pos):
                    try:
                        dt = mdates.num2date(x)
                        # 24:00表記の処理
                        if midnight_fmt == "same_day" and dt.hour == 0 and dt.minute == 0:
                            # 前日の24:00として表示
                            prev_day = dt - pd.Timedelta(days=1)
                            weekday_str = get_weekday_str(prev_day, weekday_fmt)
                            return f"{prev_day.day}日24:00\n{weekday_str}"
                        else:
                            weekday_str = get_weekday_str(dt, weekday_fmt)
                            return f"{dt.month}/{dt.day} {dt.hour:02d}:{dt.minute:02d}\n{weekday_str}"
                    except:
                        return ""
                
                canvas.ax.xaxis.set_major_formatter(ticker.FuncFormatter(custom_date_formatter))
            
            # 週の境界線を表示
            if settings.get("show_week_boundaries", False):
                week_start = settings.get("week_boundary_day", "monday")
                target_weekday = 0 if week_start == "monday" else 6  # 0=月曜, 6=日曜
                
                # 時系列データから週の境界を探す
                unique_dates = x_datetime.dt.normalize().unique()
                unique_dates = pd.Series(unique_dates).sort_values()
                
                for dt in unique_dates:
                    if pd.notna(dt):
                        if dt.weekday() == target_weekday:
                            # 週の始まりに縦線を描画
                            canvas.ax.axvline(
                                x=dt, 
                                color='#6366f1', 
                                linestyle='--', 
                                linewidth=1.5, 
                                alpha=0.7,
                                label='_nolegend_'
                            )
            
            # X軸ラベル
            if settings.get("show_xlabel", True):
                canvas.ax.set_xlabel(
                    settings["xlabel"],
                    color="#2d3748",
                    fontsize=settings["label_size"],
                    fontweight="bold",
                )
            else:
                canvas.ax.set_xlabel("")
            
            # X軸ラベルの回転角度を設定
            rotation_angle = settings.get("xaxis_tick_rotation", 45)
            canvas.fig.autofmt_xdate(rotation=rotation_angle)
        else:
            x_axis = range(len(df_filtered))
            for idx, column in enumerate(columns):
                if column in df_filtered.columns:
                    canvas.ax.plot(
                        x_axis,
                        df_filtered[column].fillna(0),
                        label=str(column),
                        color=colors[idx % len(colors)],
                        linewidth=settings["linewidth"],
                        alpha=0.9,
                    )
            # X軸ラベル
            if settings.get("show_xlabel", True):
                canvas.ax.set_xlabel(
                    settings["xlabel"],
                    color="#2d3748",
                    fontsize=settings["label_size"],
                    fontweight="bold",
                )
            else:
                canvas.ax.set_xlabel("")

        # Y軸ラベル
        if settings.get("show_ylabel", True):
            canvas.ax.set_ylabel(
                settings["ylabel"],
                color="#2d3748",
                fontsize=settings["label_size"],
                fontweight="bold",
            )
        else:
            canvas.ax.set_ylabel("")

        # 凡例
        if settings.get("show_legend", True) and settings["legend"]:
            canvas.ax.legend(
                loc=settings["legend_loc"],
                facecolor="#ffffff",
                edgecolor="#a0d2ff",
                labelcolor="#2d3748",
                fontsize=settings["font_size"],
                framealpha=0.95,
            )

        # タイトル
        # 曜日名のリスト（タイトル用）
        weekday_names_jp_title = ["月", "火", "水", "木", "金", "土", "日"]
        
        if settings["title"]:
            title_text = settings["title"]
        else:
            title_text = f"{AREA_INFO[area_code].name}エリア - {year_month[:4]}年{year_month[4:6]}月"
            if selected_date and selected_date != "all":
                date_obj = pd.to_datetime(selected_date)
                # 曜日表示設定がオンの場合は曜日も追加
                if settings.get("show_weekday", False):
                    wd = date_obj.weekday()
                    title_text += f" ({date_obj.strftime('%m月%d日')} {weekday_names_jp_title[wd]}曜日)"
                else:
                    title_text += f" ({date_obj.strftime('%m月%d日')})"

        if settings.get("show_title", True):
            canvas.ax.set_title(
                title_text,
                color="#0068B7",
                fontsize=settings["title_size"],
                fontweight="bold",
                pad=15,
            )
        else:
            canvas.ax.set_title("")

        if settings["grid"]:
            canvas.ax.grid(True, alpha=0.3, color="#cbd5e0", linestyle="--", linewidth=0.8)

        # 軸の目盛りを細かく表示するオプション
        if settings.get("detailed_ticks", False):
            # Y軸の目盛りを細かく
            canvas.ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=15, integer=True))
            canvas.ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
            
            # X軸の目盛りを細かく（時系列データの場合）
            if time_column and time_column in df_filtered.columns:
                # 時系列データの場合は自動で細かく調整
                canvas.ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=20))
                canvas.ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(2))
            else:
                canvas.ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=15, integer=True))
                canvas.ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(2))
            
            # マイナー目盛りのグリッドも表示
            canvas.ax.grid(which='minor', alpha=0.15, color="#cbd5e0", linestyle=":", linewidth=0.5)
        
        canvas.ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _pos: f"{int(x):,}"))
        canvas.update_size(settings["figsize_w"], settings["figsize_h"], settings["dpi"])
        canvas.fig.tight_layout(pad=2.0)
        canvas.draw()
        return title_text

    def render_view(self):
        code = self.area_combo.currentData()
        ym = self.ym_combo.currentData()
        if not ym:
            QtWidgets.QMessageBox.information(self, "案内", "年月を選択してください")
            return

        path = DATA_DIR / f"eria_jukyu_{ym}_{code}.csv"
        if not path.exists():
            QtWidgets.QMessageBox.warning(self, "警告", "選択年月のCSVが見つかりません")
            return

        dataset_key = (code, ym)
        if self.current_dataset_key != dataset_key or self.current_dataframe is None:
            try:
                self.update_column_checkboxes()
            except Exception as exc:
                QtWidgets.QMessageBox.critical(self, "エラー", f"CSVファイルの読み込みに失敗しました:\n{exc}")
                return

        if self.current_dataframe is None:
            QtWidgets.QMessageBox.warning(self, "警告", "データを読み込めませんでした")
            return

        df_filtered, selected_date = self.get_current_filtered_dataframe()
        if df_filtered is None:
            self.populate_preview_table(None)
            QtWidgets.QMessageBox.warning(self, "警告", "選択された条件で表示できるデータがありません")
            return

        self.populate_preview_table(df_filtered)

        if not self.selected_columns:
            self.selected_columns = [col for col, cb in self.column_checkboxes.items() if cb.isChecked()]

        if not self.selected_columns:
            QtWidgets.QMessageBox.warning(self, "警告", "表示する発電方式を選択してください")
            return

        self.draw_graph_on_canvas(
            self.canvas,
            df_filtered,
            self.current_time_column,
            self.selected_columns,
            code,
            ym,
            selected_date,
            self.graph_settings.copy(),
        )

    def create_comparison_page(self) -> QWidget:
        """発電種別比較タブを作成"""
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)

        # ヘッダー
        header = QHBoxLayout()
        title = QLabel("⚡ 発電種別比較")
        title.setStyleSheet("font-size: 20px; font-weight: 600; color: #0068B7;")
        header.addWidget(title)
        header.addStretch()
        layout.addLayout(header)

        desc = QLabel("エリア間または月間の発電種別構成（原子力、火力、再エネ等）を比較します。")
        layout.addWidget(desc)

        # コントロールエリア
        controls_frame = QFrame()
        controls_frame.setStyleSheet(".QFrame { background-color: #f8fafc; border-radius: 8px; border: 1px solid #e2e8f0; }")
        controls_layout = QHBoxLayout(controls_frame)
        controls_layout.setContentsMargins(15, 15, 15, 15)
        controls_layout.setSpacing(20)

        # モード選択
        mode_layout = QVBoxLayout()
        mode_layout.addWidget(QLabel("比較モード:"))
        self.comp_mode_combo = QComboBox()
        self.comp_mode_combo.addItems(["エリア間比較 (月指定)", "月間推移比較 (エリア指定)"])
        self.comp_mode_combo.currentIndexChanged.connect(self.update_comp_controls)
        mode_layout.addWidget(self.comp_mode_combo)
        controls_layout.addLayout(mode_layout)

        # 動的コントロール (スタック)
        self.comp_controls_stack = QtWidgets.QStackedWidget()
        
        # Page 1: エリア間比較 (月を指定)
        page1 = QWidget()
        layout1 = QVBoxLayout(page1) # Changed to VBox to accommodate area selection
        layout1.setContentsMargins(0, 0, 0, 0)
        
        # Month Selection Row
        row1 = QHBoxLayout()
        self.comp_ym_combo = QComboBox()
        self.comp_ym_combo.setMinimumWidth(150)
        row1.addWidget(QLabel("対象年月:"))
        row1.addWidget(self.comp_ym_combo)
        row1.addStretch()
        layout1.addLayout(row1)
        
        # Area Selection Group
        area_group = QGroupBox("比較対象エリア選択")
        area_layout = QGridLayout(area_group)
        self.comp_area_checks = {}
        for i, (code, meta) in enumerate(AREA_INFO.items()):
            chk = QCheckBox(meta.name)
            chk.setChecked(True)
            # Store code as property or use a map
            chk.setProperty("area_code", code)
            self.comp_area_checks[code] = chk
            area_layout.addWidget(chk, i // 5, i % 5)
        layout1.addWidget(area_group)
        
        self.comp_controls_stack.addWidget(page1)
        
        # Page 2: 月間推移比較 (エリアと年を指定)
        page2 = QWidget()
        layout2 = QHBoxLayout(page2)
        layout2.setContentsMargins(0, 0, 0, 0)
        self.comp_area_combo = QComboBox()
        self.comp_area_combo.setMinimumWidth(150)
        self.comp_year_combo = QComboBox()
        self.comp_year_combo.setMinimumWidth(100)
        layout2.addWidget(QLabel("エリア:"))
        layout2.addWidget(self.comp_area_combo)
        layout2.addWidget(QLabel("対象年:"))
        layout2.addWidget(self.comp_year_combo)
        layout2.addStretch()
        self.comp_controls_stack.addWidget(page2)
        
        controls_layout.addWidget(self.comp_controls_stack)

        # 表示単位
        unit_layout = QVBoxLayout()
        unit_layout.addWidget(QLabel("表示単位:"))
        self.comp_unit_group = QtWidgets.QButtonGroup(page)
        self.comp_unit_kw = QtWidgets.QRadioButton("電力量 (MWh)")
        self.comp_unit_ratio = QtWidgets.QRadioButton("割合 (%)")
        self.comp_unit_kw.setChecked(True)
        self.comp_unit_group.addButton(self.comp_unit_kw)
        self.comp_unit_group.addButton(self.comp_unit_ratio)
        unit_layout.addWidget(self.comp_unit_kw)
        unit_layout.addWidget(self.comp_unit_ratio)
        controls_layout.addLayout(unit_layout)

        # グラフタイプ
        graph_type_layout = QVBoxLayout()
        graph_type_layout.addWidget(QLabel("グラフタイプ:"))
        self.comp_graph_type_combo = QComboBox()
        self.comp_graph_type_combo.addItems(["積み上げ棒グラフ", "円グラフ (合計)"])
        graph_type_layout.addWidget(self.comp_graph_type_combo)
        controls_layout.addLayout(graph_type_layout)

        # 発電種別選択
        cat_group = QGroupBox("発電種別選択")
        cat_layout = QGridLayout(cat_group)
        self.comp_cat_checks = {}
        for i, cat in enumerate(GENERATION_CATEGORIES.keys()):
            chk = QCheckBox(cat)
            chk.setChecked(True)
            self.comp_cat_checks[cat] = chk
            cat_layout.addWidget(chk, i // 4, i % 4)
        controls_layout.addWidget(cat_group)
        
        # 需給実績選択
        demand_group = QGroupBox("需給実績選択")
        demand_layout = QGridLayout(demand_group)
        self.comp_demand_checks = {}
        for i, cat in enumerate(DEMAND_SUPPLY_CATEGORIES.keys()):
            chk = QCheckBox(cat)
            chk.setChecked(False)  # デフォルトはオフ
            self.comp_demand_checks[cat] = chk
            demand_layout.addWidget(chk, i // 4, i % 4)
        controls_layout.addWidget(demand_group)

        # 実行ボタン
        btn = QPushButton("集計・グラフ表示")
        btn.setMinimumHeight(40)
        btn.setStyleSheet("""
            QPushButton {
                background-color: #0068B7;
                color: white;
                font-weight: bold;
                border-radius: 5px;
                padding: 0 20px;
            }
            QPushButton:hover {
                background-color: #005090;
            }
        """)
        btn.clicked.connect(self.run_comparison_analysis)
        controls_layout.addWidget(btn)
        
        # 統計データコピーボタン
        copy_btn = QPushButton("📋 数値をコピー")
        copy_btn.setMinimumHeight(40)
        copy_btn.setStyleSheet("""
            QPushButton {
                background-color: #10b981;
                color: white;
                font-weight: bold;
                border-radius: 5px;
                padding: 0 15px;
            }
            QPushButton:hover {
                background-color: #059669;
            }
        """)
        copy_btn.clicked.connect(self.copy_comparison_data)
        controls_layout.addWidget(copy_btn)
        
        layout.addWidget(controls_frame)

        # 結果表示エリア（タブ形式）
        self.comp_tabs = QTabWidget()
        self.comp_tabs.setStyleSheet("""
            QTabWidget::pane {
                border: 2px solid #a0d2ff;
                border-radius: 8px;
                background-color: white;
            }
            QTabBar::tab {
                background: #f0f0f0;
                padding: 8px 16px;
                margin-right: 2px;
            }
            QTabBar::tab:selected {
                background: #0068B7;
                color: white;
            }
        """)
        
        # グラフタブ
        graph_widget = QWidget()
        graph_layout = QVBoxLayout(graph_widget)
        self.comp_canvas = MplCanvas(width=12, height=8)
        graph_layout.addWidget(self.comp_canvas)
        self.comp_tabs.addTab(graph_widget, "📊 グラフ")
        
        # 数値データタブ
        table_widget = QWidget()
        table_layout = QVBoxLayout(table_widget)
        self.comp_data_table = QTableWidget()
        self.comp_data_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.comp_data_table.setAlternatingRowColors(True)
        self.comp_data_table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self.comp_data_table.setStyleSheet("""
            QTableWidget {
                background-color: #ffffff;
                alternate-background-color: #f8fafc;
                border: 1px solid #e2e8f0;
                font-size: 12px;
            }
            QHeaderView::section {
                background-color: #e6f2ff;
                color: #0068B7;
                padding: 8px;
                border: 1px solid #a0d2ff;
                font-weight: 600;
            }
        """)
        table_layout.addWidget(self.comp_data_table)
        self.comp_tabs.addTab(table_widget, "📋 数値データ")
        
        layout.addWidget(self.comp_tabs, stretch=1)
        
        return page

    def update_comp_controls(self, index: int) -> None:
        self.comp_controls_stack.setCurrentIndex(index)

    def populate_comp_controls(self) -> None:
        """比較タブのコントロールを初期化"""
        # エリア間比較用の年月コンボ
        self.comp_ym_combo.clear()
        # 全ての利用可能な年月を収集
        all_yms = sorted(list(set([f[0] for f in self.files])), reverse=True)
        for ym in all_yms:
            self.comp_ym_combo.addItem(f"{ym[:4]}年{ym[4:6]}月", ym)
            
        # 月間推移比較用のエリアコンボ
        self.comp_area_combo.clear()
        for code, meta in AREA_INFO.items():
            self.comp_area_combo.addItem(f"({code}) {meta.name}", code)
            
        # 月間推移比較用の年コンボ
        self.comp_year_combo.clear()
        years = sorted(list(set([int(f[0][:4]) for f in self.files])), reverse=True)
        if not years:
            years = [datetime.now().year]
        for y in years:
            self.comp_year_combo.addItem(f"{y}年", y)

    def run_comparison_analysis(self) -> None:
        """発電種別比較を実行"""
        mode = self.comp_mode_combo.currentIndex()
        is_ratio = self.comp_unit_ratio.isChecked()
        graph_type = self.comp_graph_type_combo.currentIndex() # 0: Bar, 1: Pie
        
        # 選択されたカテゴリを取得（発電種別＋需給実績）
        selected_gen_cats = [cat for cat, chk in self.comp_cat_checks.items() if chk.isChecked()]
        selected_demand_cats = [cat for cat, chk in self.comp_demand_checks.items() if chk.isChecked()]
        selected_cats = selected_gen_cats + selected_demand_cats
        
        if not selected_cats:
            QtWidgets.QMessageBox.warning(self, "警告", "少なくとも1つの項目を選択してください。")
            return

        data_map = {} # Key: Label (Area or Month), Value: Dict[Category, Amount]
        totals_map = {} # Key: Label, Value: Total Generation (for ratio calculation)
        labels = []
        
        try:
            if mode == 0: # エリア間比較
                ym = self.comp_ym_combo.currentData()
                if not ym:
                    return
                
                # Get selected areas
                selected_areas = [code for code, chk in self.comp_area_checks.items() if chk.isChecked()]
                if not selected_areas:
                    QtWidgets.QMessageBox.warning(self, "警告", "少なくとも1つのエリアを選択してください。")
                    return

                target_files = [f for f in self.files if f[0] == ym and f[1] in selected_areas]
                if not target_files:
                    QtWidgets.QMessageBox.warning(self, "警告", "該当するデータがありません。")
                    return
                
                # エリアコード順にソート
                target_files.sort(key=lambda x: x[1])
                
                for _, code, path in target_files:
                    df, _ = read_csv(path)
                    sums = self._aggregate_categories(df, selected_gen_cats, selected_demand_cats)
                    
                    # Calculate Grand Total (sum of all categories, not just selected)
                    grand_total = sum(sums.values())
                    
                    # 選択されたカテゴリのみ抽出
                    filtered_sums = {k: v for k, v in sums.items() if k in selected_cats}
                    area_name = AREA_INFO[code].name
                    data_map[area_name] = filtered_sums
                    totals_map[area_name] = grand_total
                    labels.append(area_name)
                    
            else: # 月間推移比較
                code = self.comp_area_combo.currentData()
                year = self.comp_year_combo.currentData()
                if not code or not year:
                    return
                
                # 1月〜12月
                for m in range(1, 13):
                    ym = f"{year}{m:02d}"
                    # 該当ファイルを探す
                    path = None
                    for f_ym, f_code, f_path in self.files:
                        if f_ym == ym and f_code == code:
                            path = f_path
                            break
                    
                    month_label = f"{m}月"
                    labels.append(month_label)
                    
                    if path:
                        df, _ = read_csv(path)
                        sums = self._aggregate_categories(df, selected_gen_cats, selected_demand_cats)
                        grand_total = sum(sums.values())
                        filtered_sums = {k: v for k, v in sums.items() if k in selected_cats}
                        data_map[month_label] = filtered_sums
                        totals_map[month_label] = grand_total
                    else:
                        data_map[month_label] = {cat: 0.0 for cat in selected_cats}
                        totals_map[month_label] = 0.0

            # グラフ描画
            self._plot_comparison(data_map, labels, is_ratio, mode, graph_type, selected_cats, totals_map)
            
            # 数値データテーブルに表示
            self._update_comparison_table(data_map, labels, selected_cats, is_ratio, totals_map)
            
            # データを保存（コピー用）
            self._comp_data_map = data_map
            self._comp_labels = labels
            self._comp_selected_cats = selected_cats
            self._comp_is_ratio = is_ratio
            self._comp_totals_map = totals_map
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            QtWidgets.QMessageBox.critical(self, "エラー", f"集計中にエラーが発生しました:\n{str(e)}")

    def _aggregate_categories(self, df: pd.DataFrame, 
                               selected_gen_cats: List[str] = None,
                               selected_demand_cats: List[str] = None) -> Dict[str, float]:
        """DataFrameからカテゴリごとの合計値(MWh)を計算"""
        if selected_gen_cats is None:
            selected_gen_cats = list(GENERATION_CATEGORIES.keys())
        if selected_demand_cats is None:
            selected_demand_cats = []
        
        # 発電種別の集計
        sums = {cat: 0.0 for cat in selected_gen_cats}
        
        for col in df.columns:
            # 数値列のみ
            if not pd.api.types.is_numeric_dtype(df[col]):
                continue
                
            # カテゴリ判定（発電種別）
            for cat in selected_gen_cats:
                keywords = GENERATION_CATEGORIES.get(cat, [])
                if col in keywords:
                    val = df[col].sum()
                    sums[cat] += val / 1000.0  # MWh
                    break
        
        # 需給実績の集計
        for cat in selected_demand_cats:
            keywords = DEMAND_SUPPLY_CATEGORIES.get(cat, [])
            sums[cat] = 0.0
            for col in df.columns:
                if not pd.api.types.is_numeric_dtype(df[col]):
                    continue
                if col in keywords:
                    val = df[col].sum()
                    sums[cat] += val / 1000.0  # MWh
                    break
                
        return sums

    def _plot_comparison(self, data_map: Dict[str, Dict[str, float]], labels: List[str], is_ratio: bool, mode: int, graph_type: int, selected_cats: List[str], totals_map: Dict[str, float] = None) -> None:
        self.comp_canvas.ax.clear()
        
        # 色定義（発電種別＋需給実績）
        colors = {
            # 発電種別
            "原子力": "#8b5cf6", # 紫
            "火力": "#ef4444",   # 赤
            "水力": "#3b82f6",   # 青
            "地熱": "#d97706",   # 茶/オレンジ
            "バイオマス": "#10b981", # 緑
            "太陽光": "#f59e0b", # 黄色
            "風力": "#06b6d4",   # シアン
            "揚水": "#6366f1",   # インディゴ
            # 需給実績
            "エリア需要": "#dc2626",   # 赤
            "エリア供給": "#059669",   # 緑
            "連系線": "#7c3aed",       # 紫
            "揚水動力": "#0891b2",     # シアン
        }
        
        if graph_type == 1: # 円グラフ (合計)
            # 全データの合計を計算
            total_sums = {cat: 0.0 for cat in selected_cats}
            for label in labels:
                for cat in selected_cats:
                    total_sums[cat] += data_map[label].get(cat, 0.0)
            
            # 値が0のカテゴリを除外
            pie_labels = []
            pie_values = []
            pie_colors = []
            
            for cat in selected_cats:
                val = total_sums[cat]
                if val > 0:
                    pie_labels.append(cat)
                    pie_values.append(val)
                    pie_colors.append(colors.get(cat, "#999999"))
            
            if not pie_values:
                self.comp_canvas.ax.text(0.5, 0.5, "データなし", ha='center', va='center')
            else:
                wedges, texts, autotexts = self.comp_canvas.ax.pie(
                    pie_values, labels=pie_labels, autopct='%1.1f%%',
                    startangle=90, counterclock=False, colors=pie_colors,
                    wedgeprops={'width': 0.4, 'edgecolor': 'w'}
                )
                self.comp_canvas.ax.set_title("選択範囲の発電種別構成 (合計)", fontweight='bold')

        else: # 積み上げ棒グラフ
            x = range(len(labels))
            bottom = np.zeros(len(labels))
            
            # Calculate denominators for ratio
            denominators = np.zeros(len(labels))
            if is_ratio:
                if totals_map:
                    # Use provided totals (Grand Total of company)
                    for i, label in enumerate(labels):
                        denominators[i] = totals_map.get(label, 1.0)
                else:
                    # Fallback: Sum of selected categories
                    for c in selected_cats:
                        t_vals = []
                        for l in labels:
                            t_vals.append(data_map[l].get(c, 0.0))
                        denominators += np.array(t_vals)
                
                # Avoid division by zero
                denominators[denominators == 0] = 1.0

            for cat in selected_cats:
                values = []
                for label in labels:
                    val = data_map[label].get(cat, 0.0)
                    values.append(val)
                
                values = np.array(values)
                
                if is_ratio:
                    values = (values / denominators) * 100
                
                self.comp_canvas.ax.bar(x, values, bottom=bottom, label=cat, color=colors.get(cat, "#999999"), alpha=0.9, width=0.6)
                bottom += values
                
            self.comp_canvas.ax.set_xticks(x)
            self.comp_canvas.ax.set_xticklabels(labels, rotation=45 if mode == 0 else 0)
            
            ylabel = "構成比 (%)" if is_ratio else "発電電力量 (MWh)"
            self.comp_canvas.ax.set_ylabel(ylabel)
            
            title_suffix = " (割合)" if is_ratio else " (電力量)"
            if mode == 0:
                ym_text = self.comp_ym_combo.currentText()
                self.comp_canvas.ax.set_title(f"エリア別発電種別構成 - {ym_text}{title_suffix}", fontweight='bold')
            else:
                area_text = self.comp_area_combo.currentText()
                year_text = self.comp_year_combo.currentText()
                self.comp_canvas.ax.set_title(f"月別発電種別構成推移 - {area_text} {year_text}{title_suffix}", fontweight='bold')
                
            # 凡例をグラフの外に配置
            self.comp_canvas.ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
            self.comp_canvas.ax.grid(True, axis='y', alpha=0.3)
        
        self.comp_canvas.fig.tight_layout()
        self.comp_canvas.draw()

    def _update_comparison_table(self, data_map: Dict[str, Dict[str, float]], labels: List[str], selected_cats: List[str], is_ratio: bool, totals_map: Dict[str, float] = None) -> None:
        """比較結果をテーブルに表示"""
        # テーブル設定
        self.comp_data_table.clear()
        self.comp_data_table.setRowCount(len(labels) + 1)  # +1 for totals
        self.comp_data_table.setColumnCount(len(selected_cats) + 2)  # +2 for label and total
        
        # ヘッダー
        headers = [""] + selected_cats + ["合計"]
        self.comp_data_table.setHorizontalHeaderLabels(headers)
        
        # データ入力
        for i, label in enumerate(labels):
            self.comp_data_table.setItem(i, 0, QTableWidgetItem(label))
            row_total = 0.0
            denominator = totals_map.get(label, 1.0) if totals_map and is_ratio else 1.0
            if denominator == 0:
                denominator = 1.0
                
            for j, cat in enumerate(selected_cats):
                val = data_map[label].get(cat, 0.0)
                if is_ratio:
                    display_val = (val / denominator) * 100
                    item = QTableWidgetItem(f"{display_val:.2f}%")
                else:
                    item = QTableWidgetItem(f"{val:,.2f}")
                row_total += val
                self.comp_data_table.setItem(i, j + 1, item)
            
            # 行合計
            if is_ratio:
                row_ratio = (row_total / denominator) * 100 if denominator > 0 else 0
                self.comp_data_table.setItem(i, len(selected_cats) + 1, QTableWidgetItem(f"{row_ratio:.2f}%"))
            else:
                self.comp_data_table.setItem(i, len(selected_cats) + 1, QTableWidgetItem(f"{row_total:,.2f}"))
        
        # 合計行
        last_row = len(labels)
        self.comp_data_table.setItem(last_row, 0, QTableWidgetItem("【合計】"))
        grand_total = 0.0
        for j, cat in enumerate(selected_cats):
            cat_total = sum(data_map[l].get(cat, 0.0) for l in labels)
            grand_total += cat_total
            if is_ratio:
                total_denom = sum(totals_map.get(l, 0) for l in labels) if totals_map else grand_total
                if total_denom == 0:
                    total_denom = 1.0
                pct = (cat_total / total_denom) * 100
                self.comp_data_table.setItem(last_row, j + 1, QTableWidgetItem(f"{pct:.2f}%"))
            else:
                self.comp_data_table.setItem(last_row, j + 1, QTableWidgetItem(f"{cat_total:,.2f}"))
        
        # 総合計
        if is_ratio:
            self.comp_data_table.setItem(last_row, len(selected_cats) + 1, QTableWidgetItem("100.00%"))
        else:
            self.comp_data_table.setItem(last_row, len(selected_cats) + 1, QTableWidgetItem(f"{grand_total:,.2f}"))

    def copy_comparison_data(self) -> None:
        """比較データをクリップボードにコピー"""
        if not hasattr(self, "_comp_data_map") or self._comp_data_map is None:
            QtWidgets.QMessageBox.warning(self, "警告", "先に分析を実行してください。")
            return
        
        data_map = self._comp_data_map
        labels = self._comp_labels
        selected_cats = self._comp_selected_cats
        is_ratio = self._comp_is_ratio
        totals_map = self._comp_totals_map
        
        # タブ区切りでコピー
        lines = []
        
        # ヘッダー
        headers = [""] + selected_cats + ["合計"]
        lines.append("\t".join(headers))
        
        # データ行
        for label in labels:
            row = [label]
            row_total = 0.0
            denominator = totals_map.get(label, 1.0) if totals_map and is_ratio else 1.0
            if denominator == 0:
                denominator = 1.0
            
            for cat in selected_cats:
                val = data_map[label].get(cat, 0.0)
                if is_ratio:
                    display_val = (val / denominator) * 100
                    row.append(f"{display_val:.2f}")
                else:
                    row.append(f"{val:.2f}")
                row_total += val
            
            if is_ratio:
                row_ratio = (row_total / denominator) * 100 if denominator > 0 else 0
                row.append(f"{row_ratio:.2f}")
            else:
                row.append(f"{row_total:.2f}")
            
            lines.append("\t".join(row))
        
        # 合計行
        total_row = ["合計"]
        grand_total = 0.0
        for cat in selected_cats:
            cat_total = sum(data_map[l].get(cat, 0.0) for l in labels)
            grand_total += cat_total
            if is_ratio:
                total_denom = sum(totals_map.get(l, 0) for l in labels) if totals_map else grand_total
                if total_denom == 0:
                    total_denom = 1.0
                pct = (cat_total / total_denom) * 100
                total_row.append(f"{pct:.2f}")
            else:
                total_row.append(f"{cat_total:.2f}")
        
        if is_ratio:
            total_row.append("100.00")
        else:
            total_row.append(f"{grand_total:.2f}")
        
        lines.append("\t".join(total_row))
        
        text = "\n".join(lines)
        
        clipboard = QApplication.clipboard()
        clipboard.setText(text)
        
        QtWidgets.QMessageBox.information(
            self, "コピー完了",
            f"比較データをクリップボードにコピーしました。\n\n"
            f"ExcelやPowerPointに貼り付けできます。\n"
            f"({len(labels)}行 × {len(selected_cats)}列)"
        )

    def transfer_to_analysis_tab(self) -> None:
        """メインタブで選択したデータを統計分析タブに移行"""
        code = self.area_combo.currentData()
        year = self.year_combo.currentData() if hasattr(self, "year_combo") else None
        month = self.month_combo.currentData() if hasattr(self, "month_combo") else None
        
        if not code or not year or not month:
            QtWidgets.QMessageBox.warning(self, "警告", "エリアと年月を選択してください。")
            return
        
        # 型変換（文字列の場合に対応）
        if isinstance(month, str):
            month_int = int(month)
        else:
            month_int = month
        if isinstance(year, str):
            year_int = int(year)
        else:
            year_int = year
        
        ym = f"{year_int}{month_int:02d}"
        
        # AI分析タブのコンボボックスを同期
        # エリアを設定
        for i in range(self.ai_area_combo.count()):
            if self.ai_area_combo.itemData(i) == code:
                self.ai_area_combo.setCurrentIndex(i)
                break
        
        # 年を設定
        for i in range(self.ai_year_combo.count()):
            if self.ai_year_combo.itemData(i) == year:
                self.ai_year_combo.setCurrentIndex(i)
                break
        
        # 月を設定
        for i in range(self.ai_month_combo.count()):
            if self.ai_month_combo.itemData(i) == month:
                self.ai_month_combo.setCurrentIndex(i)
                break
        
        # 統計集計の列コンボを更新
        self._update_stats_columns()
        
        # 統計分析タブに切り替え
        self.tabs.setCurrentIndex(1)
        
        # 統計集計タブを選択
        self.ai_tabs.setCurrentWidget(self.ai_tabs.widget(self.ai_tabs.count() - 1))
        
        QtWidgets.QMessageBox.information(
            self, "データ移行",
            f"メインタブのデータを統計分析タブに移行しました。\n\n"
            f"エリア: {AREA_INFO[code].name}\n"
            f"年月: {year}年{month}月\n\n"
            f"「統計集計」タブで集計を実行できます。"
        )

    def _update_stats_columns(self) -> None:
        """統計集計の列コンボボックスを更新"""
        if not hasattr(self, "stats_column_combo"):
            return
        
        self.stats_column_combo.clear()
        
        # AI用データフレームをロード
        code = self.ai_area_combo.currentData()
        ym = self.ai_ym_combo.currentData()
        
        if not code or not ym:
            return
        
        path = DATA_DIR / f"eria_jukyu_{ym}_{code}.csv"
        if not path.exists():
            return
        
        try:
            df, _ = read_csv(path)
            # 数値列のみをリストに追加
            for col in df.columns:
                if pd.api.types.is_numeric_dtype(df[col]):
                    self.stats_column_combo.addItem(col)
        except Exception:
            pass
    
    def _update_dw_columns(self) -> None:
        """需要・天候分析の列コンボボックスを更新（積算列も追加）"""
        if not hasattr(self, "dw_column_combo"):
            return
        
        self.dw_column_combo.clear()
        
        code = self.ai_area_combo.currentData()
        year = self.ai_year_combo.currentData()
        month = self.ai_month_combo.currentData()
        
        if not code or not year or not month:
            return
        
        # 型変換
        if isinstance(month, str):
            month = int(month)
        if isinstance(year, str):
            year = int(year)
        
        ym = f"{year}{month:02d}"
        path = DATA_DIR / f"eria_jukyu_{ym}_{code}.csv"
        if not path.exists():
            return
        
        try:
            df, time_col = read_csv(path)
            
            # 時間列をパース
            if time_col and time_col in df.columns:
                df["_datetime"] = pd.to_datetime(df[time_col], errors="coerce")
            else:
                df["_datetime"] = pd.to_datetime(df.index)
            
            df["_date"] = df["_datetime"].dt.date
            
            # 数値列を取得
            numeric_cols = []
            for col in df.columns:
                if col.startswith("_"):
                    continue
                if pd.api.types.is_numeric_dtype(df[col]):
                    numeric_cols.append(col)
            
            # 通常の数値列を追加
            for col in numeric_cols:
                self.dw_column_combo.addItem(col)
            
            # 積算列を追加（日別合計として計算可能な列）
            self.dw_column_combo.insertSeparator(len(numeric_cols))
            for col in numeric_cols:
                # 30分値/1時間値として積算可能な列を追加
                self.dw_column_combo.addItem(f"[積算] {col}")
        
        except Exception:
            pass
            return
        
        try:
            df, _ = read_csv(path)
            # 数値列のみをリストに追加
            for col in df.columns:
                if pd.api.types.is_numeric_dtype(df[col]):
                    self.stats_column_combo.addItem(col)
        except Exception:
            pass

    def run_stats_summary(self) -> None:
        """統計集計を実行"""
        code = self.ai_area_combo.currentData()
        ym = self.ai_ym_combo.currentData()
        col_name = self.stats_column_combo.currentText()
        period = self.stats_period_combo.currentText()
        
        if not code or not ym or not col_name:
            QtWidgets.QMessageBox.warning(self, "警告", "エリア、年月、対象列を選択してください。")
            return
        
        path = DATA_DIR / f"eria_jukyu_{ym}_{code}.csv"
        if not path.exists():
            QtWidgets.QMessageBox.warning(self, "警告", f"ファイルが見つかりません: {path.name}")
            return
        
        try:
            df, time_col = read_csv(path)
            
            if col_name not in df.columns:
                QtWidgets.QMessageBox.warning(self, "警告", f"列 '{col_name}' が見つかりません。")
                return
            
            # 数値に変換
            df[col_name] = pd.to_numeric(df[col_name], errors="coerce")
            
            # 時間列をパース
            if time_col and time_col in df.columns:
                df["_datetime"] = pd.to_datetime(df[time_col], errors="coerce")
            else:
                df["_datetime"] = pd.to_datetime(df.index)
            
            # 集計キーを作成
            if period == "時間別":
                df["_key"] = df["_datetime"].dt.strftime("%Y-%m-%d %H:00")
                key_format = "時間"
            elif period == "日別":
                df["_key"] = df["_datetime"].dt.strftime("%Y-%m-%d")
                key_format = "日"
            elif period == "週別":
                df["_key"] = df["_datetime"].dt.strftime("%Y-W%V")
                key_format = "週"
            elif period == "月別":
                df["_key"] = df["_datetime"].dt.strftime("%Y-%m")
                key_format = "月"
            else:
                df["_key"] = "全期間"
                key_format = ""
            
            # 集計
            grouped = df.groupby("_key")[col_name].agg(["sum", "mean", "min", "max", "count"])
            grouped = grouped.reset_index()
            grouped.columns = [key_format or "期間", "合計", "平均", "最小", "最大", "データ数"]
            
            # テーブルに表示
            self.stats_table.clear()
            self.stats_table.setRowCount(len(grouped))
            self.stats_table.setColumnCount(len(grouped.columns))
            self.stats_table.setHorizontalHeaderLabels(grouped.columns.tolist())
            
            for i, row in grouped.iterrows():
                for j, val in enumerate(row):
                    if isinstance(val, (int, float)) and j > 0:
                        item = QTableWidgetItem(f"{val:,.2f}")
                    else:
                        item = QTableWidgetItem(str(val))
                    self.stats_table.setItem(i, j, item)
            
            # 全体統計を計算
            total_sum = df[col_name].sum()
            total_mean = df[col_name].mean()
            total_min = df[col_name].min()
            total_max = df[col_name].max()
            total_count = df[col_name].count()
            
            # サマリー（単位付き）
            # 列名から単位を推測（データは瞬時値[MW]が基本）
            # 瞬時値の場合、合計は意味がないため電力量に換算する
            unit = " MW"  # デフォルトはMW（瞬時値）
            is_instantaneous = True  # 瞬時値フラグ
            
            if "kWh" in col_name or "電力量" in col_name:
                unit = " kWh"
                is_instantaneous = False
            elif "MWh" in col_name:
                unit = " MWh"
                is_instantaneous = False
            elif "%" in col_name or "率" in col_name:
                unit = " %"
                is_instantaneous = False
            elif "kW" in col_name:
                unit = " kW"
            
            # データ間隔を推定（30分 = 0.5時間）
            interval_hours = 0.5
            if len(df) >= 2 and "_datetime" in df.columns:
                time_diff = df["_datetime"].diff().median()
                if pd.notna(time_diff):
                    interval_hours = time_diff.total_seconds() / 3600
            
            if is_instantaneous:
                # 瞬時値の場合、電力量を概算
                energy_sum = total_sum * interval_hours  # MW × 時間 = MWh
                energy_unit = " MWh" if "MW" in unit or unit == " MW" else " kWh"
                
                self.stats_summary_label.setText(
                    f"📊 {AREA_INFO[code].name} - {ym[:4]}年{ym[4:6]}月 - {col_name}\n\n"
                    f"【全体統計】※データは瞬時値（{unit.strip()}）\n"
                    f"  • 積算電力量（概算）: {energy_sum:,.2f}{energy_unit}\n"
                    f"    （{interval_hours}時間間隔 × {total_count:,}件で計算）\n"
                    f"  • 平均: {total_mean:,.2f}{unit}\n"
                    f"  • 最小: {total_min:,.2f}{unit}\n"
                    f"  • 最大: {total_max:,.2f}{unit}\n"
                    f"  • データ数: {total_count:,} 件\n\n"
                    f"集計期間: {period} ({len(grouped)}区間)"
                )
            else:
                self.stats_summary_label.setText(
                    f"📊 {AREA_INFO[code].name} - {ym[:4]}年{ym[4:6]}月 - {col_name}\n\n"
                    f"【全体統計】\n"
                    f"  • 合計: {total_sum:,.2f}{unit}\n"
                    f"  • 平均: {total_mean:,.2f}{unit}\n"
                    f"  • 最小: {total_min:,.2f}{unit}\n"
                    f"  • 最大: {total_max:,.2f}{unit}\n"
                    f"  • データ数: {total_count:,} 件\n\n"
                    f"集計期間: {period} ({len(grouped)}区間)"
                )
            
            # 保存用にデータを保持
            self._stats_result = grouped
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            QtWidgets.QMessageBox.critical(self, "エラー", f"集計中にエラーが発生しました:\n{str(e)}")

    def copy_stats_to_clipboard(self) -> None:
        """統計集計結果をクリップボードにコピー"""
        if not hasattr(self, "_stats_result") or self._stats_result is None:
            QtWidgets.QMessageBox.warning(self, "警告", "先に集計を実行してください。")
            return
        
        # タブ区切りでコピー（Excel/PowerPoint貼り付け用）
        text_lines = []
        
        # ヘッダー
        text_lines.append("\t".join(self._stats_result.columns.tolist()))
        
        # データ
        for _, row in self._stats_result.iterrows():
            line = []
            for val in row:
                if isinstance(val, (int, float)):
                    line.append(f"{val:.2f}")
                else:
                    line.append(str(val))
            text_lines.append("\t".join(line))
        
        text = "\n".join(text_lines)
        
        clipboard = QApplication.clipboard()
        clipboard.setText(text)
        
        QtWidgets.QMessageBox.information(
            self, "コピー完了",
            f"統計集計結果をクリップボードにコピーしました。\n\n"
            f"ExcelやPowerPointに貼り付けできます。\n"
            f"({len(self._stats_result)}行)"
        )

    def run_demand_weather_analysis(self) -> None:
        """需要・天候分析を実行（日別/時別対応）"""
        code = self.ai_area_combo.currentData()
        year = self.ai_year_combo.currentData()
        month = self.ai_month_combo.currentData()
        col_name_raw = self.dw_column_combo.currentText()
        top_n = self.dw_top_n_spin.value()
        agg_method = self.dw_agg_combo.currentText()  # 平均, 合計, 最大
        time_unit = self.dw_time_unit_combo.currentText()  # 日別, 時別
        
        if not code or not year or not month or not col_name_raw:
            QtWidgets.QMessageBox.warning(self, "警告", "エリア、年月、分析対象を選択してください。")
            return
        
        # 積算列かどうか判定
        is_cumulative = col_name_raw.startswith("[積算] ")
        if is_cumulative:
            col_name = col_name_raw.replace("[積算] ", "")
            # 積算の場合は日別合計を強制
            effective_agg = "合計"
            display_col_name = f"{col_name}（積算）"
        else:
            col_name = col_name_raw
            effective_agg = agg_method
            display_col_name = col_name
        
        # 型変換
        if isinstance(month, str):
            month = int(month)
        if isinstance(year, str):
            year = int(year)
        
        ym = f"{year}{month:02d}"
        path = DATA_DIR / f"eria_jukyu_{ym}_{code}.csv"
        
        if not path.exists():
            QtWidgets.QMessageBox.warning(self, "警告", f"ファイルが見つかりません: {path.name}")
            return
        
        try:
            # 需給データ読み込み
            df, time_col = read_csv(path)
            
            if col_name not in df.columns:
                QtWidgets.QMessageBox.warning(self, "警告", f"列 '{col_name}' が見つかりません。")
                return
            
            df[col_name] = pd.to_numeric(df[col_name], errors="coerce")
            
            # 時間列をパース
            if time_col and time_col in df.columns:
                df["_datetime"] = pd.to_datetime(df[time_col], errors="coerce")
            else:
                df["_datetime"] = pd.to_datetime(df.index)
            
            df["_date"] = df["_datetime"].dt.date
            df["_hour"] = df["_datetime"].dt.hour
            df["_weekday"] = df["_datetime"].dt.dayofweek  # 0=月曜, 6=日曜
            
            # 曜日名
            weekday_names = ["月", "火", "水", "木", "金", "土", "日"]
            
            # 天候データ読み込み
            weather_file = find_weather_file(code, ym)
            weather_df = None
            has_weather = False
            has_weather_name = False
            
            if weather_file:
                try:
                    weather_df = read_weather_csv(weather_file)
                    weather_df["_date"] = weather_df["datetime"].dt.date
                    weather_df["_hour"] = weather_df["datetime"].dt.hour
                    has_weather = True
                    has_weather_name = "weather" in weather_df.columns
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    print(f"天候データ読み込みエラー: {e}")
            
            if time_unit == "日別":
                # 日別分析（積算の場合は合計を使用）
                result_df = self._analyze_daily(df, col_name, effective_agg, weekday_names, 
                                                  weather_df, has_weather, has_weather_name)
                unit_label = "日"
            else:
                # 時別分析
                result_df = self._analyze_hourly(df, col_name, effective_agg, weekday_names,
                                                   weather_df, has_weather, has_weather_name)
                unit_label = "時間帯"
            
            # 上位Nと下位Nを取得
            sorted_df = result_df.sort_values("value", ascending=False)
            top_records = sorted_df.head(top_n).copy()
            bottom_records = sorted_df.tail(top_n).copy()
            
            # テーブル更新
            self._update_demand_weather_table(self.dw_high_table, top_records, has_weather, time_unit)
            self._update_demand_weather_table(self.dw_low_table, bottom_records, has_weather, time_unit)
            
            # サマリー作成
            summary_lines = []
            summary_lines.append(f"📊 分析対象: {AREA_INFO[code].name} - {year}年{month}月 - {display_col_name}")
            if is_cumulative:
                summary_lines.append(f"分析単位: {time_unit}  ※積算値（日別合計）で分析")
            else:
                summary_lines.append(f"分析単位: {time_unit}  集計方法: {agg_method}")
            summary_lines.append("")
            
            # 曜日分布
            top_weekday_counts = top_records["weekday_name"].value_counts()
            bottom_weekday_counts = bottom_records["weekday_name"].value_counts()
            
            summary_lines.append(f"【高い{unit_label}の曜日分布】")
            for wd in weekday_names:
                count = top_weekday_counts.get(wd, 0)
                summary_lines.append(f"  {wd}曜: {'■' * count} ({count})")
            
            summary_lines.append("")
            summary_lines.append(f"【低い{unit_label}の曜日分布】")
            for wd in weekday_names:
                count = bottom_weekday_counts.get(wd, 0)
                summary_lines.append(f"  {wd}曜: {'■' * count} ({count})")
            
            # 時別の場合は時間帯分布も追加
            if time_unit == "時別" and "hour" in top_records.columns:
                summary_lines.append("")
                summary_lines.append("【高い時間帯の分布】")
                top_hour_counts = top_records["hour"].value_counts().sort_index()
                for hour, count in top_hour_counts.items():
                    summary_lines.append(f"  {hour:02d}時: {'■' * count} ({count})")
                
                summary_lines.append("")
                summary_lines.append("【低い時間帯の分布】")
                bottom_hour_counts = bottom_records["hour"].value_counts().sort_index()
                for hour, count in bottom_hour_counts.items():
                    summary_lines.append(f"  {hour:02d}時: {'■' * count} ({count})")
            
            if has_weather:
                summary_lines.append("")
                summary_lines.append("【天候傾向】")
                
                # 高い/低いの天候比較
                high_temp = top_records["気温(℃)"].mean() if "気温(℃)" in top_records.columns else float("nan")
                low_temp = bottom_records["気温(℃)"].mean() if "気温(℃)" in bottom_records.columns else float("nan")
                high_precip = top_records["降水量(mm)"].mean() if "降水量(mm)" in top_records.columns else float("nan")
                low_precip = bottom_records["降水量(mm)"].mean() if "降水量(mm)" in bottom_records.columns else float("nan")
                
                if pd.notna(high_temp) and pd.notna(low_temp):
                    summary_lines.append(f"  高い{unit_label}: 平均気温 {high_temp:.1f}℃, 平均降水量 {high_precip:.1f}mm")
                    summary_lines.append(f"  低い{unit_label}: 平均気温 {low_temp:.1f}℃, 平均降水量 {low_precip:.1f}mm")
                    
                    temp_diff = high_temp - low_temp
                    if abs(temp_diff) > 2:
                        if temp_diff > 0:
                            summary_lines.append(f"  → 高い{unit_label}は気温が高い傾向（{temp_diff:+.1f}℃）")
                        else:
                            summary_lines.append(f"  → 高い{unit_label}は気温が低い傾向（{temp_diff:+.1f}℃）")
                
                # 天気の分布（天気名がある場合）
                if "天気" in top_records.columns:
                    summary_lines.append("")
                    summary_lines.append("【天気分布】")
                    
                    # NaNを除外してカウント
                    high_weather = top_records["天気"].dropna()
                    high_weather = high_weather[high_weather != ""]
                    low_weather = bottom_records["天気"].dropna()
                    low_weather = low_weather[low_weather != ""]
                    
                    if len(high_weather) > 0:
                        high_weather_counts = high_weather.value_counts()
                        summary_lines.append(f"  高い{unit_label}の天気:")
                        for weather, count in high_weather_counts.head(5).items():
                            pct = count / len(high_weather) * 100
                            summary_lines.append(f"    {weather}: {count} ({pct:.0f}%)")
                    
                    if len(low_weather) > 0:
                        low_weather_counts = low_weather.value_counts()
                        summary_lines.append(f"  低い{unit_label}の天気:")
                        for weather, count in low_weather_counts.head(5).items():
                            pct = count / len(low_weather) * 100
                            summary_lines.append(f"    {weather}: {count} ({pct:.0f}%)")
            else:
                summary_lines.append("")
                summary_lines.append("※天候データが見つかりませんでした")
            
            self.dw_summary_label.setText("\n".join(summary_lines))
            
            # 結果保存
            self._dw_top_records = top_records
            self._dw_bottom_records = bottom_records
            self._dw_has_weather = has_weather
            self._dw_time_unit = time_unit
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            QtWidgets.QMessageBox.critical(self, "エラー", f"分析中にエラーが発生しました:\n{str(e)}")
    
    def _analyze_daily(self, df: pd.DataFrame, col_name: str, agg_method: str, 
                       weekday_names: list, weather_df: pd.DataFrame, 
                       has_weather: bool, has_weather_name: bool) -> pd.DataFrame:
        """日別分析"""
        # 日別集計
        if agg_method == "平均":
            daily = df.groupby("_date")[col_name].mean().reset_index()
        elif agg_method == "合計":
            daily = df.groupby("_date")[col_name].sum().reset_index()
        else:  # 最大
            daily = df.groupby("_date")[col_name].max().reset_index()
        
        daily.columns = ["date", "value"]
        daily["date"] = pd.to_datetime(daily["date"])
        daily["weekday"] = daily["date"].dt.dayofweek
        daily["weekday_name"] = daily["weekday"].apply(lambda x: weekday_names[x])
        
        if has_weather and weather_df is not None:
            # 天候を日別に集計
            agg_dict = {
                "temperature": "mean",
                "precipitation": "sum",
                "sunlight": "sum",
                "wind_speed": "mean"
            }
            
            weather_daily = weather_df.groupby("_date").agg(agg_dict).reset_index()
            weather_daily.columns = ["date", "気温(℃)", "降水量(mm)", "日照時間(h)", "風速(m/s)"]
            weather_daily["date"] = pd.to_datetime(weather_daily["date"])
            
            # 天気の最頻値を取得
            if has_weather_name:
                def get_mode_weather(group):
                    weather_vals = group["weather"].dropna()
                    weather_vals = weather_vals[weather_vals != ""]
                    if len(weather_vals) > 0:
                        return weather_vals.mode().iloc[0] if len(weather_vals.mode()) > 0 else ""
                    return ""
                
                weather_mode = weather_df.groupby("_date").apply(
                    get_mode_weather, include_groups=False
                ).reset_index()
                weather_mode.columns = ["date", "天気"]
                weather_mode["date"] = pd.to_datetime(weather_mode["date"])
                weather_daily = weather_daily.merge(weather_mode, on="date", how="left")
            
            daily = daily.merge(weather_daily, on="date", how="left")
        
        return daily
    
    def _analyze_hourly(self, df: pd.DataFrame, col_name: str, agg_method: str,
                        weekday_names: list, weather_df: pd.DataFrame,
                        has_weather: bool, has_weather_name: bool) -> pd.DataFrame:
        """時別分析（各日・各時間帯の値を分析）"""
        # 日時をキーとして各行を保持
        hourly = df[["_datetime", "_date", "_hour", "_weekday", col_name]].copy()
        hourly.columns = ["datetime", "date", "hour", "weekday", "value"]
        hourly["date"] = pd.to_datetime(hourly["date"])
        hourly["weekday_name"] = hourly["weekday"].apply(lambda x: weekday_names[x])
        
        # NaNを除外
        hourly = hourly.dropna(subset=["value"])
        
        if has_weather and weather_df is not None:
            # 時間単位で天候データを結合
            weather_hourly = weather_df[["datetime", "_date", "_hour", "temperature", 
                                         "precipitation", "sunlight", "wind_speed"]].copy()
            if has_weather_name:
                weather_hourly["天気"] = weather_df["weather"]
            
            weather_hourly.columns = ["w_datetime", "w_date", "w_hour", "気温(℃)", 
                                      "降水量(mm)", "日照時間(h)", "風速(m/s)"] + (["天気"] if has_weather_name else [])
            
            # 日付と時間でマッチング
            hourly["_merge_key"] = hourly["date"].astype(str) + "_" + hourly["hour"].astype(str)
            weather_hourly["_merge_key"] = pd.to_datetime(weather_hourly["w_date"]).astype(str) + "_" + weather_hourly["w_hour"].astype(str)
            
            # 天候データから不要列を削除
            weather_hourly = weather_hourly.drop(columns=["w_datetime", "w_date", "w_hour"])
            weather_hourly = weather_hourly.drop_duplicates(subset=["_merge_key"])
            
            hourly = hourly.merge(weather_hourly, on="_merge_key", how="left")
            hourly = hourly.drop(columns=["_merge_key"])
        
        return hourly

    def _update_demand_weather_table(self, table: QTableWidget, df: pd.DataFrame, 
                                      has_weather: bool, time_unit: str = "日別") -> None:
        """需要・天候テーブルを更新（日別/時別対応）"""
        has_weather_name = "天気" in df.columns
        is_hourly = time_unit == "時別"
        has_hour = "hour" in df.columns
        
        # カラム構築
        columns = []
        if is_hourly and has_hour:
            columns = ["日時", "時", "曜日", "値"]
        else:
            columns = ["日付", "曜日", "値"]
        
        if has_weather:
            if has_weather_name:
                columns.extend(["天気", "気温(℃)", "降水量", "日照", "風速"])
            else:
                columns.extend(["気温(℃)", "降水量", "日照", "風速"])
        
        table.clear()
        table.setRowCount(len(df))
        table.setColumnCount(len(columns))
        table.setHorizontalHeaderLabels(columns)
        
        for i, (_, row) in enumerate(df.iterrows()):
            col_idx = 0
            
            if is_hourly and has_hour:
                # 時別: 日時、時、曜日、値
                date_str = row["date"].strftime("%m/%d") if hasattr(row["date"], "strftime") else str(row["date"])[:10]
                table.setItem(i, col_idx, QTableWidgetItem(date_str))
                col_idx += 1
                table.setItem(i, col_idx, QTableWidgetItem(f"{int(row['hour']):02d}"))
                col_idx += 1
            else:
                # 日別: 日付、曜日、値
                date_str = row["date"].strftime("%Y-%m-%d") if hasattr(row["date"], "strftime") else str(row["date"])[:10]
                table.setItem(i, col_idx, QTableWidgetItem(date_str))
                col_idx += 1
            
            table.setItem(i, col_idx, QTableWidgetItem(row["weekday_name"]))
            col_idx += 1
            table.setItem(i, col_idx, QTableWidgetItem(f"{row['value']:,.2f}"))
            col_idx += 1
            
            if has_weather:
                if has_weather_name:
                    weather = row.get("天気", "")
                    table.setItem(i, col_idx, QTableWidgetItem(str(weather) if pd.notna(weather) else "-"))
                    col_idx += 1
                
                temp = row.get("気温(℃)", float("nan"))
                precip = row.get("降水量(mm)", float("nan"))
                sun = row.get("日照時間(h)", float("nan"))
                wind = row.get("風速(m/s)", float("nan"))
                
                table.setItem(i, col_idx, QTableWidgetItem(f"{temp:.1f}" if pd.notna(temp) else "-"))
                col_idx += 1
                table.setItem(i, col_idx, QTableWidgetItem(f"{precip:.1f}" if pd.notna(precip) else "-"))
                col_idx += 1
                table.setItem(i, col_idx, QTableWidgetItem(f"{sun:.1f}" if pd.notna(sun) else "-"))
                col_idx += 1
                table.setItem(i, col_idx, QTableWidgetItem(f"{wind:.1f}" if pd.notna(wind) else "-"))
        
        table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)

    def copy_demand_weather_result(self) -> None:
        """需要・天候分析結果をコピー"""
        if not hasattr(self, "_dw_top_records") or self._dw_top_records is None:
            QtWidgets.QMessageBox.warning(self, "警告", "先に分析を実行してください。")
            return
        
        lines = []
        has_weather = self._dw_has_weather
        has_weather_name = "天気" in self._dw_top_records.columns
        time_unit = getattr(self, "_dw_time_unit", "日別")
        is_hourly = time_unit == "時別"
        has_hour = "hour" in self._dw_top_records.columns
        
        # ヘッダー作成
        if is_hourly and has_hour:
            header_parts = ["日付", "時", "曜日", "値"]
        else:
            header_parts = ["日付", "曜日", "値"]
        
        if has_weather:
            if has_weather_name:
                header_parts.extend(["天気", "気温(℃)", "降水量(mm)", "日照(h)", "風速(m/s)"])
            else:
                header_parts.extend(["気温(℃)", "降水量(mm)", "日照(h)", "風速(m/s)"])
        
        header = "\t".join(header_parts)
        unit_label = "時間帯" if is_hourly else "日"
        
        # 高い日/時間帯
        lines.append(f"【高い{unit_label}】")
        lines.append(header)
        
        for _, row in self._dw_top_records.iterrows():
            date_str = row["date"].strftime("%Y-%m-%d") if hasattr(row["date"], "strftime") else str(row["date"])[:10]
            
            if is_hourly and has_hour:
                line = [date_str, f"{int(row['hour']):02d}", row["weekday_name"], f"{row['value']:.2f}"]
            else:
                line = [date_str, row["weekday_name"], f"{row['value']:.2f}"]
            
            if has_weather:
                if has_weather_name:
                    weather = row.get('天気', '')
                    line.append(str(weather) if pd.notna(weather) else "")
                line.extend([
                    f"{row.get('気温(℃)', 0):.1f}",
                    f"{row.get('降水量(mm)', 0):.1f}",
                    f"{row.get('日照時間(h)', 0):.1f}",
                    f"{row.get('風速(m/s)', 0):.1f}"
                ])
            lines.append("\t".join(line))
        
        lines.append("")
        lines.append(f"【低い{unit_label}】")
        lines.append(header)
        
        for _, row in self._dw_bottom_records.iterrows():
            date_str = row["date"].strftime("%Y-%m-%d") if hasattr(row["date"], "strftime") else str(row["date"])[:10]
            
            if is_hourly and has_hour:
                line = [date_str, f"{int(row['hour']):02d}", row["weekday_name"], f"{row['value']:.2f}"]
            else:
                line = [date_str, row["weekday_name"], f"{row['value']:.2f}"]
            
            if has_weather:
                if has_weather_name:
                    weather = row.get('天気', '')
                    line.append(str(weather) if pd.notna(weather) else "")
                line.extend([
                    f"{row.get('気温(℃)', 0):.1f}",
                    f"{row.get('降水量(mm)', 0):.1f}",
                    f"{row.get('日照時間(h)', 0):.1f}",
                    f"{row.get('風速(m/s)', 0):.1f}"
                ])
            lines.append("\t".join(line))
        
        text = "\n".join(lines)
        clipboard = QApplication.clipboard()
        clipboard.setText(text)
        
        QtWidgets.QMessageBox.information(
            self, "コピー完了",
            "需要・天候分析結果をクリップボードにコピーしました。"
        )

def main():
    try:
        print("Starting application...")
        app = QApplication(sys.argv)
        print("QApplication created")
        window = MainWindow()
        print("MainWindow created")
        window.show()
        print("Window shown")
        print("Entering event loop...")
        ret = app.exec()
        print(f"Event loop exited with code {ret}")
        sys.exit(ret)
    except Exception as e:
        import traceback
        traceback.print_exc()
        input("Press Enter to exit...")

if __name__ == "__main__":
    main()

