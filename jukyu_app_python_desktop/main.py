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
from matplotlib.figure import Figure
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
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
    QVBoxLayout,
    QWidget,
)

from ml import DemandTransformerForecaster, ForecastResult

# 日本語フォントの設定
plt.rcParams['font.sans-serif'] = ['MS Gothic', 'Yu Gothic', 'Meiryo', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

@dataclass(frozen=True)
class AreaInfo:
    """Metadata describing a supply-demand area."""

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

FNAME = re.compile(r"^eria_jukyu_(\d{6})_(\d{2})\.csv$")
DATA_DIR = Path(__file__).resolve().parent / "data"

def scan_files() -> List[DataFileEntry]:
    """Return chronological list of CSV files present in the data directory."""

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
    """Build inclusive year and month ranges from YYYYMM strings."""

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
    """Read a CSV file using several common Japanese encodings and return (data, time-column)."""

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

class MplCanvas(FigureCanvas):
    """Thin matplotlib canvas wrapper that exposes the Axes for plotting."""

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
        """Adjust the canvas size and trigger redraw."""

        self.fig.set_size_inches(width, height)
        self.fig.set_dpi(dpi)
        self.draw()


class MainWindow(QMainWindow):
    """Main Qt window hosting the data availability and analytical views."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("⚡ 電力需給実績ビューア")
        self.resize(1400, 840)
        
        # グラフ設定のデフォルト値
        self.graph_settings: Dict[str, Any] = {
            'title': '',
            'xlabel': '時刻',
            'ylabel': '電力 (MW)',
            'linewidth': 2.0,
            'grid': True,
            'legend': True,
            'legend_loc': 'best',
            'figsize_w': 12,
            'figsize_h': 6,
            'dpi': 100,
            'font_size': 12,
            'title_size': 14,
            'label_size': 12
        }
        self.selected_columns: List[str] = []  # 選択された発電方式
        self.current_dataframe: Optional[pd.DataFrame] = None
        self.current_time_column: Optional[str] = None
        self.current_dataset_key: Optional[str] = None

        # AI分析関連
        self.ai_dataframe: Optional[pd.DataFrame] = None
        self.ai_time_column: Optional[str] = None
        self.ai_target_series: Optional[pd.Series] = None
        self.ai_training_index: Optional[pd.Index] = None
        self.ai_forecaster: Optional[DemandTransformerForecaster] = None
        self.area_year_months: Dict[AreaCode, List[YearMonth]] = {code: [] for code in AREA_INFO}
        
        self.apply_modern_palette()

        # タブウィジェット作成
        self.tabs = QTabWidget()
        self.tabs.setStyleSheet("""
            QTabWidget::pane {
                border: 2px solid #a0d2ff;
                border-radius: 8px;
                background-color: #ffffff;
            }
            QTabBar::tab {
                background-color: #e6f2ff;
                color: #0068B7;
                padding: 12px 24px;
                margin-right: 2px;
                border: 2px solid #a0d2ff;
                border-bottom: none;
                border-top-left-radius: 8px;
                border-top-right-radius: 8px;
                font-weight: 600;
                font-size: 14px;
            }
            QTabBar::tab:selected {
                background-color: #ffffff;
                color: #0068B7;
            }
            QTabBar::tab:hover {
                background-color: #cfe7ff;
            }
        """)
        
        # メインページタブ
        self.main_page = self.create_main_page()
        self.tabs.addTab(self.main_page, "📊 メイン")
        
        # 詳細ページタブ
        self.detail_page = self.create_detail_page()
        self.tabs.addTab(self.detail_page, "📈 詳細分析")

        # AI分析タブ
        self.ai_page = self.create_ai_page()
        self.tabs.addTab(self.ai_page, "🤖 AI分析")

        self.setCentralWidget(self.tabs)

        # (YYYYMM, area code, path) tuples discovered under data/.
        self.files = scan_files()  # type: List[DataFileEntry]
        self.avail, self.years, self.months = build_availability(self.files)
        self.refresh_area_year_months()
        self.refresh_heatmap()
        self.area_combo.currentIndexChanged.connect(self.on_area_change)
        self.on_area_change()
        self.populate_ai_controls()

    def create_main_page(self):
        """メインページの作成"""
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)

        # ヘッダーセクション
        header = QHBoxLayout()
        header.setSpacing(12)
        
        title_label = QLabel("⚡ 電力需給実績データビューア")
        title_label.setStyleSheet("""
            font-size: 24px; 
            font-weight: bold; 
            color: #0068B7; 
            padding: 8px 0px;
        """)
        header.addWidget(title_label)
        header.addStretch()
        
        # コントロールエリア
        ctrl = QHBoxLayout()
        ctrl.setSpacing(10)
        
        area_label = QLabel("📍 エリア:")
        area_label.setStyleSheet("font-weight: 600; font-size: 13px; color: #0068B7;")
        self.area_combo = QComboBox()
        self.area_combo.setMinimumWidth(180)
        for code, meta in AREA_INFO.items():
            self.area_combo.addItem(f"({code}) {meta.name}", code)
        
        self.url_btn = QPushButton("🌐 公式サイト")
        self.url_btn.setMinimumHeight(36)
        self.url_btn.clicked.connect(self.open_official)
        
        self.load_btn = QPushButton("📂 データフォルダ")
        self.load_btn.setMinimumHeight(36)
        self.load_btn.clicked.connect(self.open_folder)
        
        detail_btn = QPushButton("📈 詳細分析へ")
        detail_btn.setMinimumHeight(36)
        detail_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                          stop:0 #10b981, stop:1 #059669);
                color: white;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                          stop:0 #34d399, stop:1 #10b981);
            }
        """)
        detail_btn.clicked.connect(lambda: self.tabs.setCurrentIndex(1))
        
        ctrl.addWidget(area_label)
        ctrl.addWidget(self.area_combo)
        ctrl.addStretch()
        ctrl.addWidget(self.load_btn)
        ctrl.addWidget(self.url_btn)
        ctrl.addWidget(detail_btn)
        
        layout.addLayout(header)
        layout.addLayout(ctrl)

        # ヒートマップセクション
        heatmap_label = QLabel("📊 データ可用性マップ")
        heatmap_label.setStyleSheet("font-size: 16px; font-weight: 600; color: #0068B7; margin-top: 8px;")
        layout.addWidget(heatmap_label)
        
        self.heat_table = QTableWidget()
        self.heat_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.heat_table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self.heat_table.setStyleSheet("""
            QTableWidget {
                border: 2px solid #a0d2ff;
                border-radius: 10px;
                background-color: #ffffff;
            }
        """)
        layout.addWidget(self.heat_table, stretch=1)
        
        return page

    def create_detail_page(self):
        """詳細ページの作成"""
        page = QWidget()
        main_layout = QVBoxLayout(page)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(15)
        
        # ヘッダー
        header = QHBoxLayout()
        back_btn = QPushButton("← メインに戻る")
        back_btn.setMinimumHeight(36)
        back_btn.clicked.connect(lambda: self.tabs.setCurrentIndex(0))
        header.addWidget(back_btn)
        header.addStretch()
        main_layout.addLayout(header)
        
        # 3分割レイアウト
        content = QSplitter(Qt.Horizontal)
        
        # 左パネル: データ選択
        left_panel = self.create_data_selection_panel()
        content.addWidget(left_panel)
        
        # 中央パネル: グラフ設定
        center_panel = self.create_graph_settings_panel()
        content.addWidget(center_panel)
        
        # 右パネル: グラフ表示
        right_panel = self.create_graph_display_panel()
        content.addWidget(right_panel)
        
        content.setSizes([350, 350, 700])
        main_layout.addWidget(content)

        return page

    def create_ai_page(self) -> QWidget:
        """AIモデルを活用した分析タブを構築"""

        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)

        header = QHBoxLayout()
        title = QLabel("🤖 AI 需給分析ラボ")
        title.setStyleSheet("font-size: 20px; font-weight: 600; color: #0068B7;")
        header.addWidget(title)
        header.addStretch()
        layout.addLayout(header)

        desc = QLabel(
            "Transformerベースの需要予測モデルを試すための実験的なラボです。"
            "選択したCSVから目的の系列を取り出し、学習と予測を実行できます。"
        )
        desc.setWordWrap(True)
        layout.addWidget(desc)

        selection_group = QGroupBox("データセット選択")
        selection_form = QFormLayout()

        self.ai_area_combo = QComboBox()
        self.ai_area_combo.setMinimumHeight(32)
        for code, meta in AREA_INFO.items():
            self.ai_area_combo.addItem(f"({code}) {meta.name}", code)
        self.ai_area_combo.currentIndexChanged.connect(self.on_ai_area_change)
        selection_form.addRow("エリア", self.ai_area_combo)

        self.ai_ym_combo = QComboBox()
        self.ai_ym_combo.setMinimumHeight(32)
        self.ai_ym_combo.currentIndexChanged.connect(self.on_ai_ym_change)
        selection_form.addRow("年月", self.ai_ym_combo)

        self.ai_column_combo = QComboBox()
        self.ai_column_combo.setMinimumHeight(32)
        selection_form.addRow("目的系列", self.ai_column_combo)

        selection_group.setLayout(selection_form)
        layout.addWidget(selection_group)

        params_group = QGroupBox("Transformerハイパーパラメータ")
        params_form = QFormLayout()

        self.ai_context_spin = QSpinBox()
        self.ai_context_spin.setRange(12, 5000)
        self.ai_context_spin.setValue(96)
        params_form.addRow("コンテキスト長", self.ai_context_spin)

        self.ai_horizon_spin = QSpinBox()
        self.ai_horizon_spin.setRange(1, 240)
        self.ai_horizon_spin.setValue(24)
        params_form.addRow("予測ステップ数", self.ai_horizon_spin)

        self.ai_epoch_spin = QSpinBox()
        self.ai_epoch_spin.setRange(1, 200)
        self.ai_epoch_spin.setValue(30)
        params_form.addRow("エポック", self.ai_epoch_spin)

        self.ai_batch_spin = QSpinBox()
        self.ai_batch_spin.setRange(4, 512)
        self.ai_batch_spin.setValue(64)
        params_form.addRow("バッチサイズ", self.ai_batch_spin)

        self.ai_lr_spin = QDoubleSpinBox()
        self.ai_lr_spin.setDecimals(5)
        self.ai_lr_spin.setRange(1e-5, 1e-1)
        self.ai_lr_spin.setSingleStep(1e-4)
        self.ai_lr_spin.setValue(5e-4)
        params_form.addRow("学習率", self.ai_lr_spin)

        params_group.setLayout(params_form)
        layout.addWidget(params_group)

        btn_row = QHBoxLayout()
        self.ai_prepare_btn = QPushButton("📚 データ要約")
        self.ai_prepare_btn.clicked.connect(self.prepare_ai_dataset)
        btn_row.addWidget(self.ai_prepare_btn)

        self.ai_train_btn = QPushButton("🤖 Transformer学習＆予測")
        self.ai_train_btn.clicked.connect(self.train_transformer_model)
        btn_row.addWidget(self.ai_train_btn)

        btn_row.addStretch()
        layout.addLayout(btn_row)

        self.ai_log_output = QPlainTextEdit()
        self.ai_log_output.setReadOnly(True)
        self.ai_log_output.setPlaceholderText("AI分析の進捗がここに表示されます…")
        layout.addWidget(self.ai_log_output)

        self.ai_result_table = QTableWidget()
        self.ai_result_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.ai_result_table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self.ai_result_table.verticalHeader().setVisible(False)
        self.ai_result_table.setAlternatingRowColors(True)
        self.ai_result_table.setStyleSheet(
            """
            QTableWidget {
                border: 2px solid #a0d2ff;
                border-radius: 8px;
                background-color: #ffffff;
                alternate-background-color: #f8fafc;
            }
            """
        )
        layout.addWidget(self.ai_result_table, stretch=1)

        return page

    def refresh_area_year_months(self) -> None:
        """Recompute available year-month combinations per area."""

        for code in AREA_INFO.keys():
            self.area_year_months[code] = []
        for year_month, code, _ in self.files:
            self.area_year_months.setdefault(code, []).append(year_month)
        for code, values in self.area_year_months.items():
            unique_sorted = sorted(set(values))
            self.area_year_months[code] = unique_sorted

    def populate_ai_controls(self) -> None:
        """Fill AI tab combos based on scanned files."""

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

    def on_ai_area_change(self) -> None:
        """Populate the year-month combo when the area changes."""

        if not hasattr(self, "ai_ym_combo"):
            return
        code = self.ai_area_combo.currentData()
        self.ai_ym_combo.blockSignals(True)
        self.ai_ym_combo.clear()
        for ym in self.area_year_months.get(code, []):
            display = f"{ym[:4]}年{ym[4:6]}月"
            self.ai_ym_combo.addItem(display, ym)
        self.ai_ym_combo.blockSignals(False)
        self.ai_dataframe = None
        self.ai_time_column = None
        self.ai_target_series = None
        self.ai_training_index = None
        self.ai_column_combo.clear()
        if self.ai_ym_combo.count() > 0:
            self.ai_ym_combo.setCurrentIndex(0)
        else:
            self.append_ai_log("選択したエリアのCSVが見つかりません。data/にファイルを追加してください。")

    def on_ai_ym_change(self) -> None:
        """Load the selected dataset and populate the column combo."""

        self.load_ai_dataset()

    def load_ai_dataset(self) -> None:
        code = self.ai_area_combo.currentData()
        ym = self.ai_ym_combo.currentData()
        if not code or not ym:
            return
        path = DATA_DIR / f"eria_jukyu_{ym}_{code}.csv"
        if not path.exists():
            self.append_ai_log(f"CSVが見つかりません: {path.name}")
            self.ai_dataframe = None
            self.ai_column_combo.clear()
            return
        try:
            df, time_col = read_csv(path)
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

    def prepare_ai_dataset(self) -> None:
        if self.ai_dataframe is None:
            self.load_ai_dataset()
        if self.ai_dataframe is None:
            QtWidgets.QMessageBox.warning(self, "警告", "データが読み込まれていません。")
            return
        target_column = self.ai_column_combo.currentText()
        if not target_column:
            QtWidgets.QMessageBox.warning(self, "警告", "目的系列を選択してください。")
            return
        series = pd.to_numeric(self.ai_dataframe[target_column], errors="coerce")
        valid = series.dropna()
        self.ai_target_series = series
        self.ai_training_index = valid.index
        self.append_ai_log(
            f"列 '{target_column}' を読み込みました。利用可能なサンプル: {len(valid):,}件 / 欠損: {series.isna().sum():,}件"
        )
        context_len = min(self.ai_context_spin.value(), len(valid))
        if context_len == 0:
            self.append_ai_log("十分なデータがありません。CSVを確認してください。")
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
        """Show the latest context window in the result table."""

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
            self.ai_result_table.setItem(row_idx, 0, QTableWidgetItem("履歴"))
            self.ai_result_table.setItem(row_idx, 1, QTableWidgetItem(ts_text))
            self.ai_result_table.setItem(row_idx, 2, QTableWidgetItem(f"{float(value):,.2f}"))

    def train_transformer_model(self) -> None:
        if self.ai_dataframe is None:
            self.load_ai_dataset()
        if self.ai_dataframe is None:
            QtWidgets.QMessageBox.warning(self, "警告", "データが読み込まれていません。")
            return
        target_column = self.ai_column_combo.currentText()
        if not target_column:
            QtWidgets.QMessageBox.warning(self, "警告", "目的系列を選択してください。")
            return
        if self.ai_target_series is None:
            self.prepare_ai_dataset()
        if self.ai_target_series is None:
            return
        series = pd.to_numeric(self.ai_target_series, errors="coerce")
        series_interpolated = (
            series.interpolate(limit_direction="both")
            .fillna(method="bfill")
            .fillna(method="ffill")
        )
        context_length = self.ai_context_spin.value()
        prediction_length = self.ai_horizon_spin.value()
        if len(series_interpolated.dropna()) < context_length + prediction_length:
            QtWidgets.QMessageBox.warning(
                self,
                "警告",
                "学習に十分なデータがありません。コンテキスト長や予測ステップ数を調整してください。",
            )
            return

        epochs = self.ai_epoch_spin.value()
        batch_size = self.ai_batch_spin.value()
        learning_rate = self.ai_lr_spin.value()

        self.append_ai_log(
            f"Transformerを初期化します (context={context_length}, horizon={prediction_length}, epochs={epochs})."
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
            QtWidgets.QMessageBox.critical(self, "エラー", str(exc))
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
        """Render forecast results in the AI result table."""

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
                history_times = time_series.iloc[-history_len:]
            except Exception:
                history_times = pd.Series([None] * history_len)
        else:
            history_times = pd.Series([None] * history_len)

        for row, (idx, value) in enumerate(zip(history_index, result.history)):
            ts_text = ""
            if history_times[row] is not None and pd.notna(history_times[row]):
                ts_text = pd.to_datetime(history_times[row]).strftime("%Y-%m-%d %H:%M")
            else:
                ts_text = str(idx)
            self.ai_result_table.setItem(row, 0, QTableWidgetItem("履歴"))
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
    
    def create_data_selection_panel(self):
        """データ選択パネル"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(10, 10, 10, 10)
        
        title = QLabel("📅 データ選択")
        title.setStyleSheet("font-size: 16px; font-weight: 600; color: #0068B7;")
        layout.addWidget(title)
        
        # 年月選択
        ym_group = QGroupBox("年月")
        ym_layout = QVBoxLayout()
        self.ym_combo = QComboBox()
        self.ym_combo.setMinimumHeight(36)
        self.ym_combo.currentIndexChanged.connect(self.on_ym_change)
        ym_layout.addWidget(self.ym_combo)
        ym_group.setLayout(ym_layout)
        layout.addWidget(ym_group)
        
        # 日付選択
        date_group = QGroupBox("日付")
        date_layout = QVBoxLayout()
        self.date_combo = QComboBox()
        self.date_combo.setMinimumHeight(36)
        self.date_combo.addItem("全期間", "all")
        self.date_combo.currentIndexChanged.connect(self.on_date_change)
        date_layout.addWidget(self.date_combo)
        date_group.setLayout(date_layout)
        layout.addWidget(date_group)
        
        # 発電方式選択
        column_group = QGroupBox("表示する発電方式")
        column_layout = QVBoxLayout()
        
        # スクロールエリア
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMinimumHeight(300)
        
        scroll_widget = QWidget()
        self.column_checkbox_layout = QVBoxLayout(scroll_widget)
        self.column_checkboxes = {}
        
        scroll.setWidget(scroll_widget)
        column_layout.addWidget(scroll)
        
        # 全選択/全解除ボタン
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
        
        # 可視化ボタン
        self.view_btn = QPushButton("📈 グラフ更新")
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
        
        layout.addStretch()
        return panel
    
    def create_graph_settings_panel(self):
        """グラフ設定パネル"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(10, 10, 10, 10)
        
        title = QLabel("⚙️ グラフ設定")
        title.setStyleSheet("font-size: 16px; font-weight: 600; color: #0068B7;")
        layout.addWidget(title)
        
        # スクロールエリア
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        
        scroll_widget = QWidget()
        settings_layout = QVBoxLayout(scroll_widget)
        
        # タイトル設定
        title_group = QGroupBox("タイトル")
        title_layout = QVBoxLayout()
        self.title_input = QLineEdit()
        self.title_input.setPlaceholderText("グラフタイトル（空欄で自動）")
        self.title_input.textChanged.connect(lambda: self.update_setting('title', self.title_input.text()))
        title_layout.addWidget(self.title_input)
        title_group.setLayout(title_layout)
        settings_layout.addWidget(title_group)
        
        # 軸ラベル設定
        label_group = QGroupBox("軸ラベル")
        label_layout = QGridLayout()
        
        self.xlabel_input = QLineEdit(self.graph_settings['xlabel'])
        self.xlabel_input.textChanged.connect(lambda: self.update_setting('xlabel', self.xlabel_input.text()))
        self.ylabel_input = QLineEdit(self.graph_settings['ylabel'])
        self.ylabel_input.textChanged.connect(lambda: self.update_setting('ylabel', self.ylabel_input.text()))
        
        label_layout.addWidget(QLabel("X軸:"), 0, 0)
        label_layout.addWidget(self.xlabel_input, 0, 1)
        label_layout.addWidget(QLabel("Y軸:"), 1, 0)
        label_layout.addWidget(self.ylabel_input, 1, 1)
        
        label_group.setLayout(label_layout)
        settings_layout.addWidget(label_group)
        
        # 線の太さ
        line_group = QGroupBox("線の太さ")
        line_layout = QHBoxLayout()
        self.linewidth_spin = QDoubleSpinBox()
        self.linewidth_spin.setRange(0.5, 10.0)
        self.linewidth_spin.setSingleStep(0.5)
        self.linewidth_spin.setValue(self.graph_settings['linewidth'])
        self.linewidth_spin.valueChanged.connect(lambda: self.update_setting('linewidth', self.linewidth_spin.value()))
        line_layout.addWidget(self.linewidth_spin)
        line_group.setLayout(line_layout)
        settings_layout.addWidget(line_group)
        
        # グラフサイズ
        size_group = QGroupBox("グラフサイズ (インチ)")
        size_layout = QGridLayout()
        
        self.width_spin = QSpinBox()
        self.width_spin.setRange(4, 20)
        self.width_spin.setValue(self.graph_settings['figsize_w'])
        self.width_spin.valueChanged.connect(lambda: self.update_setting('figsize_w', self.width_spin.value()))
        
        self.height_spin = QSpinBox()
        self.height_spin.setRange(3, 15)
        self.height_spin.setValue(self.graph_settings['figsize_h'])
        self.height_spin.valueChanged.connect(lambda: self.update_setting('figsize_h', self.height_spin.value()))
        
        size_layout.addWidget(QLabel("幅:"), 0, 0)
        size_layout.addWidget(self.width_spin, 0, 1)
        size_layout.addWidget(QLabel("高さ:"), 1, 0)
        size_layout.addWidget(self.height_spin, 1, 1)
        
        size_group.setLayout(size_layout)
        settings_layout.addWidget(size_group)
        
        # DPI設定
        dpi_group = QGroupBox("DPI (解像度)")
        dpi_layout = QHBoxLayout()
        self.dpi_spin = QSpinBox()
        self.dpi_spin.setRange(50, 300)
        self.dpi_spin.setSingleStep(10)
        self.dpi_spin.setValue(self.graph_settings['dpi'])
        self.dpi_spin.valueChanged.connect(lambda: self.update_setting('dpi', self.dpi_spin.value()))
        dpi_layout.addWidget(self.dpi_spin)
        dpi_group.setLayout(dpi_layout)
        settings_layout.addWidget(dpi_group)
        
        # フォントサイズ
        font_group = QGroupBox("フォントサイズ")
        font_layout = QGridLayout()
        
        self.font_size_spin = QSpinBox()
        self.font_size_spin.setRange(6, 24)
        self.font_size_spin.setValue(self.graph_settings['font_size'])
        self.font_size_spin.valueChanged.connect(lambda: self.update_setting('font_size', self.font_size_spin.value()))
        
        self.title_size_spin = QSpinBox()
        self.title_size_spin.setRange(8, 32)
        self.title_size_spin.setValue(self.graph_settings['title_size'])
        self.title_size_spin.valueChanged.connect(lambda: self.update_setting('title_size', self.title_size_spin.value()))
        
        font_layout.addWidget(QLabel("一般:"), 0, 0)
        font_layout.addWidget(self.font_size_spin, 0, 1)
        font_layout.addWidget(QLabel("タイトル:"), 1, 0)
        font_layout.addWidget(self.title_size_spin, 1, 1)
        
        font_group.setLayout(font_layout)
        settings_layout.addWidget(font_group)
        
        # 表示オプション
        options_group = QGroupBox("表示オプション")
        options_layout = QVBoxLayout()
        
        self.grid_check = QCheckBox("グリッド表示")
        self.grid_check.setChecked(self.graph_settings['grid'])
        self.grid_check.toggled.connect(lambda: self.update_setting('grid', self.grid_check.isChecked()))
        
        self.legend_check = QCheckBox("凡例表示")
        self.legend_check.setChecked(self.graph_settings['legend'])
        self.legend_check.toggled.connect(lambda: self.update_setting('legend', self.legend_check.isChecked()))
        
        options_layout.addWidget(self.grid_check)
        options_layout.addWidget(self.legend_check)
        
        # 凡例位置
        legend_loc_layout = QHBoxLayout()
        legend_loc_layout.addWidget(QLabel("凡例位置:"))
        self.legend_loc_combo = QComboBox()
        self.legend_loc_combo.addItems(["best", "upper right", "upper left", "lower left", "lower right", "right", "center left", "center right", "lower center", "upper center", "center"])
        self.legend_loc_combo.setCurrentText(self.graph_settings['legend_loc'])
        self.legend_loc_combo.currentTextChanged.connect(lambda: self.update_setting('legend_loc', self.legend_loc_combo.currentText()))
        legend_loc_layout.addWidget(self.legend_loc_combo)
        options_layout.addLayout(legend_loc_layout)
        
        options_group.setLayout(options_layout)
        settings_layout.addWidget(options_group)
        
        # 保存ボタン
        save_btn = QPushButton("💾 グラフを保存")
        save_btn.clicked.connect(self.save_graph)
        settings_layout.addWidget(save_btn)
        
        settings_layout.addStretch()
        scroll.setWidget(scroll_widget)
        layout.addWidget(scroll)
        
        return panel
    
    def create_graph_display_panel(self):
        """グラフ表示パネル"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(10, 10, 10, 10)
        
        title = QLabel("📊 グラフ & データプレビュー")
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

        # データプレビューコンテナ
        table_container = QFrame()
        table_container.setStyleSheet(frame_style)
        table_layout = QVBoxLayout(table_container)
        table_layout.setContentsMargins(5, 5, 5, 5)

        self.preview_info_label = QLabel("データが読み込まれていません")
        self.preview_info_label.setStyleSheet("font-weight: 600; color: #0068B7;")
        table_layout.addWidget(self.preview_info_label)

        self.preview_table = QTableWidget()
        self.preview_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.preview_table.setAlternatingRowColors(True)
        self.preview_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.preview_table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.preview_table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self.preview_table.verticalHeader().setVisible(False)
        self.preview_table.setStyleSheet("""
            QTableWidget {
                background-color: #ffffff;
                alternate-background-color: #f8fafc;
                border: none;
                color: #2d3748;
            }
        """)
        table_layout.addWidget(self.preview_table)

        # キャンバスコンテナ
        canvas_container = QFrame()
        canvas_container.setStyleSheet(frame_style)
        canvas_layout = QVBoxLayout(canvas_container)
        canvas_layout.setContentsMargins(5, 5, 5, 5)

        self.canvas = MplCanvas(
            width=self.graph_settings['figsize_w'],
            height=self.graph_settings['figsize_h'],
            dpi=self.graph_settings['dpi']
        )
        canvas_layout.addWidget(self.canvas, stretch=1)

        splitter.addWidget(table_container)
        splitter.addWidget(canvas_container)
        splitter.setSizes([260, 480])

        layout.addWidget(splitter, stretch=1)
        
        return panel
    
    def update_setting(self, key, value):
        """設定を更新"""
        self.graph_settings[key] = value

    def populate_preview_table(self, df: pd.DataFrame | None):
        """プレビュー表を更新"""
        if not hasattr(self, "preview_table") or self.preview_table is None:
            return

        if df is None or df.empty:
            self.preview_table.clear()
            self.preview_table.setRowCount(0)
            self.preview_table.setColumnCount(0)
            self.preview_info_label.setText("対象データがありません")
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
            self.preview_info_label.setText(f"{total_rows:,}件中 上位{displayed_rows:,}件を表示")
        else:
            self.preview_info_label.setText(f"{total_rows:,}件のデータを表示中")
    
    def update_column_checkboxes(self):
        """発電方式チェックボックスを更新"""
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
            
            # 数値カラムを取得
            for c in df.columns:
                if pd.api.types.is_numeric_dtype(df[c]):
                    col_str = str(c).lower()
                    if not any(keyword in col_str for keyword in ['date', 'time', '時刻', '日時', '日付']):
                        checkbox = QCheckBox(str(c))
                        checkbox.setChecked(True)  # デフォルトで選択
                        checkbox.toggled.connect(self.on_column_selection_changed)
                        self.column_checkbox_layout.addWidget(checkbox)
                        self.column_checkboxes[str(c)] = checkbox

            self.selected_columns = list(self.column_checkboxes.keys())
            self.on_date_change()
        except Exception as e:
            print(f"Error loading columns: {e}")
            self.populate_preview_table(None)
    
    def on_column_selection_changed(self):
        """発電方式選択が変更された時"""
        self.selected_columns = [col for col, cb in self.column_checkboxes.items() if cb.isChecked()]
    
    def select_all_columns(self):
        """全ての発電方式を選択"""
        for checkbox in self.column_checkboxes.values():
            checkbox.setChecked(True)
    
    def deselect_all_columns(self):
        """全ての発電方式を解除"""
        for checkbox in self.column_checkboxes.values():
            checkbox.setChecked(False)
    
    def save_graph(self):
        """グラフを画像として保存"""
        from PySide6.QtWidgets import QFileDialog
        
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "グラフを保存",
            "",
            "PNG (*.png);;PDF (*.pdf);;SVG (*.svg)"
        )
        
        if filename:
            self.canvas.fig.savefig(filename, dpi=self.graph_settings['dpi'], bbox_inches='tight')
            QtWidgets.QMessageBox.information(self, "成功", f"グラフを保存しました:\n{filename}")

    def apply_modern_palette(self):
        pal = QtGui.QPalette()
        # 東京都市大学の青系カラーテーマ
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

    def open_official(self) -> None:
        """Open the official data portal for the currently selected area."""

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

    def refresh_heatmap(self):
        self.heat_table.clear()
        self.heat_table.setColumnCount(13)
        self.heat_table.setHorizontalHeaderLabels(["年"] + [f"{m}月" for m in self.months])
        self.heat_table.setRowCount(len(self.years))
        self.heat_table.verticalHeader().setVisible(False)

        code = self.area_combo.currentData()
        for r, y in enumerate(self.years):
            item = QTableWidgetItem(str(y))
            item.setFlags(Qt.ItemIsEnabled)
            item.setForeground(QtGui.QBrush(QtGui.QColor("#0068B7")))
            item.setFont(QtGui.QFont("", 11, QtGui.QFont.Bold))
            self.heat_table.setItem(r, 0, item)
            for c, m in enumerate(self.months, start=1):
                ok = self.avail.get(code, {}).get(y, {}).get(m, False)
                cell = QTableWidgetItem("✓" if ok else "—")
                cell.setTextAlignment(Qt.AlignCenter)
                if ok:
                    bg = QtGui.QColor("#10b981")
                    cell.setForeground(QtGui.QBrush(QtGui.QColor("#ffffff")))
                    cell.setFont(QtGui.QFont("", 12, QtGui.QFont.Bold))
                else:
                    bg = QtGui.QColor("#f87171")
                    cell.setForeground(QtGui.QBrush(QtGui.QColor("#ffffff")))
                cell.setBackground(bg)
                self.heat_table.setItem(r, c, cell)

    def on_area_change(self):
        self.files = scan_files()
        self.avail, self.years, self.months = build_availability(self.files)
        code = self.area_combo.currentData()
        yms = sorted([ym for (ym, a, _) in self.files if a == code])
        self.ym_combo.blockSignals(True)
        self.ym_combo.clear()
        for ym in yms:
            self.ym_combo.addItem(f"{ym[:4]}年{ym[4:6]}月", ym)
        self.ym_combo.blockSignals(False)
        self.refresh_heatmap()
        if self.ym_combo.count() > 0:
            self.ym_combo.blockSignals(True)
            self.ym_combo.setCurrentIndex(self.ym_combo.count() - 1)
            self.ym_combo.blockSignals(False)
            self.on_ym_change()
        else:
            self.date_combo.clear()
            self.date_combo.addItem("全期間", "all")
            self.populate_preview_table(None)

    def on_ym_change(self):
        """年月が変更された時に日付リストを更新"""
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
                    date_str = date.strftime("%Y年%m月%d日")
                    self.date_combo.addItem(date_str, str(date))
            self.date_combo.blockSignals(False)

            # 発電方式チェックボックスを更新
            self.update_column_checkboxes()
        except Exception:
            self.date_combo.blockSignals(False)
            self.populate_preview_table(None)

    def on_date_change(self):
        """日付選択が変わった際にプレビューを更新"""
        if self.current_dataframe is None:
            self.populate_preview_table(None)
            return

        df = self.current_dataframe.copy()
        tcol = self.current_time_column
        selected_date = self.date_combo.currentData()

        if selected_date and selected_date != "all" and tcol and tcol in df.columns:
            try:
                df['_date'] = pd.to_datetime(df[tcol]).dt.date
                filter_date = pd.to_datetime(selected_date).date()
                df = df[df['_date'] == filter_date].copy()
                df = df.drop(columns=['_date'])
            except Exception:
                pass

        self.populate_preview_table(df)

    def render_view(self):
        code = self.area_combo.currentData()
        ym = self.ym_combo.currentData()
        if not ym:
            QtWidgets.QMessageBox.information(self, "情報", "年月を選択してください。")
            return
        
        path = DATA_DIR / f"eria_jukyu_{ym}_{code}.csv"
        if not path.exists():
            QtWidgets.QMessageBox.warning(self, "警告", "選択年月のCSVが見つかりません。")
            return
        
        dataset_key = (code, ym)
        if self.current_dataset_key != dataset_key or self.current_dataframe is None:
            try:
                self.update_column_checkboxes()
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "エラー", f"CSVファイルの読み込みに失敗しました:\n{str(e)}")
                return

        if self.current_dataframe is None:
            QtWidgets.QMessageBox.warning(self, "警告", "データを読み込めませんでした。")
            return

        df = self.current_dataframe.copy()
        tcol = self.current_time_column
        
        # 日付フィルタリング
        selected_date = self.date_combo.currentData()
        df_filtered = df.copy()
        
        if selected_date and selected_date != "all" and tcol and tcol in df.columns:
            try:
                df_filtered['_date'] = pd.to_datetime(df[tcol]).dt.date
                filter_date = pd.to_datetime(selected_date).date()
                df_filtered = df_filtered[df_filtered['_date'] == filter_date].copy()
                df_filtered = df_filtered.drop(columns=['_date'])
            except Exception:
                pass
        
        if len(df_filtered) == 0:
            self.populate_preview_table(None)
            QtWidgets.QMessageBox.warning(self, "警告", "選択された日付のデータがありません")
            return

        self.populate_preview_table(df_filtered)
        
        # グラフの描画
        self.canvas.ax.clear()
        self.canvas.ax.set_facecolor('#f8fafc')
        self.canvas.ax.tick_params(colors='#2d3748', labelsize=self.graph_settings['font_size'])
        for spine in self.canvas.ax.spines.values():
            spine.set_color('#a0d2ff')
        
        # 選択された発電方式を取得
        if not self.selected_columns:
            self.selected_columns = [col for col, cb in self.column_checkboxes.items() if cb.isChecked()]
        
        if not self.selected_columns:
            QtWidgets.QMessageBox.warning(self, "警告", "表示する発電方式を選択してください")
            return
        
        # 都市大カラーパレット
        colors = ['#0068B7', '#00A0E9', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899', '#14b8a6']
        
        if tcol and tcol in df_filtered.columns:
            x_data = df_filtered[tcol]
            for idx, c in enumerate(self.selected_columns):
                if c in df_filtered.columns:
                    valid_indices = df_filtered[c].notna()
                    x_plot = x_data[valid_indices]
                    y_plot = df_filtered[c][valid_indices]
                    
                    self.canvas.ax.plot(x_plot, y_plot, label=str(c), 
                                       color=colors[idx % len(colors)], 
                                       linewidth=self.graph_settings['linewidth'], 
                                       alpha=0.9)
            self.canvas.ax.set_xlabel(self.graph_settings['xlabel'], 
                                     color='#2d3748', 
                                     fontsize=self.graph_settings['label_size'], 
                                     fontweight='bold')
            self.canvas.fig.autofmt_xdate(rotation=45)
        else:
            x = range(len(df_filtered))
            for idx, c in enumerate(self.selected_columns):
                if c in df_filtered.columns:
                    y_data = df_filtered[c].fillna(0)
                    self.canvas.ax.plot(x, y_data, label=str(c), 
                                       color=colors[idx % len(colors)], 
                                       linewidth=self.graph_settings['linewidth'], 
                                       alpha=0.9)
            self.canvas.ax.set_xlabel(self.graph_settings['xlabel'], 
                                     color='#2d3748', 
                                     fontsize=self.graph_settings['label_size'], 
                                     fontweight='bold')
        
        self.canvas.ax.set_ylabel(self.graph_settings['ylabel'], 
                                 color='#2d3748', 
                                 fontsize=self.graph_settings['label_size'], 
                                 fontweight='bold')
        
        # 凡例
        if self.graph_settings['legend']:
            legend = self.canvas.ax.legend(loc=self.graph_settings['legend_loc'], 
                                          facecolor='#ffffff', 
                                          edgecolor='#a0d2ff', 
                                          labelcolor='#2d3748', 
                                          fontsize=self.graph_settings['font_size'], 
                                          framealpha=0.95)
        
        # タイトル
        if self.graph_settings['title']:
            title_text = self.graph_settings['title']
        else:
            title_text = f"⚡ {AREA_INFO[code].name}エリア - {ym[:4]}年{ym[4:6]}月"
            if selected_date and selected_date != "all":
                date_obj = pd.to_datetime(selected_date)
                title_text += f" ({date_obj.strftime('%m月%d日')})"
        
        self.canvas.ax.set_title(title_text, 
                                color='#0068B7', 
                                fontsize=self.graph_settings['title_size'], 
                                fontweight='bold', 
                                pad=15)
        
        # グリッド
        if self.graph_settings['grid']:
            self.canvas.ax.grid(True, alpha=0.3, color='#cbd5e0', linestyle='--', linewidth=0.8)
        
        # Y軸のフォーマット
        from matplotlib.ticker import FuncFormatter
        self.canvas.ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{int(x):,}'))
        
        # サイズ更新
        self.canvas.update_size(self.graph_settings['figsize_w'], 
                               self.graph_settings['figsize_h'], 
                               self.graph_settings['dpi'])
        
        self.canvas.fig.tight_layout(pad=2.0)
        self.canvas.draw()

if __name__ == "__main__":
    app = QApplication([])
    w = MainWindow()
    w.show()
    app.exec()
