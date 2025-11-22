
import os, sys, re, webbrowser
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QTableWidget, QTableWidgetItem, QSplitter, QFrame, QTabWidget,
    QCheckBox, QSpinBox, QDoubleSpinBox, QLineEdit, QGroupBox, QScrollArea, QGridLayout
)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# 日本語フォントの設定
plt.rcParams['font.sans-serif'] = ['MS Gothic', 'Yu Gothic', 'Meiryo', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

AREA_INFO = {
    "01": {"name": "北海道", "url": "https://www.hepco.co.jp/network/con_service/public_document/supply_demand_results/index.html"},
    "02": {"name": "東北", "url": "https://setsuden.nw.tohoku-epco.co.jp/download.html"},
    "03": {"name": "東京", "url": "https://www.tepco.co.jp/forecast/html/area_jukyu-j.html"},
    "04": {"name": "中部", "url": "https://powergrid.chuden.co.jp/denkiyoho/#link02"},
    "05": {"name": "北陸", "url": "https://www.rikuden.co.jp/nw/denki-yoho/results_jyukyu.html"},
    "06": {"name": "関西", "url": "https://www.kansai-td.co.jp/denkiyoho/area-performance/index.html"},
    "07": {"name": "中国", "url": "https://www.energia.co.jp/nw/jukyuu/eria_jukyu.html"},
    "08": {"name": "四国", "url": "https://www.yonden.co.jp/nw/supply_demand/data_download.html"},
    "09": {"name": "九州", "url": "https://www.kyuden.co.jp/td_area_jukyu/jukyu.html"},
    "10": {"name": "沖縄", "url": "https://www.okiden.co.jp/business-support/service/supply-and-demand/"},
}
FNAME = re.compile(r"^eria_jukyu_(\d{6})_(\d{2})\.csv$")
DATA_DIR = Path(__file__).resolve().parent / "data"

def scan_files():
    rows = []
    if not DATA_DIR.exists(): return rows
    for p in sorted(DATA_DIR.iterdir()):
        m = FNAME.match(p.name)
        if m:
            ym, area = m.group(1), m.group(2)
            rows.append((ym, area, p))
    return rows

def build_year_month_range(all_ym):
    if not all_ym:
        y = datetime.now().year
        return list(range(y, y+1)), list(range(1, 13))
    ys = sorted({int(ym[:4]) for ym in all_ym})
    years = list(range(min(ys), max(ys)+1))
    months = list(range(1, 13))
    return years, months

def build_availability(files):
    avail = {a: {} for a in AREA_INFO.keys()}
    yms = [ym for (ym, a, _) in files]
    years, months = build_year_month_range(yms)
    for a in avail:
        for y in years:
            avail[a][y] = {m: False for m in months}
    for ym, area, _ in files:
        y = int(ym[:4]); m = int(ym[4:6])
        if area in avail and y in avail[area] and m in avail[area][y]:
            avail[area][y][m] = True
    return avail, years, months

def read_csv(path: Path):
    # 複数のエンコーディングを試す（Shift_JISを最優先）
    encodings = ["shift_jis", "cp932", "utf-8", "utf-8-sig"]
    df = None
    for enc in encodings:
        try:
            df = pd.read_csv(path, encoding=enc, engine="python", skiprows=0)
            # 最初の行が単位情報の場合はスキップ
            if '単位' in str(df.columns[0]) or 'MW' in str(df.columns[0]):
                df = pd.read_csv(path, encoding=enc, engine="python", skiprows=1)
            break
        except (UnicodeDecodeError, Exception):
            continue
    if df is None:
        raise ValueError(f"ファイルを読み込めませんでした: {path}")
    
    # DATEとTIMEカラムを結合して日時を作成
    date_col = None
    time_col = None
    for c in df.columns:
        c_str = str(c).upper()
        if 'DATE' in c_str or '日付' in str(c):
            date_col = c
        if 'TIME' in c_str or '時刻' in str(c) or '時間' in str(c):
            time_col = c
    
    tcol = None
    if date_col and time_col:
        try:
            # DATEとTIMEを結合
            df['datetime'] = pd.to_datetime(df[date_col].astype(str) + ' ' + df[time_col].astype(str), errors='coerce')
            if df['datetime'].notna().sum() > 0:
                tcol = 'datetime'
        except Exception:
            pass
    
    # 数値カラムの変換
    for c in df.columns:
        if c != tcol and c != date_col and c != time_col:
            try: 
                df[c] = pd.to_numeric(df[c], errors='coerce')
            except Exception: 
                pass
    
    # tcol がまだ見つからない場合は他の方法を試す
    if not tcol:
        for key in ["datetime","date","time","日時"]:
            for c in df.columns:
                if key.lower() in str(c).lower():
                    try:
                        parsed = pd.to_datetime(df[c], errors="coerce")
                        if parsed.notna().sum()>0:
                            df[c] = parsed; tcol=c; break
                    except Exception: pass
            if tcol: break
    
    if not tcol:
        c0 = df.columns[0]
        try:
            parsed = pd.to_datetime(df[c0], errors="coerce")
            if parsed.notna().sum()>0: df[c0]=parsed; tcol=c0
        except Exception: pass
    
    return df, tcol

class MplCanvas(FigureCanvas):
    def __init__(self):
        self.fig = Figure(figsize=(8, 5), facecolor='#ffffff')
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor('#f8fafc')
        self.ax.tick_params(colors='#2d3748', labelsize=10)
        self.ax.spines['bottom'].set_color('#a0d2ff')
        self.ax.spines['top'].set_color('#a0d2ff')
        self.ax.spines['left'].set_color('#a0d2ff')
        self.ax.spines['right'].set_color('#a0d2ff')
        self.fig.tight_layout(pad=2.0)
        super().__init__(self.fig)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("⚡ 電力需給実績ビューア")
        self.resize(1400, 840)
        self.apply_modern_palette()
        
        # グラフ設定のデフォルト値
        self.graph_settings = {
            'title': '',
            'xlabel': '時刻',
            'ylabel': '電力 (MW)',
            'linewidth': 2.0,
            'grid': True,
            'legend': True,
            'figsize_w': 12,
            'figsize_h': 6,
            'dpi': 100,
            'font_size': 12
        }
        self.selected_columns = []  # 選択された発電方式

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
                padding: 10px 20px;
                margin-right: 2px;
                border: 2px solid #a0d2ff;
                border-bottom: none;
                border-top-left-radius: 8px;
                border-top-right-radius: 8px;
                font-weight: 600;
                font-size: 13px;
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
        
        self.setCentralWidget(self.tabs)
        
        self.files = scan_files()
        self.avail, self.years, self.months = build_availability(self.files)
        self.refresh_heatmap()
        self.area_combo.currentIndexChanged.connect(self.on_area_change)
        self.on_area_change()

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
            self.area_combo.addItem(f"({code}) {meta['name']}", code)
        
        self.url_btn = QPushButton("🌐 公式サイト")
        self.url_btn.setMinimumHeight(36)
        self.url_btn.clicked.connect(self.open_official)
        
        self.load_btn = QPushButton("📂 データフォルダ")
        self.load_btn.setMinimumHeight(36)
        self.load_btn.clicked.connect(self.open_folder)
        
        detail_btn = QPushButton("📈 詳細分析へ")
        detail_btn.setMinimumHeight(36)
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

        # ボトムセクション
        bottom = QSplitter(); bottom.setOrientation(Qt.Horizontal)
        
        # 左パネル
        left = QWidget()
        left_l = QVBoxLayout(left)
        left_l.setContentsMargins(0, 0, 0, 0)
        left_l.setSpacing(10)
        
        ym_label = QLabel("📅 データ詳細")
        ym_label.setStyleSheet("font-size: 16px; font-weight: 600; color: #0068B7;")
        left_l.addWidget(ym_label)
        
        ym_row = QHBoxLayout()
        ym_row.setSpacing(8)
        self.ym_combo = QComboBox()
        self.ym_combo.setMinimumHeight(36)
        self.ym_combo.currentIndexChanged.connect(self.on_ym_change)
        
        self.view_btn = QPushButton("📈 可視化")
        self.view_btn.setMinimumHeight(36)
        self.view_btn.setMinimumWidth(100)
        self.view_btn.clicked.connect(self.render_view)
        
        ym_year_label = QLabel("年月:")
        ym_year_label.setStyleSheet("font-weight: 600; font-size: 13px; color: #0068B7;")
        ym_row.addWidget(ym_year_label)
        ym_row.addWidget(self.ym_combo, stretch=1)
        ym_row.addWidget(self.view_btn)
        left_l.addLayout(ym_row)
        
        # 日付選択行を追加
        date_row = QHBoxLayout()
        date_row.setSpacing(8)
        
        date_label = QLabel("日付選択:")
        date_label.setStyleSheet("font-weight: 600; font-size: 13px; color: #0068B7;")
        
        self.date_combo = QComboBox()
        self.date_combo.setMinimumHeight(36)
        self.date_combo.addItem("全期間", "all")
        self.date_combo.currentIndexChanged.connect(self.render_view)
        
        date_row.addWidget(date_label)
        date_row.addWidget(self.date_combo, stretch=1)
        left_l.addLayout(date_row)
        
        preview_label = QLabel("📋 データプレビュー")
        preview_label.setStyleSheet("font-size: 14px; font-weight: 600; color: #0068B7; margin-top: 5px;")
        left_l.addWidget(preview_label)
        
        self.preview = QTableWidget()
        self.preview.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.preview.setAlternatingRowColors(True)
        self.preview.setStyleSheet("""
            QTableWidget {
                border: 2px solid #a0d2ff;
                border-radius: 8px;
                background-color: #ffffff;
                alternate-background-color: #f0f9ff;
            }
        """)
        left_l.addWidget(self.preview, stretch=1)

        # 右パネル
        right = QWidget()
        right_l = QVBoxLayout(right)
        right_l.setContentsMargins(0, 0, 0, 0)
        right_l.setSpacing(10)
        
        chart_label = QLabel("📊 グラフ表示")
        chart_label.setStyleSheet("font-size: 16px; font-weight: 600; color: #0068B7;")
        right_l.addWidget(chart_label)
        
        canvas_container = QFrame()
        canvas_container.setStyleSheet("""
            QFrame {
                border: 2px solid #a0d2ff;
                border-radius: 10px;
                background-color: #ffffff;
                padding: 10px;
            }
        """)
        canvas_layout = QVBoxLayout(canvas_container)
        canvas_layout.setContentsMargins(5, 5, 5, 5)
        
        self.canvas = MplCanvas()
        canvas_layout.addWidget(self.canvas, stretch=1)
        
        right_l.addWidget(canvas_container, stretch=1)

        bottom.addWidget(left); bottom.addWidget(right); bottom.setSizes([600,600])
        main_layout.addWidget(bottom, stretch=3)

        self.files = scan_files()
        self.avail, self.years, self.months = build_availability(self.files)
        self.refresh_heatmap()
        self.area_combo.currentIndexChanged.connect(self.on_area_change)
        self.on_area_change()

    def apply_modern_palette(self):
        pal = QtGui.QPalette()
        # 東京都市大学の青系カラーテーマ
        pal.setColor(QtGui.QPalette.Window, QtGui.QColor("#f0f4f8"))  # 明るいグレー
        pal.setColor(QtGui.QPalette.WindowText, QtGui.QColor("#1a202c"))  # ダークグレー
        pal.setColor(QtGui.QPalette.Base, QtGui.QColor("#ffffff"))  # 白
        pal.setColor(QtGui.QPalette.AlternateBase, QtGui.QColor("#e6f2ff"))  # 明るい青
        pal.setColor(QtGui.QPalette.Text, QtGui.QColor("#2d3748"))  # ダークグレー
        pal.setColor(QtGui.QPalette.Button, QtGui.QColor("#ffffff"))  # 白
        pal.setColor(QtGui.QPalette.ButtonText, QtGui.QColor("#0068B7"))  # 都市大ブルー
        pal.setColor(QtGui.QPalette.Highlight, QtGui.QColor("#0068B7"))  # 都市大ブルー
        pal.setColor(QtGui.QPalette.HighlightedText, QtGui.QColor("#ffffff"))  # 白
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
        """)

    def open_official(self):
        code = self.area_combo.currentData()
        webbrowser.open(AREA_INFO[code]["url"])

    def open_folder(self):
        path = str((Path(__file__).resolve().parent / "data"))
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
                    bg = QtGui.QColor("#10b981")  # 緑
                    cell.setForeground(QtGui.QBrush(QtGui.QColor("#ffffff")))
                    cell.setFont(QtGui.QFont("", 12, QtGui.QFont.Bold))
                else:
                    bg = QtGui.QColor("#f87171")  # 赤
                    cell.setForeground(QtGui.QBrush(QtGui.QColor("#ffffff")))
                cell.setBackground(bg)
                self.heat_table.setItem(r, c, cell)
                cell.setTextAlignment(Qt.AlignCenter)
                if ok:
                    bg = QtGui.QColor("#10b981")  # 緑
                    cell.setForeground(QtGui.QBrush(QtGui.QColor("#ffffff")))
                    cell.setFont(QtGui.QFont("", 12, QtGui.QFont.Bold))
                else:
                    bg = QtGui.QColor("#ef4444")  # 赤
                    cell.setForeground(QtGui.QBrush(QtGui.QColor("#cbd5e1")))
                cell.setBackground(bg)
                self.heat_table.setItem(r, c, cell)

    def on_area_change(self):
        code = self.area_combo.currentData()
        yms = sorted([ym for (ym, a, _) in self.files if a == code])
        self.ym_combo.clear()
        for ym in yms:
            self.ym_combo.addItem(f"{ym[:4]}年{ym[4:6]}月", ym)
        self.refresh_heatmap()

    def on_ym_change(self):
        """年月が変更された時に日付リストを更新"""
        code = self.area_combo.currentData()
        ym = self.ym_combo.currentData()
        if not ym or not code:
            return
        
        path = Path(__file__).resolve().parent / "data" / f"eria_jukyu_{ym}_{code}.csv"
        if not path.exists():
            return
        
        try:
            df, tcol = read_csv(path)
            self.date_combo.clear()
            self.date_combo.addItem("全期間", "all")
            
            if tcol and tcol in df.columns:
                # 日付のユニークな値を取得
                dates = pd.to_datetime(df[tcol]).dt.date.unique()
                dates = sorted(dates)
                for date in dates:
                    date_str = date.strftime("%Y年%m月%d日")
                    self.date_combo.addItem(date_str, str(date))
        except Exception:
            pass

    def render_view(self):
        code = self.area_combo.currentData()
        ym = self.ym_combo.currentData()
        if not ym:
            QtWidgets.QMessageBox.information(self, "情報", "このエリアのCSVがまだありません。data/ に追加してください。")
            return
        path = Path(__file__).resolve().parent / "data" / f"eria_jukyu_{ym}_{code}.csv"
        if not path.exists():
            QtWidgets.QMessageBox.warning(self, "警告", "選択年月のCSVが見つかりません。")
            return
        
        try:
            df, tcol = read_csv(path)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "エラー", f"CSVファイルの読み込みに失敗しました:\n{str(e)}")
            return
        
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
        
        # データプレビューの更新
        self.preview.clear()
        preview_rows = min(50, len(df_filtered))
        self.preview.setRowCount(preview_rows)
        self.preview.setColumnCount(len(df_filtered.columns))
        self.preview.setHorizontalHeaderLabels([str(c) for c in df_filtered.columns])
        
        for i in range(preview_rows):
            for j, c in enumerate(df_filtered.columns):
                val = df_filtered.iloc[i, j]
                if pd.isna(val):
                    text = ""
                elif isinstance(val, (int, np.integer)):
                    text = str(int(val))
                elif isinstance(val, (float, np.floating)):
                    text = f"{val:.2f}"
                elif pd.api.types.is_datetime64_any_dtype(type(val)):
                    text = str(val)
                else:
                    text = str(val)
                
                item = QTableWidgetItem(text)
                item.setTextAlignment(Qt.AlignCenter)
                self.preview.setItem(i, j, item)
        
        # カラム幅の自動調整
        self.preview.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeToContents)
        self.preview.horizontalHeader().setStretchLastSection(True)

        # グラフの描画
        self.canvas.ax.clear()
        self.canvas.ax.set_facecolor('#f8fafc')
        self.canvas.ax.tick_params(colors='#2d3748', labelsize=9)
        for spine in self.canvas.ax.spines.values():
            spine.set_color('#a0d2ff')
        
        # フィルタリングされたデータが空の場合
        if len(df_filtered) == 0:
            self.canvas.ax.text(0.5, 0.5, "選択された日付のデータがありません", 
                              ha="center", va="center", color="#0068B7", fontsize=14)
            self.canvas.draw()
            return
        
        # 数値カラムを取得（日時カラムは除外）
        num_cols = []
        for c in df_filtered.columns:
            if pd.api.types.is_numeric_dtype(df_filtered[c]):
                # カラム名をチェックして明らかに時刻でないものを選択
                col_str = str(c).lower()
                if not any(keyword in col_str for keyword in ['date', 'time', '時刻', '日時', '日付']):
                    num_cols.append(c)
        
        if not num_cols:
            self.canvas.ax.text(0.5, 0.5, "数値カラムが見つかりません", 
                              ha="center", va="center", color="#0068B7", fontsize=14)
        else:
            # 都市大カラーパレット
            colors = ['#0068B7', '#00A0E9', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899', '#14b8a6']
            
            # 最大5つのカラムまで表示（見やすさのため）
            display_cols = num_cols[:5] if len(num_cols) > 5 else num_cols
            
            if tcol and tcol in df_filtered.columns:
                # 日時カラムがある場合
                x_data = df_filtered[tcol]
                for idx, c in enumerate(display_cols):
                    y_data = df_filtered[c].dropna()
                    if len(y_data) > 0:
                        # x_dataとy_dataのインデックスを合わせる
                        valid_indices = df_filtered[c].notna()
                        x_plot = x_data[valid_indices]
                        y_plot = df_filtered[c][valid_indices]
                        
                        self.canvas.ax.plot(x_plot, y_plot, label=str(c), 
                                           color=colors[idx % len(colors)], linewidth=2, alpha=0.9)
                self.canvas.ax.set_xlabel(str(tcol), color='#2d3748', fontsize=11, fontweight='bold')
                # x軸のラベルを回転
                self.canvas.fig.autofmt_xdate(rotation=45)
            else:
                # 日時カラムがない場合はインデックスを使用
                x = range(len(df_filtered))
                for idx, c in enumerate(display_cols):
                    y_data = df_filtered[c].fillna(0)
                    self.canvas.ax.plot(x, y_data, label=str(c), 
                                       color=colors[idx % len(colors)], linewidth=2, alpha=0.9)
                self.canvas.ax.set_xlabel("データポイント", color='#2d3748', fontsize=11, fontweight='bold')
            
            self.canvas.ax.set_ylabel("値 (MW)", color='#2d3748', fontsize=11, fontweight='bold')
            
            # 凡例の設定
            legend = self.canvas.ax.legend(loc="upper left", facecolor='#ffffff', edgecolor='#a0d2ff', 
                                          labelcolor='#2d3748', fontsize=9, framealpha=0.95)
            
            # タイトルに日付情報を追加
            title_text = f"⚡ {AREA_INFO[code]['name']}エリア - {ym[:4]}年{ym[4:6]}月"
            if selected_date and selected_date != "all":
                date_obj = pd.to_datetime(selected_date)
                title_text += f" ({date_obj.strftime('%m月%d日')})"
            
            self.canvas.ax.set_title(title_text, color='#0068B7', fontsize=14, 
                                    fontweight='bold', pad=15)
            self.canvas.ax.grid(True, alpha=0.3, color='#cbd5e0', linestyle='--', linewidth=0.8)
            
            # Y軸のフォーマット（カンマ区切り）
            from matplotlib.ticker import FuncFormatter
            self.canvas.ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{int(x):,}'))
        
        self.canvas.fig.tight_layout(pad=2.0)
        self.canvas.draw()
        
        self.canvas.fig.tight_layout(pad=2.0)
        self.canvas.draw()

if __name__ == "__main__":
    app = QApplication([])
    w = MainWindow(); w.show()
    app.exec()
