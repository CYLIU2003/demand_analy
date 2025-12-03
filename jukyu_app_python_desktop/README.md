# PySide6 Demand Viewer

This desktop client offers a richer, highly interactive experience for analysing Japanese electricity supply-demand CSVs. It is built with PySide6 and Matplotlib, featuring multi-series plotting, fine-grained chart styling, and export utilities.

## Highlights

- Tabbed UI separating the availability dashboard, advanced analysis tools, generation comparison, and an AI lab.
- Multi-select column list with "select all" helpers to quickly build comparison charts.
- Customisable chart settings (title, axes labels, line width, figure size, font sizes, grid and legend toggles).
- **Display toggle checkboxes** for title, axis labels, and legend visibility.
- Integrated Matplotlib canvas with image export shortcut.
- Resilient CSV loader that tries Shift_JIS/CP932/UTF-8 encodings and auto-detects datetime columns.
- **Generation type comparison** with area-to-area and month-to-month analysis modes.
- **Statistical aggregation** (hourly/daily/weekly/monthly) with clipboard export for PowerPoint.
- Experimental transformer forecaster that turns CSVs into machine-learning ready datasets and produces multi-step predictions.
- **Weather data integration** with ARIMAX model forecasting.

## Requirements

- Python 3.10 or later (tested with CPython 3.11.9).
- PySide6 6.7.2, pandas 2.2.3, numpy 2.1.3, matplotlib 3.9.2, scikit-learn 1.5.2, torch 2.4.1. Install via the provided `requirements.txt`.
- ⚠️ **Note**: Python 3.14 is NOT compatible with PySide6 6.7.2.

## Installation & Launch

```powershell
# From jukyu_app_python_desktop/
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

### Auto-launch with Virtual Environment

Use the provided launcher script that automatically creates/detects a virtual environment:

```powershell
python run.py
```

Or double-click `電力需給分析ツール.bat` on Windows.

The virtual environment is created at `%USERPROFILE%\.demand_analy` to avoid path conflicts.

On macOS/Linux replace the activation command with `source .venv/bin/activate`.

### Data Placement

- Place CSVs under `jukyu_app_python_desktop/data/`.
- File naming pattern: `eria_jukyu_YYYYMM_AA.csv` where `AA` is the two-digit area code.
- Encoding: Shift_JIS/CP932 recommended; UTF-8 is accepted.
- The loader skips a leading unit description row if present.
- **Weather data** (optional): Place weather CSVs under `data/weather/` for ARIMAX forecasting.

### Usage Flow

1. Choose an area on the **メイン** tab and review the heatmap for missing months.
2. Switch to **詳細分析** and pick a month (`YYYYMM`) as well as an optional single day filter.
3. Select the generation categories you want to plot (use the "全選択" / "全解除" shortcuts for bulk actions).
4. Adjust chart settings (title, labels, grid, legend, line width, figure size, etc.).
5. Use **display toggle checkboxes** to show/hide title, axis labels, and legend for presentation-ready charts.
6. Click **📈 グラフ更新** to render the Matplotlib chart, then **💾 グラフを保存** to export as PNG.

### 📊 Generation Comparison

1. Open the **発電種別比較** tab.
2. Choose comparison mode: **エリア間比較** (compare areas) or **月間推移** (monthly trends).
3. Select areas using checkboxes and generation categories to include.
4. Toggle between **電力量** (absolute) and **構成比** (percentage) views.
5. Switch to **📋 数値データ** tab to view the raw numbers.
6. Click **📋 数値をコピー** to copy data to clipboard for Excel/PowerPoint.

### 📈 Statistical Aggregation

1. On the **メイン** tab, click **📊 統計分析に移行** to transfer your selection to the AI analysis tab.
2. Navigate to the **統計集計** sub-tab within AI分析.
3. Select aggregation period: 時間別/日別/週別/月別/全期間.
4. Click **集計実行** to calculate sum, mean, min, max statistics.
5. Click **📋 クリップボードにコピー** to export for presentations.

### 🤖 AI Lab Workflow

1. Open the **AI分析** tab and choose the area/year-month combination you wish to analyse.
2. Select a numeric column (e.g., 総需要, 系統出力) as the forecasting target and click **📚 データ要約** to review the latest context window.
3. Adjust the transformer hyperparameters (context length, prediction horizon, epochs, batch size, learning rate) to match your research design.
4. Click **🤖 Transformer学習＆予測** to train the lightweight PyTorch model and generate forward predictions; training/validation losses and the resulting trajectory are shown in the log and table.
5. Use **🌤️ ARIMAX予測** for weather-integrated forecasting (requires weather data in `data/weather/`).
6. Export the CSV or tweak parameters iteratively to compare baselines and transformer-driven forecasts.

## Packaging (Optional)

```powershell
pip install pyinstaller
pyinstaller -F -w main.py
```

Depending on the target OS, you may need additional PySide6 deployment steps (e.g., `--add-data` for Qt plugins).

## Troubleshooting

- **Missing fonts/garbled Japanese**: ensure MS Gothic, Yu Gothic, or Meiryo fonts are installed; the app falls back to DejaVu Sans.
- **CSV fails to load**: verify the filename matches the expected pattern and that the selected area corresponds to the area code in the file.
- **Slow plotting with many columns**: reduce the number of selected series or down-sample the CSV before loading.
- **Python 3.14 errors**: Downgrade to Python 3.11.x which is fully compatible with PySide6.

## Key Files

- `main.py`: application entry point containing the Qt widgets and plotting logic.
- `run.py`: launcher script with automatic virtual environment detection.
- `requirements.txt`: locked dependency versions.
- `data/`: sample CSVs for April 2024 through October 2025 in various areas.
- `data/weather/`: weather data for ARIMAX forecasting (optional).
