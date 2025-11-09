# PySide6 Demand Viewer

This desktop client offers a richer, highly interactive experience for analysing Japanese electricity supply-demand CSVs. It is built with PySide6 and Matplotlib, featuring multi-series plotting, fine-grained chart styling, and export utilities.

## Highlights

- Tabbed UI separating the availability dashboard from advanced analysis tools.
- Multi-select column list with "select all" helpers to quickly build comparison charts.
- Customisable chart settings (title, axes labels, line width, figure size, font sizes, grid and legend toggles).
- Integrated Matplotlib canvas with image export shortcut.
- Resilient CSV loader that tries Shift_JIS/CP932/UTF-8 encodings and auto-detects datetime columns.
- "🤖 AI予測" tab that embeds a lightweight Transformer forecaster for academic experiments (PyTorch optional).

## Requirements

- Python 3.10 or later (tested with CPython).
- PySide6 6.7.2, pandas 2.2.3, numpy 2.1.3, matplotlib 3.9.2. Install via the provided `requirements.txt`.

## Installation & Launch

```powershell
# From jukyu_app_python_desktop/
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

On macOS/Linux replace the activation command with `source .venv/bin/activate`.

### Data Placement

- Place CSVs under `jukyu_app_python_desktop/data/`.
- File naming pattern: `eria_jukyu_YYYYMM_AA.csv` where `AA` is the two-digit area code.
- Encoding: Shift_JIS/CP932 recommended; UTF-8 is accepted.
- The loader skips a leading unit description row if present.

### Usage Flow

1. Choose an area on the **メイン** tab and review the heatmap for missing months.
2. Switch to **詳細分析** and pick a month (`YYYYMM`) as well as an optional single day filter.
3. Select the generation categories you want to plot (use the "全選択" / "全解除" shortcuts for bulk actions).
4. Adjust chart settings (title, labels, grid, legend, line width, figure size, etc.).
5. Click **📈 グラフ更新** to render the Matplotlib chart, then **💾 グラフを保存** to export as PNG.

### AI Forecasting Lab (Optional)

- Install PyTorch (CPU build is sufficient) to enable the **🤖 AI予測** tab:

  ```bash
  pip install torch --index-url https://download.pytorch.org/whl/cpu
  ```

- The tab lets you:
  - Select an area/month CSV and choose any numeric column as the forecasting target.
  - Tune Transformer hyper-parameters (input window, forecast horizon, epochs, learning rate).
  - Train a compact encoder-only Transformer on the fly and compare predictions vs. actual demand.
  - Export the metrics (MAE, RMSE, MAPE) displayed in the status banner for research documentation.

- Data quality tips:
  - Ensure the chosen target column has at least `(input_window + forecast_horizon)` non-null records.
  - When the CSV includes a timestamp column, it is used to align the training context and forecast timeline.
  - Missing timestamps default to an hourly synthetic index for experimentation.

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

## Key Files

- `main.py`: application entry point containing the Qt widgets and plotting logic.
- `requirements.txt`: locked dependency versions.
- `data/`: sample CSVs for April 2024 through October 2025 in various areas.
