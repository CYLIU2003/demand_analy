# Grid Demand Insights

Grid Demand Insights bundles two desktop applications - a JavaFX client and a PySide6 client - for exploring and visualising Japanese electricity supply-demand records.

- **JavaFX app** (`jukyu_app_javafx`): lightweight dashboard with an availability heatmap, table preview, and line chart for the first numeric measurement in each CSV.
- **PySide6 app** (`jukyu_app_python_desktop`): rich UI with multi-series plotting, column selection, and export controls.

## Repository Layout

```
.
├── data/                     # Shared sample CSVs (Shift_JIS/CP932 friendly)
├── jukyu_app_javafx/         # JavaFX implementation (Gradle project)
└── jukyu_app_python_desktop/ # PySide6 implementation (virtual-env friendly)
```

## Data Requirements

Both applications expect monthly CSVs placed under a `data/` directory located next to the app they run from.

- File name pattern: `eria_jukyu_YYYYMM_AA.csv`
  - `YYYYMM` is the year and month (e.g. `202509`).
   - `AA` is the two-digit area code (01-10). These codes map to Hokkaido, Tohoku, Tokyo, Chubu, Hokuriku, Kansai, Chugoku, Shikoku, Kyushu, and Okinawa, respectively.
- Encoding: Shift_JIS / CP932, although UTF-8 is also supported.
- Header: the first row should contain column labels. A leading row with unit descriptions is automatically skipped.

When new files are added, relaunch the apps or trigger the reload buttons to refresh the availability map.

## System Requirements

### Minimum Hardware Specifications

#### JavaFX Client
- **CPU**: Dual-core processor (2.0 GHz or higher)
- **RAM**: 4 GB
- **Storage**: 200 MB free space
- **Display**: 1280×720 resolution or higher

#### PySide6 Client (Basic Features)
- **CPU**: Dual-core processor (2.5 GHz or higher)
- **RAM**: 8 GB
- **Storage**: 500 MB free space (including dependencies)
- **Display**: 1920×1080 resolution recommended

#### PySide6 Client (with AI/ML Features)
- **CPU**: Quad-core processor (3.0 GHz or higher) recommended for Transformer models
- **RAM**: 16 GB recommended (minimum 8 GB)
  - ARIMA/Exponential Smoothing: 8 GB sufficient
  - Transformer models: 16 GB recommended
- **GPU**: Optional (CUDA-compatible GPU for faster Transformer training)
  - CPU-only mode: Training takes ~1 minute for 1,400 samples
  - GPU mode: Training significantly faster (requires CUDA-enabled PyTorch)
- **Storage**: 2 GB free space (including PyTorch ~1.5 GB)
- **Display**: 1920×1080 resolution or higher recommended

### Software Requirements

- **OS**: Windows 10/11, macOS 10.15+, or Linux (Ubuntu 20.04+)
- **JavaFX**: JDK 17 or later
- **Python**: 3.10, 3.11, or 3.12
- **Browser**: Modern web browser for opening data source links

## Tooling Summary

| Toolchain | Key Versions | Launch Command |
|-----------|--------------|----------------|
| JavaFX    | JDK 17+, JavaFX 21.0.4, OpenCSV 5.9 | `./gradlew run` |
| PySide6   | Python 3.10+, PySide6 6.7.2, pandas 2.2.3, scipy 1.14.1, statsmodels 0.14.4 | `python main.py` |
| PyTorch (Optional) | torch 2.4.1 (CPU), CUDA 11.8+ for GPU support | Enables Transformer forecasting |

## ✨ New Features (v2.0)

The PySide6 application now includes advanced **statistical analysis** and **AI-powered time series forecasting**:

### 📊 Statistical Analysis Tab
- **Descriptive Statistics**: Mean, median, standard deviation, skewness, kurtosis, CV
- **Distribution Analysis**: Histograms and box plots with outlier detection
- **Correlation Analysis**: Heatmap visualization of power generation correlations
- **Hourly Demand Patterns**: Time-of-day analysis with confidence intervals

### 🤖 AI Forecasting Tab
- **STL Decomposition**: Separate trend, seasonal, and residual components
- **ARIMA Forecasting**: Statistical time series prediction with model evaluation
- **Exponential Smoothing**: Holt-Winters method for short-term forecasting
- **Transformer Forecasting**: Deep learning-based predictions using attention mechanisms (requires PyTorch)
- **Weather Correlation Analysis**: Analyze relationships between weather conditions and power generation
- **Weather-Aware Forecasting (ARIMAX)**: Demand prediction using weather data as exogenous variables
- **Model Evaluation**: MAE, RMSE, MAPE metrics with residual analysis

For detailed documentation, see [FEATURES.md](FEATURES.md).

### 🌤️ Weather Data Integration (v2.1)

The application now supports integration with Japanese Meteorological Agency data:

**Supported Weather Variables:**
- Temperature (℃)
- Precipitation (mm)
- Sunshine duration (hours)
- Wind speed & direction (m/s)
- Solar radiation (MJ/m²)

**Weather Data Format:**
Place weather CSV files in `data/weather/` folder with naming pattern:
```
{city}_{YYYYMMDD}_{YYYYMMDD}.csv
```

**Example:**
```
tokyo_20250101_20250331.csv   # Tokyo area, Jan-Mar 2025
osaka_20250401_20250630.csv   # Osaka area, Apr-Jun 2025
```

**Area-City Mapping:**
| Area Code | Power Company | Weather Station |
|-----------|---------------|-----------------|
| 03 | Tokyo | tokyo |
| 04 | Chubu | nagoya |
| 06 | Kansai | osaka |

**Analysis Capabilities:**
- **Correlation Analysis**: Scatter plots showing relationships between weather and power generation
- **Weather-Aware Prediction**: ARIMAX models that use temperature, solar radiation, and wind data to improve forecast accuracy

### Performance Notes

- **Statistical methods** (STL, ARIMA, Exponential Smoothing) are lightweight and run instantly on typical datasets
- **Transformer models** require more computational resources:
  - Training time: ~30-60 seconds for 1,400 samples (30 epochs, CPU)
  - Context window: 48 hours (2 days) of historical data
  - GPU acceleration available with CUDA-enabled PyTorch

## Quick Start

### JavaFX Client

1. Open a terminal in `jukyu_app_javafx`.
2. Install dependencies (Gradle wrapper downloads JavaFX on demand):
   ```powershell
   .\gradlew.bat run
   ```
3. Select an area on the toolbar, inspect the data availability heatmap, then choose a month to preview the CSV and render the default line chart.

#### Packaging

```powershell
.\gradlew.bat jar
```

For installers, follow the `jpackage` instructions relevant to your platform.

### PySide6 Client

1. From `jukyu_app_python_desktop`, create and activate a virtual environment (recommended) and install dependencies:
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\activate
   pip install -r requirements.txt
   ```
2. Launch the desktop app:
   ```powershell
   python main.py
   ```
3. Use the tabs to navigate between the availability dashboard and the detailed multi-series plotting workspace. Select columns, adjust chart settings, and export plots as images.

## Development Notes

- Both apps share the same data naming rules, but keep separate `data/` folders for clarity.
- The Python client attempts multiple encodings when opening CSVs, making it resilient to Shift_JIS or UTF-8 sources.
- The JavaFX client caches directory scans so UI interactions remain responsive while switching areas.

## Contributing

1. Create a feature branch.
2. Update or add unit/UI tests where relevant (manual testing is common for these GUIs).
3. Open a pull request summarising the change, affected areas, and screenshots if UI updates were made.

## License

Specify the intended license for the repository (e.g., MIT, Apache 2.0) before publishing. Replace this section once a decision is made.
