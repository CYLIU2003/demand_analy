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

## Tooling Summary

| Toolchain | Key Versions | Launch Command |
|-----------|--------------|----------------|
| JavaFX    | JDK 17+, JavaFX 21.0.4, OpenCSV 5.9 | `./gradlew run` |
| PySide6   | Python 3.10+, PySide6 6.7.2, pandas 2.2.3 | `python main.py` |

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
