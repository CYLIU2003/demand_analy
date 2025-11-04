# JavaFX Demand Viewer

This Gradle project provides a JavaFX 21 desktop client for exploring monthly electricity supply-demand CSV files published by Japanese utilities. The app emphasises quick availability checks through a heatmap and a table/line-chart preview of the selected dataset.

## Features

- Area selector with shortcuts to official data portals and the local `data/` folder.
- Availability heatmap (green = file present, red = missing) organised by year and month for the chosen area.
- Table preview (first 30 rows) rendered alongside a numeric line chart derived from the first numeric column encountered in the CSV.
- Automatic caching of directory scans to keep UI interactions responsive.

## Prerequisites

- JDK 17 or later (JavaFX 21 requires a modern JDK).
- Internet access if you plan to follow the "公式サイト" links to download fresh CSVs.

## Getting Started

```powershell
# From jukyu_app_javafx/
./gradlew.bat run
```

The Gradle wrapper downloads JavaFX modules on demand, so no manual JavaFX installation is required. On macOS/Linux you can run `./gradlew run` instead.

### Data Placement

- Drop CSV files under `jukyu_app_javafx/data/`.
- File naming pattern: `eria_jukyu_YYYYMM_AA.csv` (e.g. `eria_jukyu_202509_10.csv`).
- Encoding: Shift_JIS/CP932 preferred, UTF-8 supported.
- First row should contain headers; a leading unit-description row will be skipped automatically.

### Typical Workflow

1. Launch the app and choose an area (default: Okinawa).
2. Inspect the availability heatmap; green cells mark months present in `data/`.
3. Select a `YYYYMM` entry, then click **読み込み** to populate the table and chart.
4. Use **公式サイト** to fetch missing files or **フォルダを開く** to jump directly to the local data directory.

## Packaging

Produce an executable JAR:

```powershell
./gradlew.bat jar
```

For native installers (MSI, DMG, etc.), use `jpackage` with OS-specific options. Ensure you add the JavaFX modules via `--java-options "--add-modules javafx.controls"`.

## Troubleshooting

- **JavaFX modules missing**: Gradle automatically adds `javafx.controls`; if you run the app outside Gradle, remember to include the module path (`--module-path`) and modules.
- **CSV not found**: Confirm the filename matches `eria_jukyu_YYYYMM_AA.csv` and that the area code matches the currently selected area.
- **Encoding garbled**: Convert the CSV to Shift_JIS/CP932 when possible. JavaFX assumes the platform default encoding when reading via `FileReader`.

## Development Notes

- Main entry point: `src/main/java/app/Main.java`.
- The heatmap caches previously scanned files; call `refreshAvailability()` after programmatic file changes.
- OpenCSV 5.9 is used for robust CSV parsing.
