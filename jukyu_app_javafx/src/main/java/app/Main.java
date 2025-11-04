package app;

import javafx.application.Application;
import javafx.geometry.Insets;
import javafx.geometry.Pos;
import javafx.scene.Scene;
import javafx.scene.chart.LineChart;
import javafx.scene.chart.NumberAxis;
import javafx.scene.chart.XYChart;
import javafx.scene.control.*;
import javafx.scene.layout.*;
import javafx.scene.paint.Color;
import javafx.stage.Stage;

import java.awt.Desktop;
import java.io.*;
import java.nio.file.*;
import java.time.LocalDateTime;
import java.util.*;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Collectors;
import java.net.URISyntaxException;

import com.opencsv.CSVReader;
import com.opencsv.exceptions.CsvException;

public class Main extends Application {

    /**
     * Immutable metadata for each service area.
     */
    private static final class AreaInfo {
        final String name;
        final String url;

        AreaInfo(String name, String url) {
            this.name = name;
            this.url = url;
        }
    }

    private static final Map<String, AreaInfo> AREA_INFO;

    static {
        Map<String, AreaInfo> areaInfo = new LinkedHashMap<>();
        areaInfo.put("01", new AreaInfo("北海道", "https://www.hepco.co.jp/network/con_service/public_document/supply_demand_results/index.html"));
        areaInfo.put("02", new AreaInfo("東北", "https://setsuden.nw.tohoku-epco.co.jp/download.html"));
        areaInfo.put("03", new AreaInfo("東京", "https://www.tepco.co.jp/forecast/html/area_jukyu-j.html"));
        areaInfo.put("04", new AreaInfo("中部", "https://powergrid.chuden.co.jp/denkiyoho/#link02"));
        areaInfo.put("05", new AreaInfo("北陸", "https://www.rikuden.co.jp/nw/denki-yoho/results_jyukyu.html"));
        areaInfo.put("06", new AreaInfo("関西", "https://www.kansai-td.co.jp/denkiyoho/area-performance/index.html"));
        areaInfo.put("07", new AreaInfo("中国", "https://www.energia.co.jp/nw/jukyuu/eria_jukyu.html"));
        areaInfo.put("08", new AreaInfo("四国", "https://www.yonden.co.jp/nw/supply_demand/data_download.html"));
        areaInfo.put("09", new AreaInfo("九州", "https://www.kyuden.co.jp/td_area_jukyu/jukyu.html"));
        areaInfo.put("10", new AreaInfo("沖縄", "https://www.okiden.co.jp/business-support/service/supply-and-demand/"));
        AREA_INFO = Collections.unmodifiableMap(areaInfo);
    }

    private static final Pattern CSV_FILE_PATTERN = Pattern.compile("^eria_jukyu_(\\d{6})_(\\d{2})\\.csv$");
    private static final Path DATA_DIRECTORY = Paths.get("data");

    /**
     * Nested availability map: area -> year -> month -> presence of CSV file.
     */
    private final Map<String, Map<Integer, Map<Integer, Boolean>>> fileAvailability = new LinkedHashMap<>();
    private final List<Integer> availableYears = new ArrayList<>();
    private List<Path> discoveredCsvFiles = new ArrayList<>();

    private final ComboBox<String> areaSelector = new ComboBox<>();
    private final GridPane availabilityGrid = new GridPane();
    private final ComboBox<String> yearMonthSelector = new ComboBox<>();
    private final TableView<List<String>> previewTable = new TableView<>();
    private LineChart<Number, Number> demandChart;

    @Override
    public void start(Stage stage) throws Exception {
        stage.setTitle("需給実績デスクトップアプリ (JavaFX)");

        HBox top = new HBox(10);
        top.setPadding(new Insets(10));
        top.setAlignment(Pos.CENTER_LEFT);
        for (String code : AREA_INFO.keySet()) {
            areaSelector.getItems().add("(" + code + ") " + AREA_INFO.get(code).name);
        }
        areaSelector.getSelectionModel().select("(10) 沖縄");
        Button openUrl = new Button("公式サイト");
        openUrl.setOnAction(e -> openOfficial());
        Button openDir = new Button("フォルダを開く");
        openDir.setOnAction(e -> openFolder());
        top.getChildren().addAll(new Label("エリア:"), areaSelector, openDir, openUrl);

        availabilityGrid.setHgap(4);
        availabilityGrid.setVgap(4);
        availabilityGrid.setPadding(new Insets(10));
        StackPane heatWrap = new StackPane(availabilityGrid);
        heatWrap.setPadding(new Insets(0,10,10,10));
        heatWrap.setStyle("-fx-background-color: #0f1221;");

        HBox ctl = new HBox(8);
        ctl.setPadding(new Insets(10,10,0,10));
        ctl.setAlignment(Pos.CENTER_LEFT);
        Button loadBtn = new Button("読み込み");
        loadBtn.setOnAction(e -> loadSelectedYearMonth());
        ctl.getChildren().addAll(new Label("年月:"), yearMonthSelector, loadBtn);

        VBox left = new VBox(6, ctl, previewTable);
        left.setPadding(new Insets(0,10,10,10));
        VBox.setVgrow(previewTable, Priority.ALWAYS);

        NumberAxis xAxis = new NumberAxis();
        NumberAxis yAxis = new NumberAxis();
        demandChart = new LineChart<>(xAxis, yAxis);
        demandChart.setCreateSymbols(false);
        demandChart.setLegendVisible(true);

        SplitPane bottom = new SplitPane(left, demandChart);
        bottom.setDividerPositions(0.55);

        VBox root = new VBox(top, heatWrap, bottom);
        VBox.setVgrow(bottom, Priority.ALWAYS);

        Scene scene = new Scene(root, 1200, 740, Color.web("#0f1221"));
        scene.getRoot().setStyle("-fx-base: #151833; -fx-control-inner-background:#191d3f; -fx-text-fill: #e9ebff; -fx-font-size: 13px;");
        stage.setScene(scene); stage.show();

        refreshAvailability();
        areaSelector.setOnAction(e -> onAreaChanged());
        onAreaChanged();
    }

    void refreshAvailability() throws IOException {
        fileAvailability.clear();
        availableYears.clear();

        if (!Files.exists(DATA_DIRECTORY)) {
            Files.createDirectories(DATA_DIRECTORY);
        }

        discoveredCsvFiles = listCsvFiles();

        Set<Integer> collectedYears = discoveredCsvFiles.stream()
            .map(path -> CSV_FILE_PATTERN.matcher(path.getFileName().toString()))
            .filter(Matcher::matches)
            .map(matcher -> Integer.parseInt(matcher.group(1).substring(0, 4)))
            .collect(Collectors.toCollection(TreeSet::new));

        if (collectedYears.isEmpty()) {
            collectedYears.add(LocalDateTime.now().getYear());
        }

        availableYears.addAll(collectedYears);

        for (String code : AREA_INFO.keySet()) {
            Map<Integer, Map<Integer, Boolean>> yearlyAvailability = new LinkedHashMap<>();
            for (int year : availableYears) {
                Map<Integer, Boolean> monthlyAvailability = new LinkedHashMap<>();
                for (int month = 1; month <= 12; month++) {
                    monthlyAvailability.put(month, false);
                }
                yearlyAvailability.put(year, monthlyAvailability);
            }
            fileAvailability.put(code, yearlyAvailability);
        }

        for (Path csvFile : discoveredCsvFiles) {
            Matcher matcher = CSV_FILE_PATTERN.matcher(csvFile.getFileName().toString());
            if (matcher.matches()) {
                String code = matcher.group(2);
                int year = Integer.parseInt(matcher.group(1).substring(0, 4));
                int month = Integer.parseInt(matcher.group(1).substring(4, 6));
                Map<Integer, Map<Integer, Boolean>> yearlyAvailability = fileAvailability.get(code);
                if (yearlyAvailability != null) {
                    Map<Integer, Boolean> monthlyAvailability = yearlyAvailability.get(year);
                    if (monthlyAvailability != null && monthlyAvailability.containsKey(month)) {
                        monthlyAvailability.put(month, true);
                    }
                }
            }
        }

        drawAvailabilityHeatmap();
    }

    private List<Path> listCsvFiles() throws IOException {
        try (var stream = Files.list(DATA_DIRECTORY)) {
            return stream
                .filter(this::isCsvFile)
                .sorted()
                .collect(Collectors.toList());
        }
    }

    private boolean isCsvFile(Path path) {
        return CSV_FILE_PATTERN.matcher(path.getFileName().toString()).matches();
    }

    void drawAvailabilityHeatmap() {
        availabilityGrid.getChildren().clear();
        availabilityGrid.add(new Label("Year"), 0, 0);
        for (int m=1;m<=12;m++) {
            Label l = new Label(String.format("%02d", m));
            l.setTextFill(Color.web("#b6b9d6"));
            availabilityGrid.add(l, m, 0);
        }
        String code = currentCode();
        int row=1;
        for (int y : availableYears) {
            Label yl = new Label(String.valueOf(y));
            yl.setTextFill(Color.web("#b6b9d6"));
            availabilityGrid.add(yl, 0, row);
            for (int m=1;m<=12;m++) {
                boolean ok = fileAvailability.get(code).get(y).get(m);
                Region r = new Region();
                r.setMinSize(28,18);
                r.setPrefSize(36,24);
                r.setStyle(ok ? "-fx-background-color: rgba(34,197,94,0.4); -fx-border-color: rgba(34,197,94,1.0); -fx-border-radius:6; -fx-background-radius:6;" :
                                "-fx-background-color: rgba(239,68,68,0.35); -fx-border-color: rgba(239,68,68,1.0); -fx-border-radius:6; -fx-background-radius:6;");
                availabilityGrid.add(r, m, row);
            }
            row++;
        }
        yearMonthSelector.getItems().clear();
        List<String> yearMonthOptions = discoveredCsvFiles.stream()
            .map(path -> CSV_FILE_PATTERN.matcher(path.getFileName().toString()))
            .filter(Matcher::matches)
            .filter(matcher -> matcher.group(2).equals(currentCode()))
            .map(matcher -> matcher.group(1))
            .sorted()
            .collect(Collectors.toList());
        for (String ym : yearMonthOptions) {
            yearMonthSelector.getItems().add(ym.substring(0,4) + "年" + ym.substring(4,6) + "月 (" + ym + ")");
        }
        if (!yearMonthOptions.isEmpty()) {
            yearMonthSelector.getSelectionModel().select(yearMonthSelector.getItems().size()-1);
        }
    }

    void onAreaChanged() {
        drawAvailabilityHeatmap();
    }

    String currentCode() {
        String s = areaSelector.getValue();
        if (s==null || s.length()<4) return "10";
        return s.substring(1,3);
    }

    void openOfficial() {
        String code = currentCode();
        try {
            Desktop.getDesktop().browse(new java.net.URI(AREA_INFO.get(code).url));
        } catch (IOException | URISyntaxException ex) {
            System.err.println("Failed to open official site: " + ex.getMessage());
        }
    }

    void openFolder() {
        try {
            Desktop.getDesktop().open(DATA_DIRECTORY.toFile());
        } catch (IOException ex) {
            System.err.println("Failed to open data directory: " + ex.getMessage());
        }
    }

    void loadSelectedYearMonth() {
        String item = yearMonthSelector.getValue();
        if (item==null) return;
        String ym = item.substring(item.indexOf("(")+1, item.indexOf(")"));
        String code = currentCode();
        Path file = DATA_DIRECTORY.resolve("eria_jukyu_" + ym + "_" + code + ".csv");
        if (!Files.exists(file)) { new Alert(Alert.AlertType.WARNING, "CSVが見つかりません。").showAndWait(); return; }
        try {
            List<String[]> rows;
            try (CSVReader r = new CSVReader(new FileReader(file.toFile()))) { rows = r.readAll(); }
            previewTable.getColumns().clear(); previewTable.getItems().clear();
            if (rows.isEmpty()) return;
            String[] headers = rows.get(0);
            for (int i=0;i<headers.length;i++) {
                final int idx=i;
                TableColumn<List<String>, String> col = new TableColumn<>(headers[i]);
                col.setCellValueFactory(data -> new javafx.beans.property.SimpleStringProperty(data.getValue().get(idx)));
                col.setPrefWidth(140);
                previewTable.getColumns().add(col);
            }
            for (int i=1;i<Math.min(rows.size(), 31);i++) previewTable.getItems().add(Arrays.asList(rows.get(i)));

            demandChart.getData().clear();
            int firstNumIdx=-1;
            if (rows.size()>1) {
                for (int c=0;c<headers.length;c++) {
                    try { Double.parseDouble(rows.get(1)[c].replace(",","")); firstNumIdx=c; break; }
                    catch(Exception ignored) { }
                }
            }
            if (firstNumIdx>=0) {
                XYChart.Series<Number,Number> s = new XYChart.Series<>();
                s.setName(headers[firstNumIdx]);
                int x=1;
                for (int i=1;i<rows.size();i++) {
                    try {
                        double v = Double.parseDouble(rows.get(i)[firstNumIdx].replace(",",""));
                        s.getData().add(new XYChart.Data<>(x++, v));
                    } catch(Exception ignored) { }
                }
                demandChart.getData().add(s);
            }
        } catch (IOException | CsvException ex) {
            new Alert(Alert.AlertType.ERROR, "読み込み中にエラー: " + ex.getMessage()).showAndWait();
        }
    }

    public static void main(String[] args) { launch(args); }
}
