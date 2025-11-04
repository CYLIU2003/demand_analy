package app;package app;


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
import javafx.scene.Node;
import javafx.scene.Region;
import javafx.stage.FileChooser;
import javafx.stage.Stage;
import javafx.embed.swing.SwingFXUtils;
import javafx.scene.image.WritableImage;

import javax.imageio.ImageIO;
import java.awt.Desktop;
import java.io.*;
import java.nio.charset.Charset;
import java.nio.file.*;
import java.time.LocalDate;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.*;
import java.util.regex.*;
import java.util.stream.Collectors;
import com.opencsv.CSVReader;

public class Main extends Application {
    
    // エリア情報
    static class AreaInfo {
        String name; 
        String url;
        AreaInfo(String name, String url) { 
            this.name = name; 
            this.url = url; 
        }
    }
    
    static Map<String, AreaInfo> AREA_INFO = new LinkedHashMap<>();
    static {
        AREA_INFO.put("01", new AreaInfo("北海道","https://www.hepco.co.jp/network/con_service/public_document/supply_demand_results/index.html"));
        AREA_INFO.put("02", new AreaInfo("東北","https://setsuden.nw.tohoku-epco.co.jp/download.html"));
        AREA_INFO.put("03", new AreaInfo("東京","https://www.tepco.co.jp/forecast/html/area_jukyu-j.html"));
        AREA_INFO.put("04", new AreaInfo("中部","https://powergrid.chuden.co.jp/denkiyoho/#link02"));
        AREA_INFO.put("05", new AreaInfo("北陸","https://www.rikuden.co.jp/nw/denki-yoho/results_jyukyu.html"));
        AREA_INFO.put("06", new AreaInfo("関西","https://www.kansai-td.co.jp/denkiyoho/area-performance/index.html"));
        AREA_INFO.put("07", new AreaInfo("中国","https://www.energia.co.jp/nw/jukyuu/eria_jukyu.html"));
        AREA_INFO.put("08", new AreaInfo("四国","https://www.yonden.co.jp/nw/supply_demand/data_download.html"));
        AREA_INFO.put("09", new AreaInfo("九州","https://www.kyuden.co.jp/td_area_jukyu/jukyu.html"));
        AREA_INFO.put("10", new AreaInfo("沖縄","https://www.okiden.co.jp/business-support/service/supply-and-demand/"));
    }
    
    Pattern FNAME = Pattern.compile("^eria_jukyu_(\\d{6})_(\\d{2})\\.csv$");
    Path DATA_DIR = Paths.get("data");
    
    // データ可用性マップ
    Map<String, Map<Integer, Map<Integer, Boolean>>> availability = new LinkedHashMap<>();
    List<Integer> years = new ArrayList<>();
    
    // グラフ設定
    Map<String, Object> graphSettings = new HashMap<>();
    List<String> selectedColumns = new ArrayList<>();
    
    // UIコンポーネント
    TabPane tabPane;
    ComboBox<String> areaBox = new ComboBox<>();
    GridPane heatGrid = new GridPane();
    
    // 詳細ページ用
    ComboBox<String> ymBox = new ComboBox<>();
    ComboBox<String> dateBox = new ComboBox<>();
    VBox columnCheckboxContainer = new VBox(5);
    Map<String, CheckBox> columnCheckboxes = new LinkedHashMap<>();
    
    // グラフ設定UI
    TextField titleField = new TextField();
    TextField xlabelField = new TextField("時刻");
    TextField ylabelField = new TextField("電力 (MW)");
    Spinner<Double> linewidthSpinner = new Spinner<>(0.5, 10.0, 2.0, 0.5);
    Spinner<Integer> widthSpinner = new Spinner<>(4, 20, 12, 1);
    Spinner<Integer> heightSpinner = new Spinner<>(3, 15, 6, 1);
    Spinner<Integer> fontSizeSpinner = new Spinner<>(6, 24, 12, 1);
    Spinner<Integer> titleSizeSpinner = new Spinner<>(8, 32, 14, 1);
    CheckBox gridCheck = new CheckBox("グリッド表示");
    CheckBox legendCheck = new CheckBox("凡例表示");
    
    LineChart<Number, Number> chart;
    NumberAxis xAxis;
    NumberAxis yAxis;
    
    @Override
    public void start(Stage stage) throws Exception {
        stage.setTitle("⚡ 電力需給実績ビューア (JavaFX)");
        
        // グラフ設定の初期化
        initGraphSettings();
        
        // タブペイン作成
        tabPane = new TabPane();
        tabPane.setTabClosingPolicy(TabPane.TabClosingPolicy.UNAVAILABLE);
        
        // メインページタブ
        Tab mainTab = new Tab("📊 メイン");
        mainTab.setContent(createMainPage());
        
        // 詳細ページタブ
        Tab detailTab = new Tab("📈 詳細分析");
        detailTab.setContent(createDetailPage());
        
        tabPane.getTabs().addAll(mainTab, detailTab);
        
        // シーン作成
        Scene scene = new Scene(tabPane, 1400, 840);
        applyModernStyle(scene);
        
        stage.setScene(scene);
        stage.show();
        
        // データ読み込み
        refreshAvailability();
        areaBox.setOnAction(e -> onAreaChanged());
        onAreaChanged();
    }
    
    private void initGraphSettings() {
        graphSettings.put("title", "");
        graphSettings.put("xlabel", "時刻");
        graphSettings.put("ylabel", "電力 (MW)");
        graphSettings.put("linewidth", 2.0);
        graphSettings.put("grid", true);
        graphSettings.put("legend", true);
        graphSettings.put("figsize_w", 12);
        graphSettings.put("figsize_h", 6);
        graphSettings.put("font_size", 12);
        graphSettings.put("title_size", 14);
        
        gridCheck.setSelected(true);
        legendCheck.setSelected(true);
    }
    
    private VBox createMainPage() {
        VBox page = new VBox(15);
        page.setPadding(new Insets(20));
        page.setStyle("-fx-background-color: #f0f4f8;");
        
        // ヘッダー
        HBox header = new HBox(12);
        header.setAlignment(Pos.CENTER_LEFT);
        
        Label titleLabel = new Label("⚡ 電力需給実績データビューア");
        titleLabel.setStyle("-fx-font-size: 24px; -fx-font-weight: bold; -fx-text-fill: #0068B7;");
        
        header.getChildren().add(titleLabel);
        Region spacer = new Region();
        HBox.setHgrow(spacer, Priority.ALWAYS);
        header.getChildren().add(spacer);
        
        // コントロールエリア
        HBox controls = new HBox(10);
        controls.setAlignment(Pos.CENTER_LEFT);
        
        Label areaLabel = new Label("📍 エリア:");
        areaLabel.setStyle("-fx-font-weight: bold; -fx-font-size: 13px; -fx-text-fill: #0068B7;");
        
        for (String code : AREA_INFO.keySet()) {
            areaBox.getItems().add("(" + code + ") " + AREA_INFO.get(code).name);
        }
        areaBox.getSelectionModel().select(0);
        areaBox.setPrefWidth(180);
        
        Button urlBtn = new Button("🌐 公式サイト");
        urlBtn.setMinHeight(36);
        urlBtn.setOnAction(e -> openOfficial());
        
        Button folderBtn = new Button("📂 データフォルダ");
        folderBtn.setMinHeight(36);
        folderBtn.setOnAction(e -> openFolder());
        
        Button detailBtn = new Button("📈 詳細分析へ");
        detailBtn.setMinHeight(36);
        detailBtn.setStyle("-fx-background-color: linear-gradient(to bottom, #10b981, #059669); -fx-text-fill: white; -fx-font-weight: bold;");
        detailBtn.setOnAction(e -> tabPane.getSelectionModel().select(1));
        
        controls.getChildren().addAll(areaLabel, areaBox, folderBtn, urlBtn, detailBtn);
        
        // ヒートマップ
        Label heatmapLabel = new Label("📊 データ可用性マップ");
        heatmapLabel.setStyle("-fx-font-size: 16px; -fx-font-weight: bold; -fx-text-fill: #0068B7;");
        
        heatGrid.setHgap(4);
        heatGrid.setVgap(4);
        heatGrid.setPadding(new Insets(10));
        
        ScrollPane heatScroll = new ScrollPane(heatGrid);
        heatScroll.setFitToWidth(true);
        heatScroll.setStyle("-fx-background-color: #ffffff; -fx-border-color: #a0d2ff; -fx-border-width: 2; -fx-border-radius: 10;");
        VBox.setVgrow(heatScroll, Priority.ALWAYS);
        
        page.getChildren().addAll(header, controls, heatmapLabel, heatScroll);
        
        return page;
    }
    
    private VBox createDetailPage() {
        VBox page = new VBox(15);
        page.setPadding(new Insets(20));
        page.setStyle("-fx-background-color: #f0f4f8;");
        
        // ヘッダー
        HBox header = new HBox();
        Button backBtn = new Button("← メインに戻る");
        backBtn.setMinHeight(36);
        backBtn.setOnAction(e -> tabPane.getSelectionModel().select(0));
        header.getChildren().add(backBtn);
        Region spacer = new Region();
        HBox.setHgrow(spacer, Priority.ALWAYS);
        header.getChildren().add(spacer);
        
        // 3分割パネル
        HBox content = new HBox(10);
        HBox.setHgrow(content, Priority.ALWAYS);
        
        // 左パネル: データ選択
        VBox leftPanel = createDataSelectionPanel();
        leftPanel.setPrefWidth(350);
        
        // 中央パネル: グラフ設定
        VBox centerPanel = createGraphSettingsPanel();
        centerPanel.setPrefWidth(350);
        
        // 右パネル: グラフ表示
        VBox rightPanel = createGraphDisplayPanel();
        HBox.setHgrow(rightPanel, Priority.ALWAYS);
        
        content.getChildren().addAll(leftPanel, centerPanel, rightPanel);
        
        page.getChildren().addAll(header, content);
        VBox.setVgrow(content, Priority.ALWAYS);
        
        return page;
    }
    
    private VBox createDataSelectionPanel() {
        VBox panel = new VBox(10);
        panel.setPadding(new Insets(10));
        panel.setStyle("-fx-background-color: #ffffff; -fx-border-color: #a0d2ff; -fx-border-width: 2; -fx-border-radius: 10;");
        
        Label title = new Label("📅 データ選択");
        title.setStyle("-fx-font-size: 16px; -fx-font-weight: bold; -fx-text-fill: #0068B7;");
        
        // 年月選択
        VBox ymGroup = new VBox(5);
        Label ymLabel = new Label("年月");
        ymLabel.setStyle("-fx-font-weight: bold;");
        ymBox.setPrefWidth(300);
        ymBox.setOnAction(e -> onYmChange());
        ymGroup.getChildren().addAll(ymLabel, ymBox);
        
        // 日付選択
        VBox dateGroup = new VBox(5);
        Label dateLabel = new Label("日付");
        dateLabel.setStyle("-fx-font-weight: bold;");
        dateBox.getItems().add("全期間");
        dateBox.getSelectionModel().select(0);
        dateBox.setPrefWidth(300);
        dateBox.setOnAction(e -> updateColumnCheckboxes());
        dateGroup.getChildren().addAll(dateLabel, dateBox);
        
        // 発電方式選択
        VBox columnGroup = new VBox(5);
        Label columnLabel = new Label("表示する発電方式");
        columnLabel.setStyle("-fx-font-weight: bold;");
        
        ScrollPane columnScroll = new ScrollPane(columnCheckboxContainer);
        columnScroll.setFitToWidth(true);
        columnScroll.setPrefHeight(300);
        columnScroll.setStyle("-fx-background-color: #f8fafc;");
        
        HBox btnRow = new HBox(5);
        Button selectAllBtn = new Button("全選択");
        selectAllBtn.setOnAction(e -> selectAllColumns());
        Button deselectAllBtn = new Button("全解除");
        deselectAllBtn.setOnAction(e -> deselectAllColumns());
        btnRow.getChildren().addAll(selectAllBtn, deselectAllBtn);
        
        columnGroup.getChildren().addAll(columnLabel, columnScroll, btnRow);
        VBox.setVgrow(columnScroll, Priority.ALWAYS);
        
        // 可視化ボタン
        Button viewBtn = new Button("📈 グラフ更新");
        viewBtn.setPrefHeight(44);
        viewBtn.setPrefWidth(300);
        viewBtn.setStyle("-fx-background-color: linear-gradient(to bottom, #0068B7, #005291); -fx-text-fill: white; -fx-font-size: 14px; -fx-font-weight: bold;");
        viewBtn.setOnAction(e -> renderView());
        
        panel.getChildren().addAll(title, ymGroup, dateGroup, columnGroup, viewBtn);
        VBox.setVgrow(columnGroup, Priority.ALWAYS);
        
        return panel;
    }
    
    private VBox createGraphSettingsPanel() {
        VBox panel = new VBox(10);
        panel.setPadding(new Insets(10));
        panel.setStyle("-fx-background-color: #ffffff; -fx-border-color: #a0d2ff; -fx-border-width: 2; -fx-border-radius: 10;");
        
        Label title = new Label("⚙️ グラフ設定");
        title.setStyle("-fx-font-size: 16px; -fx-font-weight: bold; -fx-text-fill: #0068B7;");
        
        ScrollPane scroll = new ScrollPane();
        scroll.setFitToWidth(true);
        
        VBox settingsBox = new VBox(10);
        settingsBox.setPadding(new Insets(5));
        
        // タイトル設定
        VBox titleGroup = new VBox(5);
        Label titleLabel = new Label("タイトル");
        titleLabel.setStyle("-fx-font-weight: bold;");
        titleField.setPromptText("グラフタイトル（空欄で自動）");
        titleGroup.getChildren().addAll(titleLabel, titleField);
        
        // 軸ラベル設定
        VBox labelGroup = new VBox(5);
        Label labelLabel = new Label("軸ラベル");
        labelLabel.setStyle("-fx-font-weight: bold;");
        GridPane labelGrid = new GridPane();
        labelGrid.setHgap(5);
        labelGrid.setVgap(5);
        labelGrid.add(new Label("X軸:"), 0, 0);
        labelGrid.add(xlabelField, 1, 0);
        labelGrid.add(new Label("Y軸:"), 0, 1);
        labelGrid.add(ylabelField, 1, 1);
        xlabelField.setPrefWidth(200);
        ylabelField.setPrefWidth(200);
        labelGroup.getChildren().addAll(labelLabel, labelGrid);
        
        // 線の太さ
        VBox lineGroup = new VBox(5);
        Label lineLabel = new Label("線の太さ");
        lineLabel.setStyle("-fx-font-weight: bold;");
        linewidthSpinner.setEditable(true);
        linewidthSpinner.setPrefWidth(200);
        lineGroup.getChildren().addAll(lineLabel, linewidthSpinner);
        
        // グラフサイズ
        VBox sizeGroup = new VBox(5);
        Label sizeLabel = new Label("グラフサイズ");
        sizeLabel.setStyle("-fx-font-weight: bold;");
        GridPane sizeGrid = new GridPane();
        sizeGrid.setHgap(5);
        sizeGrid.setVgap(5);
        sizeGrid.add(new Label("幅:"), 0, 0);
        sizeGrid.add(widthSpinner, 1, 0);
        sizeGrid.add(new Label("高さ:"), 0, 1);
        sizeGrid.add(heightSpinner, 1, 1);
        widthSpinner.setEditable(true);
        heightSpinner.setEditable(true);
        widthSpinner.setPrefWidth(100);
        heightSpinner.setPrefWidth(100);
        sizeGroup.getChildren().addAll(sizeLabel, sizeGrid);
        
        // フォントサイズ
        VBox fontGroup = new VBox(5);
        Label fontLabel = new Label("フォントサイズ");
        fontLabel.setStyle("-fx-font-weight: bold;");
        GridPane fontGrid = new GridPane();
        fontGrid.setHgap(5);
        fontGrid.setVgap(5);
        fontGrid.add(new Label("一般:"), 0, 0);
        fontGrid.add(fontSizeSpinner, 1, 0);
        fontGrid.add(new Label("タイトル:"), 0, 1);
        fontGrid.add(titleSizeSpinner, 1, 1);
        fontSizeSpinner.setEditable(true);
        titleSizeSpinner.setEditable(true);
        fontSizeSpinner.setPrefWidth(100);
        titleSizeSpinner.setPrefWidth(100);
        fontGroup.getChildren().addAll(fontLabel, fontGrid);
        
        // 表示オプション
        VBox optionsGroup = new VBox(5);
        Label optionsLabel = new Label("表示オプション");
        optionsLabel.setStyle("-fx-font-weight: bold;");
        optionsGroup.getChildren().addAll(optionsLabel, gridCheck, legendCheck);
        
        // 保存ボタン
        Button saveBtn = new Button("💾 グラフを保存");
        saveBtn.setPrefWidth(300);
        saveBtn.setOnAction(e -> saveGraph());
        
        settingsBox.getChildren().addAll(
            titleGroup, labelGroup, lineGroup, 
            sizeGroup, fontGroup, optionsGroup, saveBtn
        );
        
        scroll.setContent(settingsBox);
        panel.getChildren().addAll(title, scroll);
        VBox.setVgrow(scroll, Priority.ALWAYS);
        
        return panel;
    }
    
    private VBox createGraphDisplayPanel() {
        VBox panel = new VBox(10);
        panel.setPadding(new Insets(10));
        panel.setStyle("-fx-background-color: #ffffff; -fx-border-color: #a0d2ff; -fx-border-width: 2; -fx-border-radius: 10;");
        
        Label title = new Label("📊 グラフ表示");
        title.setStyle("-fx-font-size: 16px; -fx-font-weight: bold; -fx-text-fill: #0068B7;");
        
        // チャート作成
        xAxis = new NumberAxis();
        yAxis = new NumberAxis();
        chart = new LineChart<>(xAxis, yAxis);
        chart.setCreateSymbols(false);
        chart.setLegendVisible(true);
        chart.setStyle("-fx-background-color: #f8fafc;");
        
        VBox.setVgrow(chart, Priority.ALWAYS);
        
        panel.getChildren().addAll(title, chart);
        
        return panel;
    }
    
    private void applyModernStyle(Scene scene) {
        String css = """
            .root {
                -fx-base: #f0f4f8;
                -fx-background: #f0f4f8;
            }
            .tab-pane {
                -fx-background-color: #ffffff;
            }
            .tab {
                -fx-background-color: #e6f2ff;
                -fx-text-fill: #0068B7;
                -fx-font-weight: bold;
                -fx-font-size: 14px;
            }
            .tab:selected {
                -fx-background-color: #ffffff;
            }
            .button {
                -fx-background-color: linear-gradient(to bottom, #0068B7, #005291);
                -fx-text-fill: white;
                -fx-font-weight: bold;
                -fx-background-radius: 8;
                -fx-padding: 8 16 8 16;
            }
            .button:hover {
                -fx-background-color: linear-gradient(to bottom, #0080e0, #0068B7);
            }
            .combo-box {
                -fx-background-color: white;
                -fx-border-color: #a0d2ff;
                -fx-border-width: 2;
                -fx-border-radius: 8;
            }
            .text-field, .spinner {
                -fx-background-color: white;
                -fx-border-color: #a0d2ff;
                -fx-border-width: 2;
                -fx-border-radius: 6;
            }
            .check-box {
                -fx-text-fill: #2d3748;
            }
            .label {
                -fx-text-fill: #2d3748;
            }
            """;
        scene.getRoot().setStyle(css);
    }
    
    void refreshAvailability() throws IOException {
        availability.clear();
        years.clear();
        if (!Files.exists(DATA_DIR)) Files.createDirectories(DATA_DIR);
        
        Set<Integer> ys = new TreeSet<>();
        for (String code : AREA_INFO.keySet()) {
            availability.put(code, new LinkedHashMap<>());
        }
        
        try (DirectoryStream<Path> ds = Files.newDirectoryStream(DATA_DIR)) {
            for (Path p : ds) {
                var m = FNAME.matcher(p.getFileName().toString());
                if (m.matches()) {
                    ys.add(Integer.parseInt(m.group(1).substring(0, 4)));
                }
            }
        }
        
        if (ys.isEmpty()) ys.add(LocalDateTime.now().getYear());
        years.addAll(ys);
        
        for (String code : AREA_INFO.keySet()) {
            for (int y : years) {
                Map<Integer, Boolean> m = new LinkedHashMap<>();
                for (int i = 1; i <= 12; i++) m.put(i, false);
                availability.get(code).put(y, m);
            }
        }
        
        try (DirectoryStream<Path> ds = Files.newDirectoryStream(DATA_DIR)) {
            for (Path p : ds) {
                var m = FNAME.matcher(p.getFileName().toString());
                if (m.matches()) {
                    String code = m.group(2);
                    int y = Integer.parseInt(m.group(1).substring(0, 4));
                    int mo = Integer.parseInt(m.group(1).substring(4, 6));
                    if (availability.containsKey(code) && availability.get(code).containsKey(y)) {
                        availability.get(code).get(y).put(mo, true);
                    }
                }
            }
        }
        
        drawHeatmap();
    }
    
    void drawHeatmap() {
        heatGrid.getChildren().clear();
        
        Label yearHeader = new Label("年");
        yearHeader.setStyle("-fx-text-fill: #0068B7; -fx-font-weight: bold;");
        heatGrid.add(yearHeader, 0, 0);
        
        for (int m = 1; m <= 12; m++) {
            Label l = new Label(String.format("%02d月", m));
            l.setStyle("-fx-text-fill: #0068B7; -fx-font-weight: bold;");
            heatGrid.add(l, m, 0);
        }
        
        String code = currentCode();
        int row = 1;
        for (int y : years) {
            Label yl = new Label(String.valueOf(y));
            yl.setStyle("-fx-text-fill: #0068B7; -fx-font-weight: bold;");
            heatGrid.add(yl, 0, row);
            
            for (int m = 1; m <= 12; m++) {
                boolean ok = availability.get(code).get(y).get(m);
                Region r = new Region();
                r.setMinSize(40, 25);
                r.setPrefSize(50, 30);
                
                if (ok) {
                    r.setStyle("-fx-background-color: #10b981; -fx-border-color: #059669; -fx-border-width: 2; -fx-border-radius: 6; -fx-background-radius: 6;");
                } else {
                    r.setStyle("-fx-background-color: #f87171; -fx-border-color: #ef4444; -fx-border-width: 2; -fx-border-radius: 6; -fx-background-radius: 6;");
                }
                
                heatGrid.add(r, m, row);
            }
            row++;
        }
        
        updateYmBox();
    }
    
    void updateYmBox() {
        ymBox.getItems().clear();
        List<String> yms = new ArrayList<>();
        
        try (DirectoryStream<Path> ds = Files.newDirectoryStream(DATA_DIR)) {
            for (Path p : ds) {
                var m = FNAME.matcher(p.getFileName().toString());
                if (m.matches() && m.group(2).equals(currentCode())) {
                    yms.add(m.group(1));
                }
            }
        } catch (Exception e) {
        }
        
        Collections.sort(yms);
        for (String ym : yms) {
            ymBox.getItems().add(ym.substring(0, 4) + "年" + ym.substring(4, 6) + "月 (" + ym + ")");
        }
        
        if (!yms.isEmpty()) {
            ymBox.getSelectionModel().select(ymBox.getItems().size() - 1);
        }
    }
    
    void onAreaChanged() {
        drawHeatmap();
    }
    
    void onYmChange() {
        String item = ymBox.getValue();
        if (item == null) return;
        
        String ym = item.substring(item.indexOf("(") + 1, item.indexOf(")"));
        String code = currentCode();
        Path file = DATA_DIR.resolve("eria_jukyu_" + ym + "_" + code + ".csv");
        
        if (!Files.exists(file)) return;
        
        // 日付リストを更新
        dateBox.getItems().clear();
        dateBox.getItems().add("全期間");
        
        try {
            List<String[]> rows = readCSV(file);
            if (rows.size() > 1) {
                // 日付カラムを探す
                String[] headers = rows.get(0);
                int dateColIdx = -1;
                for (int i = 0; i < headers.length; i++) {
                    if (headers[i].toUpperCase().contains("DATE") || headers[i].contains("日付")) {
                        dateColIdx = i;
                        break;
                    }
                }
                
                if (dateColIdx >= 0) {
                    Set<String> dates = new TreeSet<>();
                    for (int i = 1; i < rows.size(); i++) {
                        if (dateColIdx < rows.get(i).length) {
                            dates.add(rows.get(i)[dateColIdx]);
                        }
                    }
                    
                    for (String date : dates) {
                        dateBox.getItems().add(date);
                    }
                }
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
        
        dateBox.getSelectionModel().select(0);
        updateColumnCheckboxes();
    }
    
    void updateColumnCheckboxes() {
        columnCheckboxContainer.getChildren().clear();
        columnCheckboxes.clear();
        
        String item = ymBox.getValue();
        if (item == null) return;
        
        String ym = item.substring(item.indexOf("(") + 1, item.indexOf(")"));
        String code = currentCode();
        Path file = DATA_DIR.resolve("eria_jukyu_" + ym + "_" + code + ".csv");
        
        if (!Files.exists(file)) return;
        
        try {
            List<String[]> rows = readCSV(file);
            if (rows.isEmpty()) return;
            
            String[] headers = rows.get(0);
            
            for (int i = 0; i < headers.length; i++) {
                String col = headers[i];
                if (col.toLowerCase().contains("date") || col.toLowerCase().contains("time") 
                    || col.contains("日付") || col.contains("時刻") || col.contains("時間")) {
                    continue;
                }
                
                // 数値カラムかチェック
                if (rows.size() > 1) {
                    try {
                        Double.parseDouble(rows.get(1)[i].replace(",", ""));
                        CheckBox cb = new CheckBox(col);
                        cb.setSelected(true);
                        columnCheckboxes.put(col, cb);
                        columnCheckboxContainer.getChildren().add(cb);
                    } catch (Exception ignored) {
                    }
                }
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
    
    void selectAllColumns() {
        for (CheckBox cb : columnCheckboxes.values()) {
            cb.setSelected(true);
        }
    }
    
    void deselectAllColumns() {
        for (CheckBox cb : columnCheckboxes.values()) {
            cb.setSelected(false);
        }
    }
    
    void renderView() {
        String item = ymBox.getValue();
        if (item == null) {
            showAlert("情報", "年月を選択してください。");
            return;
        }
        
        String ym = item.substring(item.indexOf("(") + 1, item.indexOf(")"));
        String code = currentCode();
        Path file = DATA_DIR.resolve("eria_jukyu_" + ym + "_" + code + ".csv");
        
        if (!Files.exists(file)) {
            showAlert("警告", "選択年月のCSVが見つかりません。");
            return;
        }
        
        try {
            List<String[]> rows = readCSV(file);
            if (rows.isEmpty()) return;
            
            String[] headers = rows.get(0);
            
            // 選択された発電方式を取得
            selectedColumns.clear();
            for (Map.Entry<String, CheckBox> entry : columnCheckboxes.entrySet()) {
                if (entry.getValue().isSelected()) {
                    selectedColumns.add(entry.getKey());
                }
            }
            
            if (selectedColumns.isEmpty()) {
                showAlert("警告", "表示する発電方式を選択してください。");
                return;
            }
            
            // 日付フィルタリング
            String selectedDate = dateBox.getValue();
            List<String[]> filteredRows = new ArrayList<>();
            filteredRows.add(headers);
            
            if (selectedDate != null && !selectedDate.equals("全期間")) {
                int dateColIdx = -1;
                for (int i = 0; i < headers.length; i++) {
                    if (headers[i].toUpperCase().contains("DATE") || headers[i].contains("日付")) {
                        dateColIdx = i;
                        break;
                    }
                }
                
                if (dateColIdx >= 0) {
                    for (int i = 1; i < rows.size(); i++) {
                        if (dateColIdx < rows.get(i).length && rows.get(i)[dateColIdx].equals(selectedDate)) {
                            filteredRows.add(rows.get(i));
                        }
                    }
                }
            } else {
                filteredRows.addAll(rows.subList(1, rows.size()));
            }
            
            if (filteredRows.size() <= 1) {
                showAlert("警告", "選択された日付のデータがありません。");
                return;
            }
            
            // グラフ描画
            chart.getData().clear();
            chart.setTitle(getTitleText(code, ym, selectedDate));
            xAxis.setLabel(xlabelField.getText());
            yAxis.setLabel(ylabelField.getText());
            chart.setLegendVisible(legendCheck.isSelected());
            
            // 都市大カラーパレット
            String[] colors = {"#0068B7", "#00A0E9", "#10b981", "#f59e0b", "#ef4444", "#8b5cf6", "#ec4899", "#14b8a6"};
            
            int colorIndex = 0;
            for (String colName : selectedColumns) {
                int colIdx = -1;
                for (int i = 0; i < headers.length; i++) {
                    if (headers[i].equals(colName)) {
                        colIdx = i;
                        break;
                    }
                }
                
                if (colIdx >= 0) {
                    XYChart.Series<Number, Number> series = new XYChart.Series<>();
                    series.setName(colName);
                    
                    for (int i = 1; i < filteredRows.size(); i++) {
                        if (colIdx < filteredRows.get(i).length) {
                            try {
                                double value = Double.parseDouble(filteredRows.get(i)[colIdx].replace(",", ""));
                                series.getData().add(new XYChart.Data<>(i, value));
                            } catch (Exception ignored) {
                            }
                        }
                    }
                    
                    chart.getData().add(series);
                    
                    // 線の色を設定
                    String color = colors[colorIndex % colors.length];
                    series.getNode().setStyle("-fx-stroke: " + color + "; -fx-stroke-width: " + linewidthSpinner.getValue() + ";");
                    colorIndex++;
                }
            }
            
        } catch (Exception e) {
            showAlert("エラー", "グラフ描画中にエラー: " + e.getMessage());
            e.printStackTrace();
        }
    }
    
    String getTitleText(String code, String ym, String selectedDate) {
        if (!titleField.getText().isEmpty()) {
            return titleField.getText();
        }
        
        String title = "⚡ " + AREA_INFO.get(code).name + "エリア - " + 
                      ym.substring(0, 4) + "年" + ym.substring(4, 6) + "月";
        
        if (selectedDate != null && !selectedDate.equals("全期間")) {
            title += " (" + selectedDate + ")";
        }
        
        return title;
    }
    
    void saveGraph() {
        FileChooser fileChooser = new FileChooser();
        fileChooser.setTitle("グラフを保存");
        fileChooser.getExtensionFilters().addAll(
            new FileChooser.ExtensionFilter("PNG", "*.png"),
            new FileChooser.ExtensionFilter("すべてのファイル", "*.*")
        );
        
        File file = fileChooser.showSaveDialog(chart.getScene().getWindow());
        if (file != null) {
            try {
                WritableImage image = chart.snapshot(null, null);
                ImageIO.write(SwingFXUtils.fromFXImage(image, null), "png", file);
                showAlert("成功", "グラフを保存しました:\n" + file.getAbsolutePath());
            } catch (Exception e) {
                showAlert("エラー", "保存に失敗しました: " + e.getMessage());
            }
        }
    }
    
    String currentCode() {
        String s = areaBox.getValue();
        if (s == null || s.length() < 4) return "01";
        return s.substring(1, 3);
    }
    
    void openOfficial() {
        String code = currentCode();
        try {
            Desktop.getDesktop().browse(new java.net.URI(AREA_INFO.get(code).url));
        } catch (Exception ignored) {
        }
    }
    
    void openFolder() {
        try {
            Desktop.getDesktop().open(DATA_DIR.toFile());
        } catch (Exception ignored) {
        }
    }
    
    List<String[]> readCSV(Path file) throws Exception {
        // 複数のエンコーディングを試す
        String[] encodings = {"Shift_JIS", "MS932", "UTF-8"};
        
        for (String encoding : encodings) {
            try {
                List<String[]> rows;
                try (CSVReader reader = new CSVReader(new InputStreamReader(
                        new FileInputStream(file.toFile()), Charset.forName(encoding)))) {
                    rows = reader.readAll();
                }
                
                // 単位行をスキップ
                if (!rows.isEmpty() && rows.get(0).length > 0) {
                    String firstCell = rows.get(0)[0];
                    if (firstCell.contains("単位") || firstCell.contains("MW")) {
                        rows.remove(0);
                    }
                }
                
                return rows;
            } catch (Exception e) {
                // 次のエンコーディングを試す
            }
        }
        
        throw new Exception("ファイルを読み込めませんでした");
    }
    
    void showAlert(String title, String message) {
        Alert alert = new Alert(Alert.AlertType.INFORMATION);
        alert.setTitle(title);
        alert.setHeaderText(null);
        alert.setContentText(message);
        alert.showAndWait();
    }
    
    public static void main(String[] args) {
        launch(args);
    }
}
