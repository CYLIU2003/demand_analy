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
import java.util.regex.*;
import com.opencsv.CSVReader;

public class Main extends Application {
    static class AreaInfo {
        String name; String url;
        AreaInfo(String name, String url) { this.name = name; this.url = url; }
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

    Map<String, Map<Integer, Map<Integer, Boolean>>> availability = new LinkedHashMap<>();
    List<Integer> years = new ArrayList<>();

    ComboBox<String> areaBox = new ComboBox<>();
    GridPane heat = new GridPane();
    ComboBox<String> ymBox = new ComboBox<>();
    TableView<List<String>> table = new TableView<>();
    LineChart<Number,Number> chart;

    @Override
    public void start(Stage stage) throws Exception {
        stage.setTitle("需給実績デスクトップアプリ (JavaFX)");

        HBox top = new HBox(10);
        top.setPadding(new Insets(10));
        top.setAlignment(Pos.CENTER_LEFT);
        for (String code : AREA_INFO.keySet()) areaBox.getItems().add("(" + code + ") " + AREA_INFO.get(code).name);
        areaBox.getSelectionModel().select("(10) 沖縄");
        Button openUrl = new Button("公式サイト"); openUrl.setOnAction(e -> openOfficial());
        Button openDir = new Button("フォルダを開く"); openDir.setOnAction(e -> openFolder());
        top.getChildren().addAll(new Label("エリア:"), areaBox, openDir, openUrl);

        heat.setHgap(4); heat.setVgap(4);
        heat.setPadding(new Insets(10));
        StackPane heatWrap = new StackPane(heat);
        heatWrap.setPadding(new Insets(0,10,10,10));
        heatWrap.setStyle("-fx-background-color: #0f1221;");

        HBox ctl = new HBox(8);
        ctl.setPadding(new Insets(10,10,0,10));
        ctl.setAlignment(Pos.CENTER_LEFT);
        Button loadBtn = new Button("読み込み"); loadBtn.setOnAction(e -> loadSelectedYM());
        ctl.getChildren().addAll(new Label("年月:"), ymBox, loadBtn);

        VBox left = new VBox(6, ctl, table);
        left.setPadding(new Insets(0,10,10,10));
        VBox.setVgrow(table, Priority.ALWAYS);

        NumberAxis xAxis = new NumberAxis();
        NumberAxis yAxis = new NumberAxis();
        chart = new LineChart<>(xAxis, yAxis); chart.setCreateSymbols(false); chart.setLegendVisible(true);

        SplitPane bottom = new SplitPane(left, chart); bottom.setDividerPositions(0.55);

        VBox root = new VBox(top, heatWrap, bottom);
        VBox.setVgrow(bottom, Priority.ALWAYS);

        Scene scene = new Scene(root, 1200, 740, Color.web("#0f1221"));
        scene.getRoot().setStyle("-fx-base: #151833; -fx-control-inner-background:#191d3f; -fx-text-fill: #e9ebff; -fx-font-size: 13px;");
        stage.setScene(scene); stage.show();

        refreshAvailability();
        areaBox.setOnAction(e -> onAreaChanged());
        onAreaChanged();
    }

    void refreshAvailability() throws IOException {
        availability.clear(); years.clear();
        if (!Files.exists(DATA_DIR)) Files.createDirectories(DATA_DIR);
        Set<Integer> ys = new TreeSet<>();
        for (String code : AREA_INFO.keySet()) availability.put(code, new LinkedHashMap<>());
        try (DirectoryStream<Path> ds = Files.newDirectoryStream(DATA_DIR)) {
            for (Path p : ds) {
                var m = FNAME.matcher(p.getFileName().toString());
                if (m.matches()) { ys.add(Integer.parseInt(m.group(1).substring(0,4))); }
            }
        }
        if (ys.isEmpty()) ys.add(LocalDateTime.now().getYear());
        years.addAll(ys);
        for (String code : AREA_INFO.keySet()) {
            for (int y : years) {
                Map<Integer, Boolean> m = new LinkedHashMap<>();
                for (int i=1;i<=12;i++) m.put(i,false);
                availability.get(code).put(y, m);
            }
        }
        try (DirectoryStream<Path> ds = Files.newDirectoryStream(DATA_DIR)) {
            for (Path p : ds) {
                var m = FNAME.matcher(p.getFileName().toString());
                if (m.matches()) {
                    String code = m.group(2);
                    int y = Integer.parseInt(m.group(1).substring(0,4));
                    int mo = Integer.parseInt(m.group(1).substring(4,6));
                    if (availability.containsKey(code) && availability.get(code).containsKey(y)) {
                        availability.get(code).get(y).put(mo, true);
                    }
                }
            }
        }
        drawHeatmap();
    }

    void drawHeatmap() {
        heat.getChildren().clear();
        heat.add(new Label("Year"), 0, 0);
        for (int m=1;m<=12;m++) {
            Label l = new Label(String.format("%02d", m));
            l.setTextFill(Color.web("#b6b9d6"));
            heat.add(l, m, 0);
        }
        String code = currentCode();
        int row=1;
        for (int y : years) {
            Label yl = new Label(String.valueOf(y)); yl.setTextFill(Color.web("#b6b9d6"));
            heat.add(yl, 0, row);
            for (int m=1;m<=12;m++) {
                boolean ok = availability.get(code).get(y).get(m);
                Region r = new Region();
                r.setMinSize(28,18); r.setPrefSize(36,24);
                r.setStyle(ok ? "-fx-background-color: rgba(34,197,94,0.4); -fx-border-color: rgba(34,197,94,1.0); -fx-border-radius:6; -fx-background-radius:6;" :
                                "-fx-background-color: rgba(239,68,68,0.35); -fx-border-color: rgba(239,68,68,1.0); -fx-border-radius:6; -fx-background-radius:6;");
                heat.add(r, m, row);
            }
            row++;
        }
        ymBox.getItems().clear();
        List<String> yms = new ArrayList<>();
        try (DirectoryStream<Path> ds = Files.newDirectoryStream(DATA_DIR)) {
            for (Path p : ds) {
                var m = FNAME.matcher(p.getFileName().toString());
                if (m.matches() && m.group(2).equals(currentCode())) yms.add(m.group(1));
            }
        } catch(Exception e){}
        Collections.sort(yms);
        for (String ym : yms) ymBox.getItems().add(ym.substring(0,4) + "年" + ym.substring(4,6) + "月 (" + ym + ")");
        if (!yms.isEmpty()) ymBox.getSelectionModel().select(ymBox.getItems().size()-1);
    }

    void onAreaChanged() { drawHeatmap(); }

    String currentCode() {
        String s = areaBox.getValue(); if (s==null || s.length()<4) return "10";
        return s.substring(1,3);
    }

    void openOfficial() {
        String code = currentCode();
        try { Desktop.getDesktop().browse(new java.net.URI(AREA_INFO.get(code).url)); } catch (Exception ignored) { }
    }

    void openFolder() {
        try { Desktop.getDesktop().open(new File("data")); } catch (Exception ignored) { }
    }

    void loadSelectedYM() {
        String item = ymBox.getValue();
        if (item==null) return;
        String ym = item.substring(item.indexOf("(")+1, item.indexOf(")"));
        String code = currentCode();
        Path file = DATA_DIR.resolve("eria_jukyu_" + ym + "_" + code + ".csv");
        if (!Files.exists(file)) { new Alert(Alert.AlertType.WARNING, "CSVが見つかりません。").showAndWait(); return; }
        try {
            List<String[]> rows;
            try (CSVReader r = new CSVReader(new FileReader(file.toFile()))) { rows = r.readAll(); }
            table.getColumns().clear(); table.getItems().clear();
            if (rows.isEmpty()) return;
            String[] headers = rows.get(0);
            for (int i=0;i<headers.length;i++) {
                final int idx=i;
                TableColumn<List<String>, String> col = new TableColumn<>(headers[i]);
                col.setCellValueFactory(data -> new javafx.beans.property.SimpleStringProperty(data.getValue().get(idx)));
                col.setPrefWidth(140);
                table.getColumns().add(col);
            }
            for (int i=1;i<Math.min(rows.size(), 31);i++) table.getItems().add(Arrays.asList(rows.get(i)));

            chart.getData().clear();
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
                chart.getData().add(s);
            }
        } catch (Exception ex) {
            new Alert(Alert.AlertType.ERROR, "読み込み中にエラー: " + ex.getMessage()).showAndWait();
        }
    }

    public static void main(String[] args) { launch(args); }
}
