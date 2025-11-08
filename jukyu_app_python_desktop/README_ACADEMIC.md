# 電力需給分析ツール - 学術研究版

Electric Power Supply-Demand Analysis Tool for Academic Research

## 概要

このツールは、日本の電力需給データの可視化・分析を行うデスクトップアプリケーションです。修士課程の研究に適した高度な統計分析・機械学習機能を搭載しています。

## 主な機能

### 1. データ可視化
- エリア別・月別データ可用性ヒートマップ
- 多系列時系列データのプロット
- カスタマイズ可能なグラフ設定
- 論文品質(Publication-ready)の図表生成

### 2. 統計分析機能
- **記述統計**: 平均、標準偏差、歪度、尖度など
- **外れ値検出**: IQR法、Z-score法
- **正規性検定**: Shapiro-Wilk検定
- **時間帯別・日別統計**: 時系列データの集約分析

### 3. 時系列分析機能
- **トレンド分析**: 移動平均によるトレンド抽出
- **季節性分解**: 加法・乗法モデルによる分解
- **自己相関分析**: ACF/PACF計算
- **定常性検定**: Augmented Dickey-Fuller検定
- **ピーク検出**: 需要ピークの自動検出
- **変化点検出**: 需要パターン変化の検出

### 4. 相関分析機能
- **相関行列**: Pearson/Spearman/Kendall相関
- **相互相関**: 時間遅れを考慮した相関分析
- **偏相関**: 制御変数を考慮した相関分析
- **移動相関**: 時間窓による相関の変化追跡

### 5. AIモデル (Transformer)
- **時系列予測**: Transformerベースのニューラルネットワーク
- **モデル訓練**: カスタマイズ可能なハイパーパラメータ
- **モデル評価**: MAE、RMSE、MAPE、R²スコア
- **モデル保存/読込**: 訓練済みモデルの再利用

### 6. 研究用エクスポート機能
- **データセットエクスポート**: CSV、Excel、Parquet形式
- **統計レポート**: テキスト、JSON、LaTeX形式
- **高品質図表**: PNG(300dpi)、PDF(ベクター)形式
- **モデル結果**: JSON形式での訓練履歴・評価指標

## プロジェクト構造

```
jukyu_app_python_desktop/
├── main.py                  # メインアプリケーション
├── requirements.txt         # 依存パッケージ
├── data/                   # データフォルダ
├── analysis/               # 分析モジュール
│   ├── statistics.py      # 統計分析
│   ├── timeseries.py      # 時系列分析
│   └── correlation.py     # 相関分析
├── models/                 # AIモデル
│   ├── model_interface.py # モデル基底クラス
│   └── transformer_forecaster.py  # Transformerモデル
├── exports/                # エクスポート機能
│   ├── academic_exporter.py  # データ・レポートエクスポート
│   └── figure_generator.py   # 論文品質図表生成
└── utils/                  # ユーティリティ
    └── data_processor.py  # データ前処理
```

## インストール

### 必要環境
- Python 3.10以上
- CUDA対応GPU (オプション、Transformerモデル高速化用)

### セットアップ

```bash
# 1. リポジトリのクローン
git clone https://github.com/CYLIU2003/demand_analy.git
cd demand_analy/jukyu_app_python_desktop

# 2. 仮想環境の作成
python -m venv .venv

# 3. 仮想環境の有効化
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

# 4. 依存パッケージのインストール
pip install -r requirements.txt
```

## 使用方法

### 基本的な使用

```bash
python main.py
```

### プログラムからの使用例

#### 統計分析

```python
from analysis.statistics import StatisticalAnalyzer
import pandas as pd

# データ読込
df = pd.read_csv('data/eria_jukyu_202504_01.csv', encoding='shift_jis')

# 統計分析器を初期化
analyzer = StatisticalAnalyzer(df)

# 記述統計を計算
stats = analyzer.descriptive_statistics(['需要電力', '太陽光発電'])
print(stats)

# レポート生成
report = analyzer.generate_report()
print(report)
```

#### 時系列分析

```python
from analysis.timeseries import TimeSeriesAnalyzer

# 時系列分析器を初期化
ts_analyzer = TimeSeriesAnalyzer(df, time_column='時刻')

# 季節性分解
decomposition = ts_analyzer.seasonal_decomposition('需要電力', period=24)

# 定常性検定
adf_result = ts_analyzer.stationarity_test('需要電力')
print(f"定常性: {adf_result['定常']}")

# 自己相関計算
acf_values = ts_analyzer.autocorrelation('需要電力', nlags=48)
```

#### Transformerモデルでの予測

```python
from models.transformer_forecaster import TransformerForecaster

# モデル設定
config = {
    'd_model': 64,
    'nhead': 4,
    'num_encoder_layers': 2,
    'learning_rate': 0.001
}

# モデル初期化
model = TransformerForecaster(config)

# データ準備
X, y = model.prepare_data(df, target_column='需要電力', sequence_length=24)

# 訓練/検証/テスト分割
train_size = int(0.7 * len(X))
val_size = int(0.15 * len(X))

X_train, y_train = X[:train_size], y[:train_size]
X_val, y_val = X[train_size:train_size+val_size], y[train_size:train_size+val_size]
X_test, y_test = X[train_size+val_size:], y[train_size+val_size:]

# モデル訓練
history = model.train(X_train, y_train, X_val, y_val, epochs=100, batch_size=32)

# 評価
metrics = model.evaluate(X_test, y_test)
print(f"MAE: {metrics['MAE']:.2f}, RMSE: {metrics['RMSE']:.2f}")

# 予測
predictions = model.predict(X_test)

# モデル保存
model.save_model('models/transformer_demand_forecast.pth')
```

#### 論文用図表の生成

```python
from exports.figure_generator import FigureGenerator

# 図表生成器を初期化
fig_gen = FigureGenerator(output_dir='figures')

# 時系列プロット
fig_gen.plot_time_series(
    data=df,
    time_column='時刻',
    value_columns=['需要電力', '太陽光発電', '風力発電'],
    title='電力需給の時系列変化',
    filename='demand_timeseries'
)

# 相関行列のヒートマップ
from analysis.correlation import CorrelationAnalyzer
corr_analyzer = CorrelationAnalyzer(df)
corr_matrix = corr_analyzer.correlation_matrix()

fig_gen.plot_correlation_matrix(
    correlation_matrix=corr_matrix,
    title='発電方式間の相関関係',
    filename='correlation_heatmap'
)

# 予測結果の比較
fig_gen.plot_prediction_comparison(
    time_index=test_dates,
    actual=y_test,
    predicted=predictions,
    title='需要予測の精度評価',
    filename='prediction_accuracy'
)
```

#### データ・レポートのエクスポート

```python
from exports.academic_exporter import AcademicExporter

# エクスポーター初期化
exporter = AcademicExporter(output_dir='research_outputs')

# データセットをエクスポート
exporter.export_dataset(df, filename='demand_data_2025', format='excel', include_metadata=True)

# 統計レポートをエクスポート
statistics = {
    '記述統計': stats,
    'ADF検定': adf_result,
    '相関分析': corr_matrix
}
exporter.export_statistics_report(statistics, filename='analysis_report', format='latex')

# モデル結果をエクスポート
exporter.export_model_results(
    model_name='Transformer',
    training_history=history,
    evaluation_metrics=metrics,
    config=config
)
```

## データ形式

### 入力データ
- ファイル名: `eria_jukyu_YYYYMM_AA.csv`
  - `YYYYMM`: 年月 (例: 202504)
  - `AA`: エリアコード (01-10)
- エンコーディング: Shift_JIS / UTF-8
- 必須カラム: 時刻を表すカラム + 数値データ

### エリアコード
- 01: 北海道
- 02: 東北
- 03: 東京
- 04: 中部
- 05: 北陸
- 06: 関西
- 07: 中国
- 08: 四国
- 09: 九州
- 10: 沖縄

## 研究での活用例

### 1. 需要予測精度の向上
- Transformerモデルによる高精度な短期予測
- 複数モデルの比較評価
- 時系列特徴量エンジニアリング

### 2. 再生可能エネルギー統合の分析
- 太陽光・風力発電の変動パターン分析
- 需要と再エネ出力の相関分析
- 季節性・トレンド分析

### 3. 需給バランスの最適化
- ピーク需要の予測と検出
- 需要変動の統計的特性の解明
- 異常パターンの検出

### 4. 地域間比較研究
- 複数エリアの需給パターン比較
- 地域特性の統計的分析
- エリア間相関分析

## 論文執筆のサポート

### 図表の品質
- 300dpiの高解像度PNG
- ベクター形式PDF (拡大しても劣化しない)
- LaTeX対応の表形式エクスポート

### 再現性の確保
- モデル設定のJSON出力
- 訓練履歴の完全記録
- データ前処理パイプラインの明示

### 統計的検定
- 正規性検定、定常性検定の実装
- 有意性検定結果の自動出力
- 信頼区間の可視化

## トラブルシューティング

### PyTorchのインストール
CUDA対応GPUがある場合:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

CPU版のみ:
```bash
pip install torch torchvision torchaudio
```

### メモリ不足
大規模データセットの場合:
- データのダウンサンプリング
- バッチサイズの削減
- モデルパラメータの削減 (d_model, num_layers)

### 日本語フォントの問題
フォントが正しく表示されない場合:
```python
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'MS Gothic'  # Windowsの場合
```

## 引用

この研究ツールを使用した場合は、以下のように引用してください:

```
@software{demand_analy,
  author = {Your Name},
  title = {Electric Power Supply-Demand Analysis Tool for Academic Research},
  year = {2025},
  url = {https://github.com/CYLIU2003/demand_analy}
}
```

## ライセンス

MIT License

## 連絡先

- GitHub: https://github.com/CYLIU2003/demand_analy
- Issues: https://github.com/CYLIU2003/demand_analy/issues

## 参考文献

### Transformer モデル
- Vaswani et al. (2017). "Attention Is All You Need". NeurIPS.

### 時系列分析
- Box, G. E. P., & Jenkins, G. M. (1976). Time Series Analysis: Forecasting and Control.

### 電力需給分析
- 各エリア送配電事業者の公開データに基づく

---

**更新日**: 2025年1月
**バージョン**: 2.0.0 (学術研究機能追加版)
