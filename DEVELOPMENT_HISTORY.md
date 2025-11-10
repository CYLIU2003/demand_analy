# 開発履歴 - 電力需要分析ツール改善プロジェクト

## プロジェクト概要
ブランチ: `power_demand_ai`  
期間: 2025年11月9日 - 2025年11月10日  
目的: 電力需要分析アプリケーションの詳細分析タブとAI予測機能の抜本的な改善

## 開発の経緯

### 初期要望（2025年11月9日）
ユーザーからの要望:
1. **詳細分析タブの改善**: 「詳細分析のタブに入っているものがすごく直観に反するので抜本的に直してほしい」
2. **AI予測の学術的強化**: 「AI予測のほうももう少し学術的に意味のあることができるようにしたい」

### 実装内容

#### 1. 詳細分析タブの完全リニューアル
5つのサブタブを持つ統計分析インターフェースを実装:

**タブ1: 基本統計**
- 平均値、中央値、標準偏差
- 最小値、最大値、四分位数
- 歪度、尖度
- データ概要テーブル

**タブ2: 時系列分析**
- 時系列プロット（トレンドライン付き）
- 移動平均（7日、30日）
- 変化率の可視化

**タブ3: 分布分析**
- ヒストグラム（正規分布曲線オーバーレイ）
- Q-Qプロット（正規性検定）
- ボックスプロット（外れ値検出）

**タブ4: 相関分析**
- 散布図行列
- 相関係数ヒートマップ
- ペアワイズ相関

**タブ5: 時間帯分析**
- 時間帯別平均需要
- 曜日別パターン
- 時間帯ヒートマップ

#### 2. AI予測タブの学術的機能追加

**実装した時系列予測手法:**

1. **STL時系列分解 (Seasonal and Trend decomposition using Loess)**
   - トレンド成分の抽出
   - 24時間周期の季節性成分
   - 残差成分の分析

2. **ARIMA予測 (AutoRegressive Integrated Moving Average)**
   - 自動次数選定 (p,d,q)
   - 訓練/テスト分割
   - MAE, RMSE, MAPE評価指標

3. **指数平滑法 (Exponential Smoothing)**
   - 季節性対応（加法・乗法モデル）
   - Holt-Winters法
   - 予測精度評価

4. **残差分析**
   - 残差の時系列プロット
   - 残差ヒストグラム
   - Q-Qプロット
   - ACF (自己相関関数)

**評価指標:**
- MAE (Mean Absolute Error): 平均絶対誤差
- RMSE (Root Mean Squared Error): 二乗平均平方根誤差
- MAPE (Mean Absolute Percentage Error): 平均絶対パーセント誤差

### 技術的な課題と解決

#### 課題1: AIタブが無効化される問題
**問題**: PyTorchがインストールされていない環境でAIタブ全体が無効化された

**解決策**:
```python
# ml/__init__.py で条件付きインポート
try:
    import torch
    TRANSFORMER_AVAILABLE = True
except ImportError:
    TRANSFORMER_AVAILABLE = False

# TransformerボタンだけをTRANSFORMER_AVAILABLEフラグで制御
# STL、ARIMA、指数平滑法は常に利用可能
```

#### 課題2: STL時系列分解エラー
**エラー**: `Unable to determine period from endog`

**原因分析**:
1. 最初: `seasonal`パラメータに周期の長さを直接指定していた（誤り）
2. 次: `series.dropna()`で欠損値を削除したため、時系列の連続性が失われた

**解決策（2段階）**:

**第1段階**: 欠損値の補間処理
```python
def get_ai_data_series(self, interpolate=False):
    """
    interpolate=True: 線形補間で欠損値を埋める（STL用）
    interpolate=False: 欠損値を削除（ARIMA/指数平滑法用）
    """
    if interpolate and missing_count > 0:
        series_clean = series.interpolate(method='linear', limit_direction='both')
        series_clean = series_clean.ffill().bfill()
```

**第2段階**: STLパラメータの正しい指定
```python
# 誤った実装
stl = STL(series.values, seasonal=period, robust=True)

# 正しい実装
stl = STL(series.values, period=24, seasonal=7, robust=True)
# period: 周期の長さ（24時間）
# seasonal: 季節成分のスムージング幅（奇数、最低7）
```

#### 課題3: 依存ライブラリ不足
**問題**: `statsmodels`と`scikit-learn`がインストールされていない

**解決策**:
```python
# メソッドの最初でインポートエラーを検出
try:
    from statsmodels.tsa.arima.model import ARIMA
    from sklearn.metrics import mean_absolute_error
except ImportError as e:
    self.append_ai_log(f"ライブラリのインポートエラー: {str(e)}")
    QtWidgets.QMessageBox.critical(
        self, "エラー", 
        f"必要なライブラリがインストールされていません:\n{str(e)}\n\n"
        "pip install statsmodels scikit-learn"
    )
    return
```

```bash
# Python 3.11環境にインストール
pip install statsmodels==0.14.1 scikit-learn==1.5.2 scipy==1.11.4
```

## 依存ライブラリ

### 必須
- Python 3.11+
- PySide6==6.7.2 (Qt GUI)
- pandas==2.2.3 (データ処理)
- numpy==2.1.3 (数値計算)
- matplotlib==3.9.2 (可視化)
- scipy==1.11.4 (統計関数)
- statsmodels==0.14.1 (時系列分析)
- scikit-learn==1.5.2 (機械学習評価指標)

### 任意
- torch==2.4.1 (Transformer予測用、必須ではない)

## Git履歴

### コミット一覧

**1. 初回実装** (コミットハッシュ未記録)
- 詳細分析タブの5サブタブ実装
- AI予測タブの時系列分析機能追加
- ブランチ名を`power_demand_ai`にリネーム

**2. AIタブ無効化修正**
```
コミット: 未記録
メッセージ: PyTorch未インストール時のAIタブ無効化を修正
```

**3. エラーハンドリング強化** (コミット 115ffc6)
```
メッセージ: 改善: ARIMA/指数平滑法のエラーハンドリング強化とインポートエラーの早期検出
変更内容:
- インポートエラーの早期検出と詳細メッセージ表示
- statsmodels, scikit-learnのインストール
```

**4. 欠損値補間対応** (コミット b1d33de)
```
メッセージ: 修正: STL時系列分解で欠損値を線形補間して連続性を保つ
変更内容:
- get_ai_data_series()にinterpolateパラメータ追加
- 線形補間による欠損値処理
- 時系列の連続性を保持
```

**5. STLパラメータ修正** (コミット 5c9ec96)
```
メッセージ: 修正: STLのperiodとseasonalパラメータを正しく指定
変更内容:
- period=24 (周期の長さ)
- seasonal=7 (スムージング幅)
- パラメータの意味を正しく理解して実装
```

## ファイル構成

### 主要ファイル
```
jukyu_app_python_desktop/
├── main.py                    # メインアプリケーション (約2360行)
│   ├── create_detail_page()   # 詳細分析タブ（5サブタブ）
│   ├── create_ai_page()       # AI予測タブ
│   ├── get_ai_data_series()   # データ取得（補間対応）
│   ├── run_stl_decomposition() # STL分解
│   ├── run_arima_forecast()    # ARIMA予測
│   ├── run_exponential_smoothing() # 指数平滑法
│   └── plot_residual_analysis() # 残差分析
├── ml/__init__.py             # オプションのML機能
├── requirements.txt           # 依存関係
├── data/                      # CSVデータ（60ファイル）
└── README.md                  # プロジェクト説明
```

### ドキュメント
```
├── DEVELOPMENT_HISTORY.md     # 本ファイル（開発履歴）
├── FEATURES.md                # 機能詳細説明
└── README.md                  # 使用方法
```

## 使用方法

### 環境セットアップ
```bash
# リポジトリをクローン
git clone https://github.com/CYLIU2003/demand_analy.git
cd demand_analy

# power_demand_aiブランチに切り替え
git checkout power_demand_ai

# 依存関係をインストール
cd jukyu_app_python_desktop
pip install -r requirements.txt

# アプリケーション起動
python main.py
```

### 基本的な操作フロー

**1. データ読み込み**
- エリア選択（eria_jukyu_202504_01など）
- 年月選択

**2. 詳細分析タブ**
- エリアと年月を選択
- 5つのサブタブで統計分析
- グラフとテーブルで結果を確認

**3. AI予測タブ**
- CSVファイルを読み込み
- 目的系列（列）を選択
- 予測手法を選択:
  - **STL時系列分解**: トレンド・季節性・残差を分離
  - **ARIMA予測**: 自動次数選定による予測
  - **指数平滑法**: 季節性を考慮した予測
- パラメータ調整:
  - 予測期間 (1-240)
  - 訓練データ比率 (0.5-0.95)
- ログ出力で詳細を確認

## トラブルシューティング

### エラー: "Unable to determine period from endog"
**原因**: STLが時系列の周期を検出できない

**解決策**:
1. 欠損値が補間されているか確認
2. データ数が十分か確認（最低48サンプル推奨）
3. `period`パラメータが正しく設定されているか確認

### エラー: "No module named 'statsmodels'"
**原因**: 必要なライブラリがインストールされていない

**解決策**:
```bash
pip install statsmodels scikit-learn scipy
```

### AIタブが押せない
**原因**: 古いバージョンのコードを使用している

**解決策**:
```bash
git pull origin power_demand_ai
```

---

## 追加開発（2025年11月10日）

### Transformer予測機能の実装

#### 背景
ユーザーからの要望: 「transformer機能もおいおい実装したいな」

既存のAI予測タブにTransformerベースのディープラーニング予測機能を追加することになった。

#### 実装内容

**1. Transformerモデルアーキテクチャ**
- エンコーダーのみのTransformer構造
- 位置エンコーディング（Sinusoidal）
- マルチヘッドアテンション機構

**モデル仕様:**
```python
DemandTransformerForecaster(
    context_length=48,        # 48時間（2日分）の履歴を使用
    prediction_length=24,     # 24時間先まで予測（設定可能）
    d_model=128,              # 埋め込み次元
    nhead=4,                  # アテンションヘッド数
    num_layers=2,             # Transformerレイヤー数
    dropout=0.1,              # ドロップアウト率
    feedforward_dim=256,      # フィードフォワード層の次元
    learning_rate=5e-4,       # 学習率
    batch_size=32,            # バッチサイズ
    epochs=30                 # 学習エポック数
)
```

**2. 学習プロセス**
- スライディングウィンドウ方式でデータセット作成
- 訓練データの80%で学習、20%で検証
- MSE損失関数、Adam最適化
- 勾配クリッピング（max_norm=1.0）
- StandardScalerによる正規化

**3. UI統合**
- 既存のARIMA/指数平滑法と同じインターフェース
- 「Transformer予測」ボタンで実行
- 学習曲線のログ出力（5エポックごと）
- 予測結果、評価指標、残差分析の可視化

#### 技術的な課題と解決

**課題1: 依存ライブラリの競合**
**問題**: scipy 1.11.4がnumpy 2.1.3に対応していない

**エラーメッセージ:**
```
ERROR: Cannot install ... because these package versions have conflicting dependencies.
scipy 1.11.4 depends on numpy<1.28.0 and >=1.21.6
```

**解決策**: requirements.txtを更新
```diff
- scipy==1.11.4
+ scipy==1.14.1
- statsmodels==0.14.1
+ statsmodels==0.14.4
```

**課題2: f-string内での条件式エラー**
**問題**: フォーマット指定子内で三項演算子を使用

**エラー:**
```python
f"{training_log.val_loss[-1]:.6f if training_log.val_loss[-1] else 'N/A'}"
# ValueError: Invalid format specifier
```

**解決策**: 条件式を事前に評価
```python
val_loss_str = f"{training_log.val_loss[-1]:.6f}" if training_log.val_loss[-1] is not None else "N/A"
f"最終検証損失: {val_loss_str}\n"
```

#### 実装されたメソッド

1. **`run_transformer_forecast()`**
   - データの訓練/テスト分割
   - Transformerモデルの作成と学習
   - 予測実行と評価指標計算
   - 結果のプロット

2. **`plot_training_curves()`**
   - 学習損失と検証損失のログ出力
   - エポックごとの進捗表示

#### パフォーマンス

**実測値（1,440サンプル、CPU）:**
- 学習時間: 約51秒（30エポック）
- 訓練データ: 1,152サンプル（80%）
- 検証データ: 288サンプル（20%）
- 予測時間: 瞬時（<0.1秒）

**学習曲線例:**
```
Epoch  5: Train: 0.356048, Val: 0.416264
Epoch 10: Train: 0.223000, Val: 0.312003
Epoch 15: Train: 0.185088, Val: 0.312274
Epoch 20: Train: 0.129399, Val: 0.287797
Epoch 25: Train: 0.116420, Val: 0.303150
Epoch 30: Train: 0.103616, Val: 0.316190
```

#### システム要件への影響

**追加要件（Transformer使用時）:**
- **CPU**: クアッドコア以上推奨（デュアルコアでも動作可能）
- **RAM**: 16 GB推奨（最小8 GB）
- **ストレージ**: 追加1.5 GB（PyTorch）
- **GPU**: オプション（CUDA対応でトレーニング高速化）

**ライブラリ追加:**
- PyTorch 2.4.1（CPU版で約1.5 GB）
- CUDA Toolkit（GPU使用時）

---

## 今後の拡張案

### 短期的改善
- [ ] 複数エリアの比較分析機能
- [ ] 予測結果のエクスポート機能（CSV, PNG）
- [ ] カスタム予測期間の設定UI改善
- [x] Transformerによるディープラーニング予測 ✅ **完了 (2025-11-10)**

### 中期的改善
- [ ] Prophet（Facebookの時系列予測ライブラリ）の統合
- [ ] LSTM予測モデルの追加
- [ ] Transformerのハイパーパラメータ調整UI
- [ ] GPU対応の自動検出と切り替え
- [ ] 気象データとの相関分析

### 長期的改善
- [ ] リアルタイムデータ対応
- [ ] Webアプリケーション化（Flask/Django）
- [ ] マルチユーザー対応とデータベース統合

## 参考資料

### 使用した統計手法
- **STL分解**: Cleveland, R. B., et al. (1990). "STL: A Seasonal-Trend Decomposition Procedure Based on Loess"
- **ARIMA**: Box, G. E. P., & Jenkins, G. M. (1970). "Time Series Analysis: Forecasting and Control"
- **指数平滑法**: Holt, C. C. (1957). "Forecasting seasonals and trends by exponentially weighted moving averages"
- **Transformer**: Vaswani, A., et al. (2017). "Attention Is All You Need" - NIPS 2017

### ライブラリドキュメント
- [statsmodels](https://www.statsmodels.org/stable/index.html)
- [scikit-learn](https://scikit-learn.org/stable/)
- [pandas](https://pandas.pydata.org/docs/)
- [matplotlib](https://matplotlib.org/stable/contents.html)
- [PyTorch](https://pytorch.org/docs/stable/index.html)

## 連絡先・サポート

### リポジトリ
- GitHub: https://github.com/CYLIU2003/demand_analy
- ブランチ: `power_demand_ai`

### 開発環境
- OS: Windows 10/11
- Python: 3.11.9
- Shell: PowerShell 5.1
- IDE: Visual Studio Code

## まとめ

このプロジェクトでは、電力需要分析ツールに以下の改善を加えました:

### フェーズ1: 統計分析機能（2025-11-09）
1. **ユーザーインターフェースの改善**: 直感的でない詳細分析タブを5つの明確なサブタブに再構成
2. **学術的分析機能の追加**: STL分解、ARIMA、指数平滑法などの標準的な時系列分析手法を実装
3. **堅牢なエラーハンドリング**: ライブラリ不足や欠損値に対する適切なエラーメッセージと処理
4. **評価指標の実装**: MAE、RMSE、MAPEによる予測精度の定量評価
5. **可視化の充実**: 統計グラフ、予測プロット、残差分析などの包括的な可視化

### フェーズ2: ディープラーニング機能（2025-11-10）
6. **Transformer予測モデル**: アテンション機構を使った最先端の時系列予測
7. **スケーラブルなアーキテクチャ**: CPU/GPU両対応、バッチ学習、学習曲線モニタリング
8. **依存関係管理**: PyTorchのオプショナルインストール、互換性の確保
9. **システム要件の文書化**: ハードウェア/ソフトウェア要件をREADMEに明記

すべての変更は`power_demand_ai`ブランチにコミットされ、他のPCやAIエージェントツールから参照可能です。
