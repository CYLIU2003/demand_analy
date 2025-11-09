"""
学術研究用サンプルスクリプト
電力需給データの分析例を示す
"""

from pathlib import Path
import pandas as pd
import numpy as np

# 分析モジュール
from analysis.statistics import StatisticalAnalyzer
from analysis.timeseries import TimeSeriesAnalyzer
from analysis.correlation import CorrelationAnalyzer

# モデル
try:
    from models.transformer_forecaster import TransformerForecaster
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available. Transformer examples will be skipped.")

# エクスポート
from exports.academic_exporter import AcademicExporter
from exports.figure_generator import FigureGenerator

# ユーティリティ
from utils.data_processor import DataProcessor


def load_sample_data():
    """サンプルデータの読み込み"""
    data_dir = Path(__file__).parent / 'data'
    
    # 最初に見つかったCSVファイルを読み込む
    csv_files = list(data_dir.glob('eria_jukyu_*.csv'))
    
    if not csv_files:
        print("Error: No data files found in data/")
        return None
    
    filepath = csv_files[0]
    print(f"Loading: {filepath.name}")
    
    # 複数のエンコーディングで試行
    for encoding in ['shift_jis', 'cp932', 'utf-8']:
        try:
            df = pd.read_csv(filepath, encoding=encoding, skiprows=0)
            
            # 単位行をスキップ
            if '単位' in str(df.columns[0]) or 'MW' in str(df.columns[0]):
                df = pd.read_csv(filepath, encoding=encoding, skiprows=1)
            
            print(f"Successfully loaded with {encoding} encoding")
            print(f"Shape: {df.shape}")
            print(f"Columns: {df.columns.tolist()}")
            return df
        except:
            continue
    
    print("Error: Failed to load data")
    return None


def example_statistical_analysis(df):
    """統計分析の例"""
    print("\n" + "="*60)
    print("統計分析の例")
    print("="*60)
    
    analyzer = StatisticalAnalyzer(df)
    
    # 数値カラムを取得
    numeric_cols = analyzer.numeric_columns[:5]  # 最初の5列
    
    # 記述統計
    print("\n【記述統計量】")
    stats = analyzer.descriptive_statistics(numeric_cols)
    print(stats)
    
    # 外れ値検出
    if len(numeric_cols) > 0:
        col = numeric_cols[0]
        print(f"\n【外れ値検出: {col}】")
        outliers = analyzer.detect_outliers(col, method='iqr')
        print(f"外れ値数: {outliers.sum()} / {len(df)} ({outliers.sum()/len(df)*100:.2f}%)")
    
    # レポート生成
    print("\n【統計レポート生成中...】")
    report = analyzer.generate_report(numeric_cols)
    
    # レポートをファイルに保存
    output_dir = Path('research_outputs')
    output_dir.mkdir(exist_ok=True)
    with open(output_dir / 'statistics_report.txt', 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"レポート保存: research_outputs/statistics_report.txt")


def example_timeseries_analysis(df, time_column):
    """時系列分析の例"""
    print("\n" + "="*60)
    print("時系列分析の例")
    print("="*60)
    
    ts_analyzer = TimeSeriesAnalyzer(df, time_column)
    
    # 数値カラムを取得
    numeric_cols = [col for col in df.columns 
                   if pd.api.types.is_numeric_dtype(df[col]) and col != time_column]
    
    if len(numeric_cols) == 0:
        print("No numeric columns found for analysis")
        return
    
    target_col = numeric_cols[0]
    print(f"\n分析対象: {target_col}")
    
    # 定常性検定
    print("\n【定常性検定 (ADF検定)】")
    try:
        adf_result = ts_analyzer.stationarity_test(target_col)
        print(f"ADF統計量: {adf_result['ADF統計量']:.4f}")
        print(f"p値: {adf_result['p値']:.4f}")
        print(f"定常: {adf_result['定常']}")
    except Exception as e:
        print(f"Error: {e}")
    
    # 自己相関
    print("\n【自己相関分析】")
    try:
        acf_values = ts_analyzer.autocorrelation(target_col, nlags=24)
        print(f"ラグ1の自己相関: {acf_values[1]:.4f}")
        if len(acf_values) > 24:
            print(f"ラグ24の自己相関: {acf_values[24]:.4f}")
    except Exception as e:
        print(f"Error: {e}")
    
    # ピーク検出
    print("\n【ピーク検出】")
    try:
        peaks, properties = ts_analyzer.detect_peaks(target_col)
        print(f"検出されたピーク数: {len(peaks)}")
        if len(peaks) > 1:
            print(f"平均ピーク間隔: {np.mean(np.diff(peaks)):.2f} サンプル")
    except Exception as e:
        print(f"Error: {e}")


def example_correlation_analysis(df):
    """相関分析の例"""
    print("\n" + "="*60)
    print("相関分析の例")
    print("="*60)
    
    corr_analyzer = CorrelationAnalyzer(df)
    
    # 数値カラムを取得（最大10列）
    numeric_cols = corr_analyzer.numeric_columns[:10]
    
    if len(numeric_cols) < 2:
        print("Not enough numeric columns for correlation analysis")
        return
    
    # 相関行列
    print("\n【相関行列 (Pearson)】")
    corr_matrix = corr_analyzer.correlation_matrix(numeric_cols, method='pearson')
    print(corr_matrix)
    
    # 高相関ペア
    print("\n【高相関変数ペア (|r| >= 0.7)】")
    try:
        high_corr = corr_analyzer.find_high_correlations(threshold=0.7)
        if high_corr:
            for var1, var2, corr in high_corr[:5]:  # 上位5件
                print(f"  {var1} ⇔ {var2}: r = {corr:.4f}")
        else:
            print("  該当なし")
    except Exception as e:
        print(f"Error: {e}")


def example_transformer_model(df, time_column):
    """Transformerモデルの例"""
    if not TORCH_AVAILABLE:
        print("\n" + "="*60)
        print("Transformer モデルの例 (スキップ: PyTorchが利用不可)")
        print("="*60)
        return
    
    print("\n" + "="*60)
    print("Transformer モデルの例")
    print("="*60)
    
    # 数値カラムを取得
    numeric_cols = [col for col in df.columns 
                   if pd.api.types.is_numeric_dtype(df[col]) and col != time_column]
    
    if len(numeric_cols) == 0:
        print("No numeric columns found for modeling")
        return
    
    target_col = numeric_cols[0]
    print(f"\n予測対象: {target_col}")
    
    # データ前処理
    processor = DataProcessor()
    df_processed = processor.handle_missing_values(df, method='interpolate', columns=[target_col, time_column])
    df_processed = processor.create_time_features(df_processed, time_column)
    
    # モデル設定
    config = {
        'd_model': 32,  # 小さめに設定（デモ用）
        'nhead': 2,
        'num_encoder_layers': 1,
        'learning_rate': 0.001
    }
    
    # モデル初期化
    model = TransformerForecaster(config)
    
    print("\n【データ準備中...】")
    # データ準備
    try:
        X, y = model.prepare_data(df_processed, target_column=target_col, 
                                  sequence_length=24, feature_columns=[target_col])
        print(f"データ形状: X={X.shape}, y={y.shape}")
    except Exception as e:
        print(f"Error in data preparation: {e}")
        return
    
    # データ分割
    train_size = int(0.7 * len(X))
    val_size = int(0.15 * len(X))
    
    X_train, y_train = X[:train_size], y[:train_size]
    X_val, y_val = X[train_size:train_size+val_size], y[train_size:train_size+val_size]
    X_test, y_test = X[train_size+val_size:], y[train_size+val_size:]
    
    print(f"訓練データ: {len(X_train)}, 検証データ: {len(X_val)}, テストデータ: {len(X_test)}")
    
    # モデル訓練（短いエポック数でデモ）
    print("\n【モデル訓練中...】")
    try:
        history = model.train(X_train, y_train, X_val, y_val, epochs=10, batch_size=32)
        print("訓練完了")
    except Exception as e:
        print(f"Error in training: {e}")
        return
    
    # 評価
    print("\n【モデル評価】")
    try:
        metrics = model.evaluate(X_test, y_test)
        print(f"MAE: {metrics['MAE']:.2f}")
        print(f"RMSE: {metrics['RMSE']:.2f}")
        print(f"MAPE: {metrics['MAPE']:.2f}%")
        print(f"R²: {metrics['R2']:.4f}")
    except Exception as e:
        print(f"Error in evaluation: {e}")
        return
    
    # モデル保存
    output_dir = Path('research_outputs')
    output_dir.mkdir(exist_ok=True)
    model_path = output_dir / 'transformer_model.pth'
    try:
        model.save_model(str(model_path))
        print(f"\nモデル保存: {model_path}")
    except Exception as e:
        print(f"Error in saving model: {e}")


def example_export(df):
    """エクスポート機能の例"""
    print("\n" + "="*60)
    print("エクスポート機能の例")
    print("="*60)
    
    exporter = AcademicExporter(output_dir='research_outputs')
    
    # データセットのエクスポート
    print("\n【データセットエクスポート】")
    try:
        filepath = exporter.export_dataset(df, filename='sample_data', format='csv')
        print(f"CSV保存: {filepath}")
    except Exception as e:
        print(f"Error: {e}")
    
    # 統計レポートのエクスポート
    print("\n【統計レポートエクスポート】")
    analyzer = StatisticalAnalyzer(df)
    stats = analyzer.descriptive_statistics()
    
    statistics_dict = {
        '記述統計': stats.to_dict()
    }
    
    try:
        filepath = exporter.export_statistics_report(
            statistics_dict, 
            filename='statistics_report',
            format='json'
        )
        print(f"レポート保存: {filepath}")
    except Exception as e:
        print(f"Error: {e}")


def example_figure_generation(df, time_column):
    """図表生成の例"""
    print("\n" + "="*60)
    print("図表生成の例")
    print("="*60)
    
    fig_gen = FigureGenerator(output_dir='figures')
    
    # 数値カラムを取得
    numeric_cols = [col for col in df.columns 
                   if pd.api.types.is_numeric_dtype(df[col]) and col != time_column]
    
    if len(numeric_cols) == 0:
        print("No numeric columns found for plotting")
        return
    
    # 時系列プロット
    print("\n【時系列プロット生成中...】")
    try:
        # 最初の3列をプロット
        plot_cols = numeric_cols[:3]
        filepath = fig_gen.plot_time_series(
            data=df,
            time_column=time_column,
            value_columns=plot_cols,
            title='電力需給の時系列変化',
            filename='timeseries_sample'
        )
        print(f"図表保存: {filepath}")
    except Exception as e:
        print(f"Error: {e}")
    
    # 相関行列のヒートマップ
    print("\n【相関行列ヒートマップ生成中...】")
    try:
        corr_analyzer = CorrelationAnalyzer(df)
        corr_matrix = corr_analyzer.correlation_matrix(numeric_cols[:8])  # 最大8列
        
        filepath = fig_gen.plot_correlation_matrix(
            correlation_matrix=corr_matrix,
            title='発電方式間の相関関係',
            filename='correlation_sample'
        )
        print(f"図表保存: {filepath}")
    except Exception as e:
        print(f"Error: {e}")


def main():
    """メイン実行関数"""
    print("="*60)
    print("電力需給分析ツール - 学術研究用サンプル")
    print("="*60)
    
    # データ読み込み
    df = load_sample_data()
    
    if df is None:
        print("\nデータの読み込みに失敗しました")
        print("data/ フォルダに eria_jukyu_YYYYMM_AA.csv ファイルを配置してください")
        return
    
    # 時刻カラムを特定
    time_column = None
    for col in df.columns:
        col_str = str(col).lower()
        if any(keyword in col_str for keyword in ['time', '時刻', '時間', 'date', '日時']):
            time_column = col
            break
    
    if time_column:
        df[time_column] = pd.to_datetime(df[time_column], errors='coerce')
        print(f"\n時刻カラム: {time_column}")
    
    # 各分析例を実行
    try:
        example_statistical_analysis(df)
    except Exception as e:
        print(f"Error in statistical analysis: {e}")
    
    if time_column:
        try:
            example_timeseries_analysis(df, time_column)
        except Exception as e:
            print(f"Error in timeseries analysis: {e}")
    
    try:
        example_correlation_analysis(df)
    except Exception as e:
        print(f"Error in correlation analysis: {e}")
    
    if time_column:
        try:
            example_transformer_model(df, time_column)
        except Exception as e:
            print(f"Error in transformer model: {e}")
    
    try:
        example_export(df)
    except Exception as e:
        print(f"Error in export: {e}")
    
    if time_column:
        try:
            example_figure_generation(df, time_column)
        except Exception as e:
            print(f"Error in figure generation: {e}")
    
    print("\n" + "="*60)
    print("全ての例が完了しました")
    print("出力ファイル:")
    print("  - research_outputs/ : データ・レポート")
    print("  - figures/ : 図表 (PNG, PDF)")
    print("="*60)


if __name__ == '__main__':
    main()
