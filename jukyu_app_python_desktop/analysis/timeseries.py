"""
時系列分析モジュール
トレンド分析、季節性分析、自己相関分析などを実施
"""

from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from scipy import signal
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf, pacf, adfuller


class TimeSeriesAnalyzer:
    """電力需給データの時系列分析を行うクラス"""
    
    def __init__(self, dataframe: pd.DataFrame, time_column: str):
        """
        Args:
            dataframe: 分析対象のデータフレーム
            time_column: 時刻を表すカラム名
        """
        self.df = dataframe.copy()
        self.time_column = time_column
        
        # 時刻カラムをdatetimeに変換してインデックスに設定
        self.df[time_column] = pd.to_datetime(self.df[time_column])
        self.df = self.df.sort_values(time_column)
        self.df.set_index(time_column, inplace=True)
    
    def detect_trend(self, column: str, window: int = 24) -> pd.Series:
        """
        移動平均によるトレンド検出
        
        Args:
            column: 分析対象カラム
            window: 移動平均のウィンドウサイズ（デフォルト24時間）
        
        Returns:
            トレンド成分のSeries
        """
        if column not in self.df.columns:
            raise ValueError(f"Column {column} not found")
        
        trend = self.df[column].rolling(window=window, center=True).mean()
        return trend
    
    def seasonal_decomposition(self, column: str, period: int = 24, 
                              model: str = 'additive') -> Dict[str, pd.Series]:
        """
        季節性分解（トレンド、季節性、残差に分解）
        
        Args:
            column: 分析対象カラム
            period: 季節周期（デフォルト24時間）
            model: 'additive'（加法モデル）または'multiplicative'（乗法モデル）
        
        Returns:
            トレンド、季節性、残差を含む辞書
        """
        if column not in self.df.columns:
            raise ValueError(f"Column {column} not found")
        
        # 欠損値を補間
        data = self.df[column].interpolate(method='linear')
        
        # 最小データ数をチェック
        if len(data) < 2 * period:
            raise ValueError(f"データ数が不足しています（最低 {2*period} 件必要）")
        
        try:
            result = seasonal_decompose(data, model=model, period=period, extrapolate_trend='freq')
            
            return {
                'trend': result.trend,
                'seasonal': result.seasonal,
                'residual': result.resid,
                'observed': data
            }
        except Exception as e:
            raise ValueError(f"季節性分解に失敗: {str(e)}")
    
    def autocorrelation(self, column: str, nlags: int = 48) -> np.ndarray:
        """
        自己相関係数を計算
        
        Args:
            column: 分析対象カラム
            nlags: ラグの最大数（デフォルト48時間）
        
        Returns:
            自己相関係数の配列
        """
        if column not in self.df.columns:
            raise ValueError(f"Column {column} not found")
        
        data = self.df[column].dropna()
        
        if len(data) < nlags + 1:
            nlags = len(data) - 1
        
        acf_values = acf(data, nlags=nlags, fft=True)
        return acf_values
    
    def partial_autocorrelation(self, column: str, nlags: int = 48) -> np.ndarray:
        """
        偏自己相関係数を計算
        
        Args:
            column: 分析対象カラム
            nlags: ラグの最大数
        
        Returns:
            偏自己相関係数の配列
        """
        if column not in self.df.columns:
            raise ValueError(f"Column {column} not found")
        
        data = self.df[column].dropna()
        
        if len(data) < nlags + 1:
            nlags = len(data) - 1
        
        pacf_values = pacf(data, nlags=nlags)
        return pacf_values
    
    def stationarity_test(self, column: str) -> Dict[str, float]:
        """
        定常性検定（Augmented Dickey-Fuller検定）
        
        Args:
            column: 検定対象カラム
        
        Returns:
            検定統計量、p値、臨界値を含む辞書
        """
        if column not in self.df.columns:
            raise ValueError(f"Column {column} not found")
        
        data = self.df[column].dropna()
        
        result = adfuller(data, autolag='AIC')
        
        return {
            'ADF統計量': result[0],
            'p値': result[1],
            '使用ラグ数': result[2],
            '観測数': result[3],
            '臨界値_1%': result[4]['1%'],
            '臨界値_5%': result[4]['5%'],
            '臨界値_10%': result[4]['10%'],
            '定常': result[1] < 0.05  # p値が0.05未満なら定常
        }
    
    def detect_peaks(self, column: str, prominence: float = None) -> Tuple[np.ndarray, Dict]:
        """
        ピーク検出
        
        Args:
            column: 分析対象カラム
            prominence: ピークの prominence（目立ち度）の閾値
        
        Returns:
            ピークのインデックスとプロパティ
        """
        if column not in self.df.columns:
            raise ValueError(f"Column {column} not found")
        
        data = self.df[column].dropna().values
        
        if prominence is None:
            prominence = np.std(data) * 0.5
        
        peaks, properties = signal.find_peaks(data, prominence=prominence)
        
        return peaks, properties
    
    def detect_change_points(self, column: str, window: int = 24, 
                            threshold: float = 2.0) -> List[int]:
        """
        変化点検出（移動分散ベース）
        
        Args:
            column: 分析対象カラム
            window: 移動ウィンドウサイズ
            threshold: 変化点判定の閾値（標準偏差の倍数）
        
        Returns:
            変化点のインデックスリスト
        """
        if column not in self.df.columns:
            raise ValueError(f"Column {column} not found")
        
        data = self.df[column].dropna()
        
        # 移動分散を計算
        rolling_var = data.rolling(window=window).var()
        
        # 分散の変化率
        var_change = rolling_var.diff().abs()
        
        # 閾値を超える点を変化点として検出
        threshold_value = var_change.mean() + threshold * var_change.std()
        change_points = var_change[var_change > threshold_value].index.tolist()
        
        return change_points
    
    def calculate_growth_rate(self, column: str, period: int = 24) -> pd.Series:
        """
        成長率（変化率）を計算
        
        Args:
            column: 分析対象カラム
            period: 比較期間（デフォルト24時間前との比較）
        
        Returns:
            成長率のSeries（%）
        """
        if column not in self.df.columns:
            raise ValueError(f"Column {column} not found")
        
        data = self.df[column]
        growth_rate = ((data - data.shift(period)) / data.shift(period)) * 100
        
        return growth_rate
    
    def moving_statistics(self, column: str, window: int = 24) -> pd.DataFrame:
        """
        移動統計量を計算
        
        Args:
            column: 分析対象カラム
            window: ウィンドウサイズ
        
        Returns:
            移動平均、移動標準偏差、移動最大値、移動最小値を含むDataFrame
        """
        if column not in self.df.columns:
            raise ValueError(f"Column {column} not found")
        
        data = self.df[column]
        
        result = pd.DataFrame({
            f'{column}_移動平均': data.rolling(window=window).mean(),
            f'{column}_移動標準偏差': data.rolling(window=window).std(),
            f'{column}_移動最大': data.rolling(window=window).max(),
            f'{column}_移動最小': data.rolling(window=window).min(),
        })
        
        return result
    
    def generate_report(self, column: str) -> str:
        """
        時系列分析レポートをテキスト形式で生成
        
        Args:
            column: レポート対象カラム
        
        Returns:
            時系列分析レポートの文字列
        """
        report_lines = ["=" * 60]
        report_lines.append(f"時系列分析レポート: {column}")
        report_lines.append("=" * 60)
        report_lines.append("")
        
        # 定常性検定
        report_lines.append("【定常性検定 (ADF検定)】")
        try:
            adf_result = self.stationarity_test(column)
            report_lines.append(f"  ADF統計量: {adf_result['ADF統計量']:.4f}")
            report_lines.append(f"  p値: {adf_result['p値']:.4f}")
            report_lines.append(f"  臨界値 (1%): {adf_result['臨界値_1%']:.4f}")
            report_lines.append(f"  臨界値 (5%): {adf_result['臨界値_5%']:.4f}")
            report_lines.append(f"  臨界値 (10%): {adf_result['臨界値_10%']:.4f}")
            report_lines.append(f"  定常と判定: {'はい' if adf_result['定常'] else 'いいえ'}")
        except Exception as e:
            report_lines.append(f"  エラー: {str(e)}")
        report_lines.append("")
        
        # 自己相関
        report_lines.append("【自己相関分析】")
        try:
            acf_values = self.autocorrelation(column, nlags=24)
            report_lines.append(f"  ラグ1の自己相関: {acf_values[1]:.4f}")
            report_lines.append(f"  ラグ24の自己相関: {acf_values[24]:.4f}" if len(acf_values) > 24 else "  ラグ24: データ不足")
        except Exception as e:
            report_lines.append(f"  エラー: {str(e)}")
        report_lines.append("")
        
        # ピーク検出
        report_lines.append("【ピーク検出】")
        try:
            peaks, properties = self.detect_peaks(column)
            report_lines.append(f"  検出されたピーク数: {len(peaks)}")
            if len(peaks) > 0:
                report_lines.append(f"  平均ピーク間隔: {np.mean(np.diff(peaks)):.2f} サンプル")
        except Exception as e:
            report_lines.append(f"  エラー: {str(e)}")
        report_lines.append("")
        
        report_lines.append("=" * 60)
        
        return "\n".join(report_lines)
