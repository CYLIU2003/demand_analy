"""
基本統計分析モジュール
電力需給データの記述統計、分布解析などを実施
"""

from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from scipy import stats


class StatisticalAnalyzer:
    """電力需給データの統計分析を行うクラス"""
    
    def __init__(self, dataframe: pd.DataFrame):
        """
        Args:
            dataframe: 分析対象のデータフレーム
        """
        self.df = dataframe
        self.numeric_columns = self._get_numeric_columns()
    
    def _get_numeric_columns(self) -> List[str]:
        """数値型のカラムを取得"""
        return [col for col in self.df.columns 
                if pd.api.types.is_numeric_dtype(self.df[col])]
    
    def descriptive_statistics(self, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        記述統計量を計算
        
        Args:
            columns: 分析対象のカラムリスト（Noneの場合は全数値カラム）
        
        Returns:
            記述統計量のDataFrame（平均、標準偏差、最小値、最大値など）
        """
        if columns is None:
            columns = self.numeric_columns
        
        stats_dict = {}
        for col in columns:
            if col not in self.df.columns:
                continue
            
            data = self.df[col].dropna()
            if len(data) == 0:
                continue
            
            stats_dict[col] = {
                '件数': len(data),
                '平均': data.mean(),
                '標準偏差': data.std(),
                '最小値': data.min(),
                '25%点': data.quantile(0.25),
                '中央値': data.median(),
                '75%点': data.quantile(0.75),
                '最大値': data.max(),
                '歪度': stats.skew(data),
                '尖度': stats.kurtosis(data)
            }
        
        return pd.DataFrame(stats_dict).T
    
    def detect_outliers(self, column: str, method: str = 'iqr', 
                       threshold: float = 1.5) -> pd.Series:
        """
        外れ値を検出
        
        Args:
            column: 分析対象カラム
            method: 検出方法 ('iqr' or 'zscore')
            threshold: 閾値（IQR法: 1.5推奨、Z-score法: 3.0推奨）
        
        Returns:
            外れ値のブール値Series
        """
        if column not in self.df.columns:
            raise ValueError(f"Column {column} not found")
        
        data = self.df[column].dropna()
        
        if method == 'iqr':
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            outliers = (self.df[column] < lower_bound) | (self.df[column] > upper_bound)
        
        elif method == 'zscore':
            z_scores = np.abs(stats.zscore(data))
            outliers = pd.Series(False, index=self.df.index)
            outliers.loc[data.index] = z_scores > threshold
        
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return outliers
    
    def hourly_statistics(self, time_column: str, 
                         value_columns: List[str]) -> pd.DataFrame:
        """
        時間帯別の統計を計算
        
        Args:
            time_column: 時刻カラム
            value_columns: 集計対象のカラムリスト
        
        Returns:
            時間帯別統計のDataFrame
        """
        df_copy = self.df.copy()
        df_copy['hour'] = pd.to_datetime(df_copy[time_column]).dt.hour
        
        hourly_stats = {}
        for col in value_columns:
            if col in df_copy.columns:
                hourly_stats[f'{col}_平均'] = df_copy.groupby('hour')[col].mean()
                hourly_stats[f'{col}_最大'] = df_copy.groupby('hour')[col].max()
                hourly_stats[f'{col}_最小'] = df_copy.groupby('hour')[col].min()
        
        return pd.DataFrame(hourly_stats)
    
    def daily_statistics(self, time_column: str, 
                        value_columns: List[str]) -> pd.DataFrame:
        """
        日別の統計を計算
        
        Args:
            time_column: 時刻カラム
            value_columns: 集計対象のカラムリスト
        
        Returns:
            日別統計のDataFrame
        """
        df_copy = self.df.copy()
        df_copy['date'] = pd.to_datetime(df_copy[time_column]).dt.date
        
        daily_stats = {}
        for col in value_columns:
            if col in df_copy.columns:
                daily_stats[f'{col}_平均'] = df_copy.groupby('date')[col].mean()
                daily_stats[f'{col}_最大'] = df_copy.groupby('date')[col].max()
                daily_stats[f'{col}_最小'] = df_copy.groupby('date')[col].min()
                daily_stats[f'{col}_合計'] = df_copy.groupby('date')[col].sum()
        
        return pd.DataFrame(daily_stats)
    
    def normality_test(self, column: str) -> Dict[str, float]:
        """
        正規性検定（Shapiro-Wilk検定）
        
        Args:
            column: 検定対象カラム
        
        Returns:
            検定統計量とp値を含む辞書
        """
        if column not in self.df.columns:
            raise ValueError(f"Column {column} not found")
        
        data = self.df[column].dropna()
        
        # サンプルサイズが大きすぎる場合はサンプリング
        if len(data) > 5000:
            data = data.sample(n=5000, random_state=42)
        
        statistic, p_value = stats.shapiro(data)
        
        return {
            '統計量': statistic,
            'p値': p_value,
            '正規分布': p_value > 0.05  # α=0.05
        }
    
    def peak_analysis(self, column: str, top_n: int = 10) -> pd.DataFrame:
        """
        ピーク値の分析
        
        Args:
            column: 分析対象カラム
            top_n: 上位何件を抽出するか
        
        Returns:
            ピーク値のDataFrame
        """
        if column not in self.df.columns:
            raise ValueError(f"Column {column} not found")
        
        # 上位n件を取得
        top_peaks = self.df.nlargest(top_n, column)
        
        return top_peaks
    
    def generate_report(self, columns: Optional[List[str]] = None) -> str:
        """
        統計レポートをテキスト形式で生成
        
        Args:
            columns: レポート対象のカラムリスト
        
        Returns:
            統計レポートの文字列
        """
        if columns is None:
            columns = self.numeric_columns[:5]  # 最初の5列
        
        report_lines = ["=" * 60]
        report_lines.append("電力需給データ統計分析レポート")
        report_lines.append("=" * 60)
        report_lines.append("")
        
        # 記述統計
        desc_stats = self.descriptive_statistics(columns)
        report_lines.append("【記述統計量】")
        report_lines.append(desc_stats.to_string())
        report_lines.append("")
        
        # 各カラムの詳細分析
        for col in columns:
            if col not in self.df.columns:
                continue
            
            report_lines.append(f"【{col} の詳細分析】")
            
            # 外れ値検出
            outliers = self.detect_outliers(col)
            outlier_count = outliers.sum()
            outlier_ratio = outlier_count / len(self.df) * 100
            report_lines.append(f"  外れ値: {outlier_count}件 ({outlier_ratio:.2f}%)")
            
            # 正規性検定
            try:
                normality = self.normality_test(col)
                report_lines.append(f"  正規性検定 (Shapiro-Wilk): p値={normality['p値']:.4f}")
                report_lines.append(f"  正規分布と判定: {'はい' if normality['正規分布'] else 'いいえ'}")
            except Exception as e:
                report_lines.append(f"  正規性検定: エラー ({str(e)})")
            
            report_lines.append("")
        
        report_lines.append("=" * 60)
        
        return "\n".join(report_lines)
