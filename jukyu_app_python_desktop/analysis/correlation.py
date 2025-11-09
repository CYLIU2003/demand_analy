"""
相関分析モジュール
変数間の相関関係、因果関係の分析を実施
"""

from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import pearsonr, spearmanr


class CorrelationAnalyzer:
    """電力需給データの相関分析を行うクラス"""
    
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
    
    def correlation_matrix(self, columns: Optional[List[str]] = None, 
                          method: str = 'pearson') -> pd.DataFrame:
        """
        相関行列を計算
        
        Args:
            columns: 分析対象のカラムリスト（Noneの場合は全数値カラム）
            method: 'pearson', 'spearman', 'kendall'
        
        Returns:
            相関係数行列のDataFrame
        """
        if columns is None:
            columns = self.numeric_columns
        
        # 有効なカラムのみを抽出
        valid_columns = [col for col in columns if col in self.df.columns]
        
        if not valid_columns:
            raise ValueError("有効なカラムがありません")
        
        corr_matrix = self.df[valid_columns].corr(method=method)
        return corr_matrix
    
    def pairwise_correlation(self, column1: str, column2: str, 
                           method: str = 'pearson') -> Dict[str, float]:
        """
        2変数間の相関係数と有意性を計算
        
        Args:
            column1: 第1変数
            column2: 第2変数
            method: 'pearson' or 'spearman'
        
        Returns:
            相関係数、p値、サンプル数を含む辞書
        """
        if column1 not in self.df.columns or column2 not in self.df.columns:
            raise ValueError("指定されたカラムが見つかりません")
        
        # 両方が有効な行のみを抽出
        data = self.df[[column1, column2]].dropna()
        
        if len(data) < 3:
            raise ValueError("有効なデータ数が不足しています")
        
        if method == 'pearson':
            corr, p_value = pearsonr(data[column1], data[column2])
        elif method == 'spearman':
            corr, p_value = spearmanr(data[column1], data[column2])
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return {
            '相関係数': corr,
            'p値': p_value,
            'サンプル数': len(data),
            '有意': p_value < 0.05
        }
    
    def cross_correlation(self, column1: str, column2: str, 
                         max_lag: int = 24) -> Dict[int, float]:
        """
        相互相関関数を計算（時間遅れを考慮）
        
        Args:
            column1: 第1変数
            column2: 第2変数
            max_lag: 最大ラグ数
        
        Returns:
            ラグごとの相関係数を含む辞書
        """
        if column1 not in self.df.columns or column2 not in self.df.columns:
            raise ValueError("指定されたカラムが見つかりません")
        
        data1 = self.df[column1].dropna()
        data2 = self.df[column2].dropna()
        
        # 両方のデータの共通インデックスを取得
        common_index = data1.index.intersection(data2.index)
        data1 = data1.loc[common_index]
        data2 = data2.loc[common_index]
        
        cross_corr = {}
        for lag in range(-max_lag, max_lag + 1):
            if lag < 0:
                shifted_data1 = data1.iloc[-lag:]
                shifted_data2 = data2.iloc[:lag]
            elif lag > 0:
                shifted_data1 = data1.iloc[:-lag]
                shifted_data2 = data2.iloc[lag:]
            else:
                shifted_data1 = data1
                shifted_data2 = data2
            
            if len(shifted_data1) > 0 and len(shifted_data2) > 0:
                # インデックスを揃える
                min_len = min(len(shifted_data1), len(shifted_data2))
                corr = np.corrcoef(
                    shifted_data1.values[:min_len],
                    shifted_data2.values[:min_len]
                )[0, 1]
                cross_corr[lag] = corr
        
        return cross_corr
    
    def partial_correlation(self, x: str, y: str, z: List[str]) -> float:
        """
        偏相関係数を計算（制御変数を考慮）
        
        Args:
            x: 第1変数
            y: 第2変数
            z: 制御変数のリスト
        
        Returns:
            偏相関係数
        """
        all_vars = [x, y] + z
        for var in all_vars:
            if var not in self.df.columns:
                raise ValueError(f"Column {var} not found")
        
        # 有効なデータのみを抽出
        data = self.df[all_vars].dropna()
        
        if len(data) < len(all_vars) + 2:
            raise ValueError("データ数が不足しています")
        
        # 相関行列を計算
        corr_matrix = data.corr().values
        
        # 偏相関の計算（行列演算）
        try:
            inv_corr = np.linalg.inv(corr_matrix)
            idx_x = all_vars.index(x)
            idx_y = all_vars.index(y)
            partial_corr = -inv_corr[idx_x, idx_y] / np.sqrt(
                inv_corr[idx_x, idx_x] * inv_corr[idx_y, idx_y]
            )
        except np.linalg.LinAlgError:
            raise ValueError("相関行列が特異です")
        
        return partial_corr
    
    def find_high_correlations(self, threshold: float = 0.7, 
                              method: str = 'pearson') -> List[Tuple[str, str, float]]:
        """
        高い相関を持つ変数ペアを検出
        
        Args:
            threshold: 相関係数の閾値（絶対値）
            method: 相関係数の計算方法
        
        Returns:
            (変数1, 変数2, 相関係数)のタプルのリスト
        """
        corr_matrix = self.correlation_matrix(method=method)
        
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                corr_value = corr_matrix.iloc[i, j]
                if abs(corr_value) >= threshold:
                    high_corr_pairs.append((
                        corr_matrix.columns[i],
                        corr_matrix.columns[j],
                        corr_value
                    ))
        
        # 相関係数の絶対値でソート
        high_corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
        
        return high_corr_pairs
    
    def rolling_correlation(self, column1: str, column2: str, 
                           window: int = 24) -> pd.Series:
        """
        移動相関係数を計算
        
        Args:
            column1: 第1変数
            column2: 第2変数
            window: ウィンドウサイズ
        
        Returns:
            移動相関係数のSeries
        """
        if column1 not in self.df.columns or column2 not in self.df.columns:
            raise ValueError("指定されたカラムが見つかりません")
        
        rolling_corr = self.df[column1].rolling(window=window).corr(self.df[column2])
        return rolling_corr
    
    def correlation_with_lag(self, column1: str, column2: str, 
                           lag: int) -> float:
        """
        ラグを考慮した相関係数を計算
        
        Args:
            column1: 第1変数
            column2: 第2変数（ラグを適用）
            lag: ラグ数（正の値で column2 を遅らせる）
        
        Returns:
            相関係数
        """
        if column1 not in self.df.columns or column2 not in self.df.columns:
            raise ValueError("指定されたカラムが見つかりません")
        
        data1 = self.df[column1]
        data2 = self.df[column2].shift(lag)
        
        # 両方が有効な行のみを抽出
        valid_data = pd.DataFrame({column1: data1, column2: data2}).dropna()
        
        if len(valid_data) < 3:
            return np.nan
        
        corr = valid_data[column1].corr(valid_data[column2])
        return corr
    
    def generate_report(self, columns: Optional[List[str]] = None) -> str:
        """
        相関分析レポートをテキスト形式で生成
        
        Args:
            columns: レポート対象のカラムリスト
        
        Returns:
            相関分析レポートの文字列
        """
        if columns is None:
            columns = self.numeric_columns[:10]  # 最初の10列
        
        report_lines = ["=" * 60]
        report_lines.append("相関分析レポート")
        report_lines.append("=" * 60)
        report_lines.append("")
        
        # 相関行列
        report_lines.append("【相関行列 (Pearson)】")
        try:
            corr_matrix = self.correlation_matrix(columns, method='pearson')
            report_lines.append(corr_matrix.to_string())
        except Exception as e:
            report_lines.append(f"エラー: {str(e)}")
        report_lines.append("")
        
        # 高相関ペア
        report_lines.append("【高相関変数ペア (|r| >= 0.7)】")
        try:
            high_corr = self.find_high_correlations(threshold=0.7)
            if high_corr:
                for var1, var2, corr in high_corr[:10]:  # 上位10件
                    report_lines.append(f"  {var1} ⇔ {var2}: r = {corr:.4f}")
            else:
                report_lines.append("  該当なし")
        except Exception as e:
            report_lines.append(f"  エラー: {str(e)}")
        report_lines.append("")
        
        report_lines.append("=" * 60)
        
        return "\n".join(report_lines)
