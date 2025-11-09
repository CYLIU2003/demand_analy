"""
データ前処理ユーティリティ
欠損値処理、外れ値処理、特徴量エンジニアリングを提供
"""

from typing import List, Optional
import pandas as pd
import numpy as np


class DataPreprocessor:
    """データ前処理を行うクラス"""
    
    def __init__(self, dataframe: pd.DataFrame):
        """
        Args:
            dataframe: 前処理対象のDataFrame
        """
        self.df = dataframe.copy()
    
    def fill_missing_values(self, method: str = 'interpolate', 
                           columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        欠損値を補完
        
        Args:
            method: 'interpolate', 'forward', 'backward', 'mean', 'median', 'zero'
            columns: 対象カラム（Noneの場合は全数値カラム）
        
        Returns:
            補完後のDataFrame
        """
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns.tolist()
        
        df_filled = self.df.copy()
        
        for col in columns:
            if col not in df_filled.columns:
                continue
            
            if method == 'interpolate':
                df_filled[col] = df_filled[col].interpolate(method='linear')
            elif method == 'forward':
                df_filled[col] = df_filled[col].fillna(method='ffill')
            elif method == 'backward':
                df_filled[col] = df_filled[col].fillna(method='bfill')
            elif method == 'mean':
                df_filled[col] = df_filled[col].fillna(df_filled[col].mean())
            elif method == 'median':
                df_filled[col] = df_filled[col].fillna(df_filled[col].median())
            elif method == 'zero':
                df_filled[col] = df_filled[col].fillna(0)
            else:
                raise ValueError(f"Unknown method: {method}")
        
        return df_filled
    
    def remove_outliers(self, column: str, method: str = 'iqr', 
                       threshold: float = 1.5) -> pd.DataFrame:
        """
        外れ値を除去
        
        Args:
            column: 対象カラム
            method: 'iqr' or 'zscore'
            threshold: 閾値
        
        Returns:
            外れ値除去後のDataFrame
        """
        if column not in self.df.columns:
            raise ValueError(f"Column {column} not found")
        
        df_cleaned = self.df.copy()
        data = df_cleaned[column]
        
        if method == 'iqr':
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            mask = (data >= lower_bound) & (data <= upper_bound)
        
        elif method == 'zscore':
            from scipy import stats
            z_scores = np.abs(stats.zscore(data.dropna()))
            mask = pd.Series(True, index=df_cleaned.index)
            mask.loc[data.dropna().index] = z_scores <= threshold
        
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return df_cleaned[mask]
    
    def add_time_features(self, time_column: str) -> pd.DataFrame:
        """
        時刻特徴量を追加
        
        Args:
            time_column: 時刻カラム
        
        Returns:
            特徴量追加後のDataFrame
        """
        if time_column not in self.df.columns:
            raise ValueError(f"Column {time_column} not found")
        
        df_featured = self.df.copy()
        dt = pd.to_datetime(df_featured[time_column])
        
        # 基本的な時刻特徴量
        df_featured['year'] = dt.dt.year
        df_featured['month'] = dt.dt.month
        df_featured['day'] = dt.dt.day
        df_featured['hour'] = dt.dt.hour
        df_featured['dayofweek'] = dt.dt.dayofweek  # 月曜=0, 日曜=6
        df_featured['dayofyear'] = dt.dt.dayofyear
        df_featured['weekofyear'] = dt.dt.isocalendar().week
        
        # 周期的特徴量（sin/cos変換）
        df_featured['hour_sin'] = np.sin(2 * np.pi * df_featured['hour'] / 24)
        df_featured['hour_cos'] = np.cos(2 * np.pi * df_featured['hour'] / 24)
        df_featured['month_sin'] = np.sin(2 * np.pi * df_featured['month'] / 12)
        df_featured['month_cos'] = np.cos(2 * np.pi * df_featured['month'] / 12)
        df_featured['dayofweek_sin'] = np.sin(2 * np.pi * df_featured['dayofweek'] / 7)
        df_featured['dayofweek_cos'] = np.cos(2 * np.pi * df_featured['dayofweek'] / 7)
        
        # 休日フラグ（土日）
        df_featured['is_weekend'] = (df_featured['dayofweek'] >= 5).astype(int)
        
        return df_featured
    
    def add_lag_features(self, column: str, lags: List[int]) -> pd.DataFrame:
        """
        ラグ特徴量を追加
        
        Args:
            column: 対象カラム
            lags: ラグ数のリスト（例: [1, 24, 168]）
        
        Returns:
            特徴量追加後のDataFrame
        """
        if column not in self.df.columns:
            raise ValueError(f"Column {column} not found")
        
        df_lagged = self.df.copy()
        
        for lag in lags:
            df_lagged[f'{column}_lag{lag}'] = df_lagged[column].shift(lag)
        
        return df_lagged
    
    def add_rolling_features(self, column: str, windows: List[int]) -> pd.DataFrame:
        """
        移動統計特徴量を追加
        
        Args:
            column: 対象カラム
            windows: ウィンドウサイズのリスト（例: [24, 168]）
        
        Returns:
            特徴量追加後のDataFrame
        """
        if column not in self.df.columns:
            raise ValueError(f"Column {column} not found")
        
        df_rolled = self.df.copy()
        
        for window in windows:
            df_rolled[f'{column}_rolling_mean_{window}'] = df_rolled[column].rolling(window).mean()
            df_rolled[f'{column}_rolling_std_{window}'] = df_rolled[column].rolling(window).std()
            df_rolled[f'{column}_rolling_min_{window}'] = df_rolled[column].rolling(window).min()
            df_rolled[f'{column}_rolling_max_{window}'] = df_rolled[column].rolling(window).max()
        
        return df_rolled
    
    def normalize(self, columns: Optional[List[str]] = None, 
                 method: str = 'zscore') -> pd.DataFrame:
        """
        データを正規化
        
        Args:
            columns: 対象カラム（Noneの場合は全数値カラム）
            method: 'zscore' or 'minmax'
        
        Returns:
            正規化後のDataFrame
        """
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns.tolist()
        
        df_normalized = self.df.copy()
        
        for col in columns:
            if col not in df_normalized.columns:
                continue
            
            if method == 'zscore':
                mean = df_normalized[col].mean()
                std = df_normalized[col].std()
                if std != 0:
                    df_normalized[col] = (df_normalized[col] - mean) / std
            
            elif method == 'minmax':
                min_val = df_normalized[col].min()
                max_val = df_normalized[col].max()
                if max_val != min_val:
                    df_normalized[col] = (df_normalized[col] - min_val) / (max_val - min_val)
            
            else:
                raise ValueError(f"Unknown method: {method}")
        
        return df_normalized
    
    def get_processed_data(self) -> pd.DataFrame:
        """
        現在のDataFrameを取得
        
        Returns:
            DataFrame
        """
        return self.df.copy()
