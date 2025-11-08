"""
データ処理ユーティリティ
研究用データの前処理、クリーニング、変換機能
"""

from typing import List, Optional, Tuple
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler


class DataProcessor:
    """電力需給データの前処理クラス"""
    
    def __init__(self):
        self.scaler = None
        self.scaler_type = None
    
    def handle_missing_values(self, data: pd.DataFrame, 
                             method: str = 'interpolate',
                             columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        欠損値の処理
        
        Args:
            data: データフレーム
            method: 処理方法 ('interpolate', 'forward_fill', 'backward_fill', 'drop', 'mean')
            columns: 処理対象のカラム（Noneの場合は全カラム）
        
        Returns:
            欠損値処理後のデータフレーム
        """
        df = data.copy()
        
        if columns is None:
            columns = df.columns.tolist()
        
        for col in columns:
            if col not in df.columns:
                continue
            
            if method == 'interpolate':
                df[col] = df[col].interpolate(method='linear', limit_direction='both')
            
            elif method == 'forward_fill':
                df[col] = df[col].fillna(method='ffill')
            
            elif method == 'backward_fill':
                df[col] = df[col].fillna(method='bfill')
            
            elif method == 'drop':
                df = df.dropna(subset=[col])
            
            elif method == 'mean':
                mean_value = df[col].mean()
                df[col] = df[col].fillna(mean_value)
            
            else:
                raise ValueError(f"Unknown method: {method}")
        
        return df
    
    def remove_outliers(self, data: pd.DataFrame, column: str,
                       method: str = 'iqr', threshold: float = 1.5) -> pd.DataFrame:
        """
        外れ値の除去
        
        Args:
            data: データフレーム
            column: 対象カラム
            method: 検出方法 ('iqr' or 'zscore')
            threshold: 閾値
        
        Returns:
            外れ値除去後のデータフレーム
        """
        df = data.copy()
        
        if column not in df.columns:
            raise ValueError(f"Column {column} not found")
        
        if method == 'iqr':
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
        
        elif method == 'zscore':
            z_scores = np.abs((df[column] - df[column].mean()) / df[column].std())
            df = df[z_scores <= threshold]
        
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return df
    
    def normalize_data(self, data: pd.DataFrame, 
                      columns: List[str],
                      method: str = 'standard') -> pd.DataFrame:
        """
        データの正規化
        
        Args:
            data: データフレーム
            columns: 正規化対象のカラム
            method: 'standard' (標準化) or 'minmax' (最小最大正規化)
        
        Returns:
            正規化後のデータフレーム
        """
        df = data.copy()
        
        valid_columns = [col for col in columns if col in df.columns]
        
        if not valid_columns:
            return df
        
        if method == 'standard':
            self.scaler = StandardScaler()
        elif method == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unknown method: {method}")
        
        self.scaler_type = method
        
        df[valid_columns] = self.scaler.fit_transform(df[valid_columns])
        
        return df
    
    def denormalize_data(self, data: pd.DataFrame,
                        columns: List[str]) -> pd.DataFrame:
        """
        正規化されたデータを元に戻す
        
        Args:
            data: 正規化されたデータフレーム
            columns: 逆変換対象のカラム
        
        Returns:
            元のスケールに戻したデータフレーム
        """
        if self.scaler is None:
            raise RuntimeError("No scaler has been fitted yet")
        
        df = data.copy()
        valid_columns = [col for col in columns if col in df.columns]
        
        if not valid_columns:
            return df
        
        df[valid_columns] = self.scaler.inverse_transform(df[valid_columns])
        
        return df
    
    def create_time_features(self, data: pd.DataFrame, 
                           time_column: str) -> pd.DataFrame:
        """
        時刻から特徴量を生成
        
        Args:
            data: データフレーム
            time_column: 時刻カラム
        
        Returns:
            特徴量を追加したデータフレーム
        """
        df = data.copy()
        
        if time_column not in df.columns:
            raise ValueError(f"Column {time_column} not found")
        
        # datetimeに変換
        df[time_column] = pd.to_datetime(df[time_column])
        
        # 各種時刻特徴量を生成
        df['hour'] = df[time_column].dt.hour
        df['day_of_week'] = df[time_column].dt.dayofweek  # 0=月曜, 6=日曜
        df['day_of_month'] = df[time_column].dt.day
        df['month'] = df[time_column].dt.month
        df['quarter'] = df[time_column].dt.quarter
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        
        # 周期的特徴量（サイン・コサイン変換）
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        return df
    
    def create_lag_features(self, data: pd.DataFrame,
                          column: str,
                          lags: List[int]) -> pd.DataFrame:
        """
        ラグ特徴量を生成
        
        Args:
            data: データフレーム
            column: 対象カラム
            lags: ラグのリスト（例: [1, 2, 24]）
        
        Returns:
            ラグ特徴量を追加したデータフレーム
        """
        df = data.copy()
        
        if column not in df.columns:
            raise ValueError(f"Column {column} not found")
        
        for lag in lags:
            df[f'{column}_lag_{lag}'] = df[column].shift(lag)
        
        return df
    
    def create_rolling_features(self, data: pd.DataFrame,
                               column: str,
                               windows: List[int]) -> pd.DataFrame:
        """
        移動統計量特徴量を生成
        
        Args:
            data: データフレーム
            column: 対象カラム
            windows: ウィンドウサイズのリスト（例: [24, 168]）
        
        Returns:
            移動統計量特徴量を追加したデータフレーム
        """
        df = data.copy()
        
        if column not in df.columns:
            raise ValueError(f"Column {column} not found")
        
        for window in windows:
            df[f'{column}_rolling_mean_{window}'] = df[column].rolling(window=window).mean()
            df[f'{column}_rolling_std_{window}'] = df[column].rolling(window=window).std()
            df[f'{column}_rolling_max_{window}'] = df[column].rolling(window=window).max()
            df[f'{column}_rolling_min_{window}'] = df[column].rolling(window=window).min()
        
        return df
    
    def split_train_val_test(self, data: pd.DataFrame,
                             train_ratio: float = 0.7,
                             val_ratio: float = 0.15,
                             test_ratio: float = 0.15) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        データを訓練/検証/テストセットに分割
        
        Args:
            data: データフレーム
            train_ratio: 訓練データの割合
            val_ratio: 検証データの割合
            test_ratio: テストデータの割合
        
        Returns:
            (train, val, test)のタプル
        """
        if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
            raise ValueError("Ratios must sum to 1.0")
        
        n = len(data)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        
        train_data = data.iloc[:train_end]
        val_data = data.iloc[train_end:val_end]
        test_data = data.iloc[val_end:]
        
        return train_data, val_data, test_data
    
    def resample_data(self, data: pd.DataFrame, time_column: str,
                     freq: str = 'H', agg_func: str = 'mean') -> pd.DataFrame:
        """
        データのリサンプリング
        
        Args:
            data: データフレーム
            time_column: 時刻カラム
            freq: リサンプリング頻度（'H': 時間, 'D': 日, 'W': 週など）
            agg_func: 集約関数 ('mean', 'sum', 'max', 'min')
        
        Returns:
            リサンプリング後のデータフレーム
        """
        df = data.copy()
        
        if time_column not in df.columns:
            raise ValueError(f"Column {time_column} not found")
        
        df[time_column] = pd.to_datetime(df[time_column])
        df = df.set_index(time_column)
        
        if agg_func == 'mean':
            df_resampled = df.resample(freq).mean()
        elif agg_func == 'sum':
            df_resampled = df.resample(freq).sum()
        elif agg_func == 'max':
            df_resampled = df.resample(freq).max()
        elif agg_func == 'min':
            df_resampled = df.resample(freq).min()
        else:
            raise ValueError(f"Unknown aggregation function: {agg_func}")
        
        df_resampled = df_resampled.reset_index()
        
        return df_resampled
    
    def detect_and_fix_anomalies(self, data: pd.DataFrame, column: str,
                                window: int = 24, threshold: float = 3.0) -> pd.DataFrame:
        """
        異常値を検出して修正
        
        Args:
            data: データフレーム
            column: 対象カラム
            window: 移動ウィンドウサイズ
            threshold: 標準偏差の倍数
        
        Returns:
            異常値を修正したデータフレーム
        """
        df = data.copy()
        
        if column not in df.columns:
            raise ValueError(f"Column {column} not found")
        
        # 移動平均と移動標準偏差を計算
        rolling_mean = df[column].rolling(window=window, center=True).mean()
        rolling_std = df[column].rolling(window=window, center=True).std()
        
        # 異常値を検出
        lower_bound = rolling_mean - threshold * rolling_std
        upper_bound = rolling_mean + threshold * rolling_std
        
        # 異常値を移動平均で置換
        df.loc[df[column] < lower_bound, column] = rolling_mean
        df.loc[df[column] > upper_bound, column] = rolling_mean
        
        return df
