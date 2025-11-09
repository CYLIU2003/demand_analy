"""
AIモデルの基本インターフェース
学術研究用に標準化されたモデル操作を提供
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple
import numpy as np
import pandas as pd


class ModelInterface(ABC):
    """
    AIモデルの基底クラス
    全ての予測モデルはこのインターフェースを実装する
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Args:
            config: モデルの設定パラメータ
        """
        self.config = config or {}
        self.is_trained = False
        self.training_history = {}
    
    @abstractmethod
    def prepare_data(self, data: pd.DataFrame, target_column: str,
                    sequence_length: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        データの前処理
        
        Args:
            data: 入力データフレーム
            target_column: 予測対象のカラム
            sequence_length: シーケンス長
        
        Returns:
            (入力データ, ターゲットデータ)のタプル
        """
        pass
    
    @abstractmethod
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
             X_val: Optional[np.ndarray] = None,
             y_val: Optional[np.ndarray] = None,
             epochs: int = 100, batch_size: int = 32) -> Dict[str, list]:
        """
        モデルの訓練
        
        Args:
            X_train: 訓練データ（入力）
            y_train: 訓練データ（ターゲット）
            X_val: 検証データ（入力）
            y_val: 検証データ（ターゲット）
            epochs: エポック数
            batch_size: バッチサイズ
        
        Returns:
            訓練履歴（loss, metricsなど）
        """
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        予測を実行
        
        Args:
            X: 入力データ
        
        Returns:
            予測結果
        """
        pass
    
    @abstractmethod
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        モデルの評価
        
        Args:
            X_test: テストデータ（入力）
            y_test: テストデータ（ターゲット）
        
        Returns:
            評価指標の辞書（MAE, RMSE, MAPE等）
        """
        pass
    
    @abstractmethod
    def save_model(self, filepath: str) -> None:
        """
        モデルを保存
        
        Args:
            filepath: 保存先のファイルパス
        """
        pass
    
    @abstractmethod
    def load_model(self, filepath: str) -> None:
        """
        モデルを読み込み
        
        Args:
            filepath: モデルファイルのパス
        """
        pass
    
    def get_training_history(self) -> Dict[str, list]:
        """
        訓練履歴を取得
        
        Returns:
            訓練履歴の辞書
        """
        return self.training_history
    
    def calculate_metrics(self, y_true: np.ndarray, 
                         y_pred: np.ndarray) -> Dict[str, float]:
        """
        評価指標を計算
        
        Args:
            y_true: 真の値
            y_pred: 予測値
        
        Returns:
            MAE, RMSE, MAPE, R2などの評価指標
        """
        mae = np.mean(np.abs(y_true - y_pred))
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        
        # MAPE (Mean Absolute Percentage Error)
        # ゼロ除算を避けるため、真の値が0の場合は除外
        mask = y_true != 0
        if np.sum(mask) > 0:
            mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        else:
            mape = np.nan
        
        # R2 score
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else np.nan
        
        return {
            'MAE': mae,
            'RMSE': rmse,
            'MAPE': mape,
            'R2': r2
        }
