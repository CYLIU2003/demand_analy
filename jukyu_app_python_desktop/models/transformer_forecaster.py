"""
Transformer時系列予測モデル
電力需給予測のためのTransformerベースのニューラルネットワーク
PyTorchを使用して実装
"""

from typing import Any, Dict, Optional, Tuple
import numpy as np
import pandas as pd
from pathlib import Path

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch is not installed. Transformer model will not be available.")

from .model_interface import ModelInterface


if TORCH_AVAILABLE:
    class PositionalEncoding(nn.Module):
        """位置エンコーディング"""
        
        def __init__(self, d_model: int, max_len: int = 5000):
            super().__init__()
            
            position = torch.arange(max_len).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
            
            pe = torch.zeros(max_len, 1, d_model)
            pe[:, 0, 0::2] = torch.sin(position * div_term)
            pe[:, 0, 1::2] = torch.cos(position * div_term)
            
            self.register_buffer('pe', pe)
        
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            Args:
                x: shape (seq_len, batch, d_model)
            """
            return x + self.pe[:x.size(0)]
    
    
    class TransformerTimeSeries(nn.Module):
        """Transformer時系列予測モデル"""
        
        def __init__(self, input_dim: int, d_model: int = 64, nhead: int = 4,
                    num_encoder_layers: int = 2, dim_feedforward: int = 256,
                    dropout: float = 0.1):
            super().__init__()
            
            self.input_projection = nn.Linear(input_dim, d_model)
            self.pos_encoder = PositionalEncoding(d_model)
            
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=False
            )
            
            self.transformer_encoder = nn.TransformerEncoder(
                encoder_layer,
                num_layers=num_encoder_layers
            )
            
            self.output_projection = nn.Linear(d_model, 1)
            self.d_model = d_model
        
        def forward(self, src: torch.Tensor) -> torch.Tensor:
            """
            Args:
                src: shape (batch, seq_len, input_dim)
            Returns:
                shape (batch, 1)
            """
            # (batch, seq_len, input_dim) -> (seq_len, batch, input_dim)
            src = src.transpose(0, 1)
            
            # 入力射影
            src = self.input_projection(src) * np.sqrt(self.d_model)
            
            # 位置エンコーディング
            src = self.pos_encoder(src)
            
            # Transformer encoder
            output = self.transformer_encoder(src)
            
            # 最後のタイムステップを使用
            # (seq_len, batch, d_model) -> (batch, d_model)
            output = output[-1, :, :]
            
            # 出力射影
            output = self.output_projection(output)
            
            return output


class TransformerForecaster(ModelInterface):
    """
    Transformer時系列予測器
    電力需給データの予測に特化したTransformerモデル
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Args:
            config: モデル設定
                - d_model: Transformerの隠れ層次元数（デフォルト: 64）
                - nhead: アテンションヘッド数（デフォルト: 4）
                - num_encoder_layers: エンコーダ層数（デフォルト: 2）
                - dim_feedforward: フィードフォワード層の次元数（デフォルト: 256）
                - dropout: ドロップアウト率（デフォルト: 0.1）
                - learning_rate: 学習率（デフォルト: 0.001）
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for TransformerForecaster. Please install torch.")
        
        super().__init__(config)
        
        self.d_model = self.config.get('d_model', 64)
        self.nhead = self.config.get('nhead', 4)
        self.num_encoder_layers = self.config.get('num_encoder_layers', 2)
        self.dim_feedforward = self.config.get('dim_feedforward', 256)
        self.dropout = self.config.get('dropout', 0.1)
        self.learning_rate = self.config.get('learning_rate', 0.001)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.scaler_mean = None
        self.scaler_std = None
    
    def prepare_data(self, data: pd.DataFrame, target_column: str,
                    sequence_length: int = 24, 
                    feature_columns: Optional[list] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        時系列データをシーケンスデータに変換
        
        Args:
            data: 入力データフレーム
            target_column: 予測対象のカラム
            sequence_length: 入力シーケンスの長さ
            feature_columns: 使用する特徴量カラム（Noneの場合はtarget_columnのみ）
        
        Returns:
            (X, y) のタプル
            X: shape (samples, sequence_length, features)
            y: shape (samples, 1)
        """
        if feature_columns is None:
            feature_columns = [target_column]
        
        # 特徴量を抽出
        features = data[feature_columns].values
        
        # 正規化
        self.scaler_mean = np.mean(features, axis=0)
        self.scaler_std = np.std(features, axis=0)
        self.scaler_std[self.scaler_std == 0] = 1  # ゼロ除算回避
        
        features_normalized = (features - self.scaler_mean) / self.scaler_std
        
        # シーケンスデータを作成
        X, y = [], []
        target_idx = feature_columns.index(target_column)
        
        for i in range(len(features_normalized) - sequence_length):
            X.append(features_normalized[i:i+sequence_length])
            y.append(features[i+sequence_length, target_idx])  # 正規化前の値を使用
        
        return np.array(X), np.array(y).reshape(-1, 1)
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
             X_val: Optional[np.ndarray] = None,
             y_val: Optional[np.ndarray] = None,
             epochs: int = 100, batch_size: int = 32) -> Dict[str, list]:
        """
        モデルを訓練
        
        Args:
            X_train: shape (samples, sequence_length, features)
            y_train: shape (samples, 1)
            X_val: 検証データ（オプション）
            y_val: 検証データのターゲット（オプション）
            epochs: エポック数
            batch_size: バッチサイズ
        
        Returns:
            訓練履歴
        """
        input_dim = X_train.shape[2]
        
        # モデルを初期化
        self.model = TransformerTimeSeries(
            input_dim=input_dim,
            d_model=self.d_model,
            nhead=self.nhead,
            num_encoder_layers=self.num_encoder_layers,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout
        ).to(self.device)
        
        # データローダーを作成
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train),
            torch.FloatTensor(y_train)
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        if X_val is not None and y_val is not None:
            val_dataset = TensorDataset(
                torch.FloatTensor(X_val),
                torch.FloatTensor(y_val)
            )
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        else:
            val_loader = None
        
        # 最適化器と損失関数
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.MSELoss()
        
        # 訓練履歴
        history = {
            'train_loss': [],
            'val_loss': []
        }
        
        # 訓練ループ
        for epoch in range(epochs):
            # 訓練フェーズ
            self.model.train()
            train_loss = 0.0
            
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                optimizer.zero_grad()
                
                # 予測
                y_pred = self.model(X_batch)
                
                # ターゲットを正規化
                y_batch_normalized = (y_batch - self.scaler_mean[0]) / self.scaler_std[0]
                
                loss = criterion(y_pred, y_batch_normalized)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item() * X_batch.size(0)
            
            train_loss /= len(train_loader.dataset)
            history['train_loss'].append(train_loss)
            
            # 検証フェーズ
            if val_loader is not None:
                self.model.eval()
                val_loss = 0.0
                
                with torch.no_grad():
                    for X_batch, y_batch in val_loader:
                        X_batch = X_batch.to(self.device)
                        y_batch = y_batch.to(self.device)
                        
                        y_pred = self.model(X_batch)
                        y_batch_normalized = (y_batch - self.scaler_mean[0]) / self.scaler_std[0]
                        
                        loss = criterion(y_pred, y_batch_normalized)
                        val_loss += loss.item() * X_batch.size(0)
                
                val_loss /= len(val_loader.dataset)
                history['val_loss'].append(val_loss)
            
            # 進捗表示
            if (epoch + 1) % 10 == 0:
                if val_loader is not None:
                    print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
                else:
                    print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}")
        
        self.is_trained = True
        self.training_history = history
        
        return history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        予測を実行
        
        Args:
            X: shape (samples, sequence_length, features)
        
        Returns:
            予測値: shape (samples, 1)
        """
        if not self.is_trained:
            raise RuntimeError("Model is not trained yet.")
        
        self.model.eval()
        
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        with torch.no_grad():
            y_pred_normalized = self.model(X_tensor).cpu().numpy()
        
        # 逆正規化
        y_pred = y_pred_normalized * self.scaler_std[0] + self.scaler_mean[0]
        
        return y_pred
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        モデルを評価
        
        Args:
            X_test: テストデータ（入力）
            y_test: テストデータ（ターゲット）
        
        Returns:
            評価指標
        """
        y_pred = self.predict(X_test)
        metrics = self.calculate_metrics(y_test, y_pred)
        
        return metrics
    
    def save_model(self, filepath: str) -> None:
        """
        モデルを保存
        
        Args:
            filepath: 保存先のファイルパス
        """
        if not self.is_trained:
            raise RuntimeError("Model is not trained yet.")
        
        save_dict = {
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'scaler_mean': self.scaler_mean,
            'scaler_std': self.scaler_std,
            'training_history': self.training_history
        }
        
        torch.save(save_dict, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """
        モデルを読み込み
        
        Args:
            filepath: モデルファイルのパス
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.config = checkpoint['config']
        self.scaler_mean = checkpoint['scaler_mean']
        self.scaler_std = checkpoint['scaler_std']
        self.training_history = checkpoint.get('training_history', {})
        
        # モデルを再構築
        input_dim = len(self.scaler_mean)
        self.model = TransformerTimeSeries(
            input_dim=input_dim,
            d_model=self.config.get('d_model', 64),
            nhead=self.config.get('nhead', 4),
            num_encoder_layers=self.config.get('num_encoder_layers', 2),
            dim_feedforward=self.config.get('dim_feedforward', 256),
            dropout=self.config.get('dropout', 0.1)
        ).to(self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.is_trained = True
        
        print(f"Model loaded from {filepath}")
