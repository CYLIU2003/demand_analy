"""Lightweight transformer forecaster for electricity demand series."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset


def _infer_device() -> torch.device:
    """Return a CUDA device when available, otherwise fall back to CPU."""

    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class _SequenceDataset(Dataset[Tuple[torch.Tensor, torch.Tensor]]):
    """Sliding-window dataset producing (context, future) pairs."""

    def __init__(self, series: np.ndarray, context_length: int, prediction_length: int) -> None:
        if series.ndim != 1:
            raise ValueError("Series must be 1-dimensional")
        self.series = series.astype(np.float32)
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.window = context_length + prediction_length
        if len(self.series) < self.window:
            raise ValueError(
                "Series is shorter than the combined context and prediction length."
            )

    def __len__(self) -> int:
        return len(self.series) - self.window + 1

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        window = self.series[idx : idx + self.window]
        context = window[: self.context_length]
        future = window[self.context_length :]
        return (
            torch.from_numpy(context).unsqueeze(-1),
            torch.from_numpy(future),
        )


class _PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding for transformer inputs."""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, seq_len, d_model)
        seq_len = x.size(1)
        x = x + self.pe[:seq_len, :]
        return self.dropout(x)


class _DemandTransformerModel(nn.Module):
    """Encoder-only transformer producing a fixed-horizon forecast."""

    def __init__(
        self,
        context_length: int,
        prediction_length: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        feedforward_dim: int = 256,
    ) -> None:
        super().__init__()
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.d_model = d_model

        self.input_proj = nn.Linear(1, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pos_encoder = _PositionalEncoding(d_model, dropout)
        self.readout = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, prediction_length),
        )

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        if src.ndim != 3:
            raise ValueError("Expected src shape (batch, seq, features)")
        if src.size(1) != self.context_length:
            raise ValueError("Unexpected context length")
        x = self.input_proj(src)
        x = self.pos_encoder(x)
        encoded = self.encoder(x)
        pooled = encoded.mean(dim=1)
        return self.readout(pooled)

    def forecast(self, context: torch.Tensor) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            preds = self.forward(context)
        return preds


@dataclass
class ForecastTrainingLog:
    """Tracks training and validation losses for each epoch."""

    train_loss: List[float]
    val_loss: List[Optional[float]]


@dataclass
class ForecastResult:
    """Forecast output comprising the scaler-adjusted arrays."""

    history: np.ndarray
    prediction: np.ndarray
    training_log: ForecastTrainingLog


class DemandTransformerForecaster:
    """Convenience wrapper around the lightweight transformer forecaster."""

    def __init__(
        self,
        context_length: int,
        prediction_length: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        feedforward_dim: int = 256,
        learning_rate: float = 5e-4,
        batch_size: int = 32,
        epochs: int = 20,
        device: Optional[torch.device] = None,
    ) -> None:
        if context_length <= 0 or prediction_length <= 0:
            raise ValueError("context_length and prediction_length must be positive")
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = device or _infer_device()
        self.model = _DemandTransformerModel(
            context_length=context_length,
            prediction_length=prediction_length,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dropout=dropout,
            feedforward_dim=feedforward_dim,
        ).to(self.device)
        self.scaler = StandardScaler()
        self._fitted = False
        self._last_training_log: Optional[ForecastTrainingLog] = None

    def _build_dataloaders(
        self, series: np.ndarray, validation_split: float
    ) -> Tuple[DataLoader, Optional[DataLoader]]:
        if len(series) < self.context_length + self.prediction_length:
            raise ValueError(
                "Series is too short for the requested context/prediction length."
            )
        cutoff = int(len(series) * (1 - validation_split))
        cutoff = max(cutoff, self.context_length + self.prediction_length)
        train_series = series[:cutoff]
        val_series = series[cutoff - self.context_length - self.prediction_length :]
        train_dataset = _SequenceDataset(train_series, self.context_length, self.prediction_length)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader: Optional[DataLoader] = None
        if len(val_series) >= self.context_length + self.prediction_length:
            val_dataset = _SequenceDataset(val_series, self.context_length, self.prediction_length)
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        return train_loader, val_loader

    def fit(
        self,
        series: Sequence[float],
        validation_split: float = 0.2,
        verbose: bool = False,
        callback=None,
    ) -> ForecastTrainingLog:
        series = np.asarray(series, dtype=np.float32)
        if np.isnan(series).any():
            # Defer to pandas for interpolation without importing globally.
            import pandas as pd

            series = (
                pd.Series(series)
                .interpolate(limit_direction="both")
                .fillna(method="bfill")
                .fillna(method="ffill")
                .to_numpy(dtype=np.float32)
            )
        scaled = self.scaler.fit_transform(series.reshape(-1, 1)).flatten()
        train_loader, val_loader = self._build_dataloaders(scaled, validation_split)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        train_losses: List[float] = []
        val_losses: List[Optional[float]] = []

        for epoch in range(self.epochs):
            self.model.train()
            epoch_losses: List[float] = []
            for batch_context, batch_future in train_loader:
                batch_context = batch_context.to(self.device)
                batch_future = batch_future.to(self.device)
                optimizer.zero_grad()
                preds = self.model(batch_context)
                loss = criterion(preds, batch_future)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                epoch_losses.append(loss.item())
            epoch_train_loss = float(np.mean(epoch_losses)) if epoch_losses else float("nan")
            train_losses.append(epoch_train_loss)

            val_loss_value: Optional[float] = None
            if val_loader is not None:
                self.model.eval()
                val_epoch_losses: List[float] = []
                with torch.no_grad():
                    for batch_context, batch_future in val_loader:
                        batch_context = batch_context.to(self.device)
                        batch_future = batch_future.to(self.device)
                        preds = self.model(batch_context)
                        loss = criterion(preds, batch_future)
                        val_epoch_losses.append(loss.item())
                if val_epoch_losses:
                    val_loss_value = float(np.mean(val_epoch_losses))
            val_losses.append(val_loss_value)

            if verbose:
                msg = f"Epoch {epoch + 1}/{self.epochs} - train_loss={epoch_train_loss:.6f}"
                if val_loss_value is not None:
                    msg += f", val_loss={val_loss_value:.6f}"
                print(msg)
            
            if callback:
                callback(epoch + 1, self.epochs, epoch_train_loss, val_loss_value)

        log = ForecastTrainingLog(train_loss=train_losses, val_loss=val_losses)
        self._last_training_log = log
        self._fitted = True
        return log

    def predict(self, series: Sequence[float]) -> ForecastResult:
        if not self._fitted:
            raise RuntimeError("fit must be called before predict")
        series = np.asarray(series, dtype=np.float32)
        scaled = self.scaler.transform(series.reshape(-1, 1)).flatten()
        if len(scaled) < self.context_length:
            raise ValueError("Series shorter than context length")
        context = scaled[-self.context_length :]
        context_tensor = torch.from_numpy(context).unsqueeze(0).unsqueeze(-1).to(self.device)
        prediction_scaled = (
            self.model.forecast(context_tensor)
            .squeeze(0)
            .detach()
            .cpu()
            .numpy()
        )
        prediction = self.scaler.inverse_transform(prediction_scaled.reshape(-1, 1)).flatten()
        history = np.asarray(series[-self.context_length :], dtype=np.float32)
        result = ForecastResult(
            history=history,
            prediction=prediction,
            training_log=self._last_training_log
            if self._last_training_log is not None
            else ForecastTrainingLog(train_loss=[], val_loss=[]),
        )
        return result


__all__ = [
    "DemandTransformerForecaster",
    "ForecastResult",
    "ForecastTrainingLog",
]
