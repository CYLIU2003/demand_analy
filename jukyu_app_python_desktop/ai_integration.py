"""Utility classes to integrate transformer-based forecasting into the demand viewer.

The module keeps the PySide GUI lightweight by providing a compact training and
inference pipeline that can be invoked from the UI.  The implementation uses a
small Transformer encoder that consumes univariate time-series sequences and
produces multi-step forecasts.  All logic is written so that importing this
module does not immediately require PyTorch—callers should check
``TORCH_AVAILABLE`` before using the modelling APIs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

try:  # Optional dependency – the GUI shows a helpful prompt when missing.
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Dataset

    TORCH_AVAILABLE = True
except Exception:  # pragma: no cover - triggered when torch is not installed.
    torch = None  # type: ignore
    nn = None  # type: ignore
    DataLoader = Dataset = None  # type: ignore
    TORCH_AVAILABLE = False


@dataclass(frozen=True)
class ForecastConfig:
    """Hyper-parameters for the Transformer forecaster."""

    input_window: int = 48
    forecast_horizon: int = 24
    epochs: int = 30
    batch_size: int = 32
    learning_rate: float = 1e-3
    d_model: int = 64
    nhead: int = 4
    num_layers: int = 2
    dim_feedforward: int = 128
    dropout: float = 0.1


@dataclass
class ForecastResult:
    """Container with the forecast outputs and evaluation metrics."""

    target_name: str
    forecast_index: Sequence[pd.Timestamp]
    predicted: np.ndarray
    actual: Optional[np.ndarray]
    context_index: Sequence[pd.Timestamp]
    context_values: np.ndarray
    loss_history: List[float] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)


class DemandSequenceDataset(Dataset):
    """Sliding-window dataset for sequence-to-sequence forecasting."""

    def __init__(
        self,
        values: np.ndarray,
        input_window: int,
        forecast_horizon: int,
    ) -> None:
        if not TORCH_AVAILABLE:  # pragma: no cover - safety guard
            raise RuntimeError("PyTorch is required to build the dataset")

        self.inputs: List[np.ndarray] = []
        self.targets: List[np.ndarray] = []

        end = len(values) - input_window - forecast_horizon + 1
        for start in range(end):
            input_slice = values[start : start + input_window]
            target_slice = values[
                start + input_window : start + input_window + forecast_horizon
            ]
            self.inputs.append(input_slice[:, None])  # (seq_len, 1)
            self.targets.append(target_slice)  # (forecast_horizon,)

        if not self.inputs:
            raise ValueError(
                "Insufficient samples – increase the dataset size or adjust windows."
            )

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, idx: int):  # type: ignore[override]
        x = torch.as_tensor(self.inputs[idx], dtype=torch.float32)
        y = torch.as_tensor(self.targets[idx], dtype=torch.float32)
        return x, y


class PositionalEncoding(nn.Module):
    """Classic sinusoidal positional encoding."""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 10000) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-np.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)


class DemandTransformerModel(nn.Module):
    """Minimal Transformer encoder for univariate time-series forecasting."""

    def __init__(self, config: ForecastConfig) -> None:
        super().__init__()
        self.config = config
        self.input_projection = nn.Linear(1, config.d_model)
        self.positional_encoding = PositionalEncoding(config.d_model, config.dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            batch_first=False,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)
        self.output_layer = nn.Linear(config.d_model, config.forecast_horizon)

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        """src: (seq_len, batch, 1) -> forecast: (batch, horizon)"""

        projected = self.input_projection(src)
        encoded = self.encoder(self.positional_encoding(projected))
        last_timestep = encoded[-1]  # (batch, d_model)
        forecast = self.output_layer(last_timestep)
        return forecast


def _ensure_timestamps(
    timestamps: Optional[Iterable], length: int
) -> pd.DatetimeIndex:
    if timestamps is None:
        return pd.date_range("2000-01-01", periods=length, freq="H")
    ts = pd.to_datetime(pd.Index(list(timestamps)))
    if len(ts) != length:
        raise ValueError("Timestamp count does not match value count")
    return pd.DatetimeIndex(ts)


def run_transformer_forecast(
    series: Sequence[float] | pd.Series,
    timestamps: Optional[Iterable] = None,
    *,
    target_name: str = "需要",
    config: Optional[ForecastConfig] = None,
) -> ForecastResult:
    """Train the Transformer and return the forecast for the hold-out horizon."""

    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is not installed – install torch to use AI forecasting.")

    cfg = config or ForecastConfig()

    values = pd.Series(series).dropna().astype(np.float32).to_numpy()
    if values.ndim != 1:
        raise ValueError("Series must be one-dimensional")

    if len(values) <= cfg.input_window + cfg.forecast_horizon:
        raise ValueError(
            "データ数が少なすぎます。入力ウィンドウと予測期間を小さくしてください。"
        )

    index = _ensure_timestamps(timestamps, len(values))

    train_values = values[: -cfg.forecast_horizon]
    holdout_values = values[-cfg.forecast_horizon :]

    mean = float(train_values.mean())
    std = float(train_values.std())
    if std == 0.0:
        std = 1.0

    norm_values = (values - mean) / std
    norm_train = norm_values[: -cfg.forecast_horizon]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = DemandSequenceDataset(norm_train, cfg.input_window, cfg.forecast_horizon)
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, drop_last=False)

    model = DemandTransformerModel(cfg).to(device)
    criterion = nn.MSELoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)

    loss_history: List[float] = []
    model.train()
    for epoch in range(cfg.epochs):
        total_loss = 0.0
        batches = 0
        for batch_inputs, batch_targets in loader:
            optimiser.zero_grad()
            batch_inputs = batch_inputs.permute(1, 0, 2).to(device)  # (seq, batch, 1)
            batch_targets = batch_targets.to(device)
            outputs = model(batch_inputs)
            loss = criterion(outputs, batch_targets)
            loss.backward()
            optimiser.step()
            total_loss += float(loss.item())
            batches += 1

        if batches:
            loss_history.append(total_loss / batches)

    model.eval()
    with torch.no_grad():
        context_start = len(values) - cfg.forecast_horizon - cfg.input_window
        context_end = len(values) - cfg.forecast_horizon
        context_values = values[context_start:context_end]
        context_norm = (context_values - mean) / std
        input_tensor = torch.as_tensor(context_norm[:, None], dtype=torch.float32)
        input_tensor = input_tensor.permute(1, 0, 2).to(device)  # (seq, batch, feat)
        prediction_norm = model(input_tensor).cpu().numpy()[0]

    predicted = prediction_norm * std + mean
    forecast_index = index[-cfg.forecast_horizon :]
    context_index = index[context_start:context_end]

    mae = float(np.mean(np.abs(predicted - holdout_values)))
    rmse = float(np.sqrt(np.mean((predicted - holdout_values) ** 2)))
    with np.errstate(divide="ignore", invalid="ignore"):
        mape = np.abs((predicted - holdout_values) / holdout_values) * 100
        mape = float(np.nanmean(mape))

    metrics = {"MAE": mae, "RMSE": rmse, "MAPE": mape}

    return ForecastResult(
        target_name=target_name,
        forecast_index=forecast_index,
        predicted=predicted,
        actual=holdout_values,
        context_index=context_index,
        context_values=context_values,
        loss_history=loss_history,
        metrics=metrics,
    )


__all__ = [
    "ForecastConfig",
    "ForecastResult",
    "TORCH_AVAILABLE",
    "run_transformer_forecast",
]

