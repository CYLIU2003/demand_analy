"""Utilities for machine-learning powered demand forecasting."""

try:
    from .transformer import (
        DemandTransformerForecaster,
        ForecastResult,
        ForecastTrainingLog,
    )
    TRANSFORMER_AVAILABLE = True
except ImportError:
    # torch is not installed, transformer features will be disabled
    DemandTransformerForecaster = None
    ForecastResult = None
    ForecastTrainingLog = None
    TRANSFORMER_AVAILABLE = False

__all__ = [
    "DemandTransformerForecaster",
    "ForecastResult",
    "ForecastTrainingLog",
    "TRANSFORMER_AVAILABLE",
]
