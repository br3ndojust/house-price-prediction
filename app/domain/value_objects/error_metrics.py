from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ErrorMetrics:
    """Espelha `src/evaluation/error_matrix.py::compute_metrics` — reuso de vocabulário, P5."""

    n: int
    mae: float
    rmse: float
    median_ae: float
    mape: float
    median_ape: float
    p90_ae: float
    bias: float
    low_sample: bool

    @classmethod
    def from_dict(cls, d: dict) -> "ErrorMetrics":
        return cls(**{k: d[k] for k in (
            "n", "mae", "rmse", "median_ae", "mape", "median_ape", "p90_ae", "bias", "low_sample"
        )})
