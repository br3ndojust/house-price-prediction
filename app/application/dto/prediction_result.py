from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PredictionResult:
    prediction_id: str
    predicted_price: float
    price_band: str
    property_cluster: int
    model_version: str
    confidence_score: float | None = None
    confidence_category: str | None = None
