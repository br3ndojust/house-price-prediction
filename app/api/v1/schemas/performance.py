from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel


class PredictionRecordOut(BaseModel):
    id: str
    created_at: datetime
    predicted_price: float
    price_band: str
    property_cluster: int
    model_version: str
    confidence_score: float | None = None
    confidence_category: str | None = None
    actual_price: float | None = None
    feedback_recorded_at: datetime | None = None


class PerformanceSummaryOut(BaseModel):
    total_predictions: int
    predictions_last_24h: int
    predictions_with_feedback: int
    by_price_band: dict[str, int]
    by_property_cluster: dict[str, int]
    by_confidence_category: dict[str, int]
    latency_note: str


class ErrorMatrixOut(BaseModel):
    n: int
    global_: dict | None = None
    by_price_band: list[dict]
    by_property_cluster: list[dict]
    note: str | None = None

    model_config = {"populate_by_name": True}
