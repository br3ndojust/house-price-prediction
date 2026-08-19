from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel


class DriftReportOut(BaseModel):
    id: str
    n_samples: int
    psi_by_feature: dict[str, float]
    overall_status: str
    mean_feature_space_distance: float | None
    prediction_drift: dict | None = None
    model_version: str | None = None
    created_at: datetime
