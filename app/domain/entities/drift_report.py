from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass(slots=True)
class DriftReport:
    id: str
    n_samples: int
    psi_by_feature: dict[str, float]
    overall_status: str  # stable | moderate | significant
    mean_feature_space_distance: float | None = None
    prediction_drift: dict | None = None
    model_version: str | None = None
    created_at: datetime = field(default_factory=datetime.utcnow)

    PSI_MODERATE = 0.1
    PSI_SIGNIFICANT = 0.25

    @staticmethod
    def classify(max_psi: float) -> str:
        if max_psi >= DriftReport.PSI_SIGNIFICANT:
            return "significant"
        if max_psi >= DriftReport.PSI_MODERATE:
            return "moderate"
        return "stable"
