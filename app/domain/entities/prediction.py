from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from app.domain.entities.property import Property


@dataclass(slots=True)
class Prediction:
    property: Property
    predicted_price: float
    price_band: str
    property_cluster: int
    model_version: str
    confidence_score: float | None = None
    confidence_category: str | None = None
    id: str | None = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    actual_price: float | None = None
    feedback_recorded_at: datetime | None = None

    @property
    def has_feedback(self) -> bool:
        return self.actual_price is not None
