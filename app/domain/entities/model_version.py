from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass(slots=True)
class ModelVersion:
    version: str
    algorithm: str
    hyperparameters: dict
    artifact_path: str
    status: str  # candidate | active | rejected | superseded
    cv_mae: float | None = None
    test_mae: float | None = None
    val_mae: float | None = None
    trained_on: str | None = None
    n_train_rows: int | None = None
    n_features: int | None = None
    feature_cols: list[str] | None = None
    triggered_by: str | None = None
    training_job_id: str | None = None
    recommended_for_promotion: bool | None = None
    promotion_eval: dict | None = None
    performance_metrics: dict | None = None  # {"train"/"test"/"val": {"mae","rmse","mape","r2_log"}}
    created_at: datetime = field(default_factory=datetime.utcnow)
    promoted_at: datetime | None = None
    notes: str | None = None
