from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class ModelInfoOut(BaseModel):
    """Passthrough do contrato ativo (`production_contract.yaml` + estado dinâmico de reload)."""

    model_config = {"extra": "allow"}


class ModelVersionOut(BaseModel):
    version: str
    algorithm: str
    hyperparameters: dict
    artifact_path: str
    status: str
    cv_mae: float | None
    test_mae: float | None = None
    val_mae: float | None
    trained_on: str | None
    n_train_rows: int | None = None
    n_features: int | None = None
    feature_cols: list[str] | None = None
    triggered_by: str | None = None
    training_job_id: str | None
    recommended_for_promotion: bool | None = None
    promotion_eval: dict | None = None
    performance_metrics: dict | None = None
    created_at: datetime
    promoted_at: datetime | None
    notes: str | None


class PromoteIn(BaseModel):
    version: str = Field(..., description="Versão candidata a promover (ver GET /model/versions)")
    alias: str | None = Field(None, description="Apelido opcional para a versão (gravado em `notes`)")
