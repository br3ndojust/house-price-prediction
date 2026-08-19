from __future__ import annotations

from pydantic import BaseModel, Field


class FeedbackIn(BaseModel):
    prediction_id: str = Field(..., description="Id retornado por POST /predictions")
    actual_price: float = Field(..., gt=0, description="Preço de venda real do imóvel (dólares)")


class FeedbackOut(BaseModel):
    status: str = Field(..., examples=["recorded"])
    auto_retrain_job_id: str | None = Field(
        None, description="Preenchido quando este feedback cruzou o limite do gatilho automático e disparou um retraining"
    )
