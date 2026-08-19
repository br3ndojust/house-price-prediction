from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class TrainingJobIn(BaseModel):
    feature_cols: list[str] | None = Field(
        None,
        description=(
            "Subconjunto das features oficiais a usar neste treino (ver GET /model/feature-importance "
            "para a lista completa). Omitido/null = usa as 20 features oficiais travadas."
        ),
    )
    metric: str | None = Field(
        None,
        description=(
            "Métrica usada para ranquear os candidatos deste treino: 'mae' (padrão/oficial), 'rmse', "
            "'mape' ou 'r2'. Omitido/null = usa 'mae', a métrica oficial de seleção (P4)."
        ),
    )
    test_size: float | None = Field(
        None,
        ge=0.0, le=1.0,
        description=(
            "Só se aplica a jobs com dado novo (from-feedback/from-upload): fração sorteada "
            "aleatoriamente para `split='test'`, o resto vai para `split='train'` — nunca 100% "
            "`train` como antes. Omitido/null = 0.2 (20%)."
        ),
    )


class TrainingJobOut(BaseModel):
    id: str
    status: str
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None
    result: dict | None
    error: str | None
    triggered_by: str | None


class AutoRetrainConfigOut(BaseModel):
    threshold: int
    enabled: bool
    consumed_feedback_count: int
    new_since_last: int = Field(description="Feedback novo acumulado desde o último disparo automático")
    updated_at: datetime


class AutoRetrainConfigIn(BaseModel):
    threshold: int | None = Field(None, ge=1, description="Nº de feedback novo acumulado que dispara o próximo retraining automático")
    enabled: bool | None = Field(None, description="Liga/desliga o gatilho automático (não afeta retraining manual)")
