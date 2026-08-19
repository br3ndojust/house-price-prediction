from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass(slots=True)
class AutoRetrainConfig:
    """Gatilho automático de retraining por volume de dado novo (feedback acumulado) — pipeline
    semi-automático: dispara sozinho, mas nunca promove sozinho (P6, aprovação humana continua
    obrigatória em `POST /model/promote`)."""

    threshold: int = 500
    enabled: bool = False
    consumed_feedback_count: int = 0
    updated_at: datetime = field(default_factory=datetime.utcnow)
