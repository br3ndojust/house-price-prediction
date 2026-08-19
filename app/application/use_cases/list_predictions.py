from __future__ import annotations

from app.domain.entities.prediction import Prediction
from app.domain.ports.prediction_logger import PredictionLogger


class ListPredictions:
    """Lista predições recentes — fonte de `GET /performance/predictions` (tabela de predições na
    tela Performance & Error Matrix do portal, incluindo o valor real quando já registrado via
    `POST /feedback`)."""

    def __init__(self, prediction_logger: PredictionLogger):
        self._logger = prediction_logger

    def execute(self, with_feedback: bool = False, limit: int = 100,
                model_version: str | None = None) -> list[Prediction]:
        if with_feedback:
            predictions = self._logger.list_with_feedback(model_version=model_version)
            return sorted(predictions, key=lambda p: p.created_at, reverse=True)[:limit]
        return self._logger.list_recent(limit=limit, model_version=model_version)
