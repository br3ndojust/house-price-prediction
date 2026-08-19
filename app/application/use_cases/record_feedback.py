from __future__ import annotations

from app.domain.ports.feedback_repository import FeedbackRepository
from app.domain.ports.prediction_logger import PredictionLogger


class RecordFeedback:
    def __init__(self, feedback_repository: FeedbackRepository, prediction_logger: PredictionLogger):
        self._feedback = feedback_repository
        self._logger = prediction_logger

    def execute(self, prediction_id: str, actual_price: float) -> None:
        if self._logger.get(prediction_id) is None:
            raise ValueError(f"predição {prediction_id} não encontrada")
        if actual_price <= 0:
            raise ValueError("actual_price deve ser positivo")
        self._feedback.record(prediction_id, actual_price)
