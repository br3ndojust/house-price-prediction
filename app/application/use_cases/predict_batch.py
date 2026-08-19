from __future__ import annotations

from app.application.dto.prediction_result import PredictionResult
from app.application.use_cases.predict_price import PredictPrice
from app.domain.entities.property import Property
from app.domain.ports.confidence_service import ConfidenceService
from app.domain.ports.model_repository import ModelRepository
from app.domain.ports.prediction_logger import PredictionLogger


class PredictBatch:
    def __init__(self, model_repository: ModelRepository, prediction_logger: PredictionLogger,
                 confidence_service: ConfidenceService | None = None):
        self._predict_one = PredictPrice(model_repository, prediction_logger, confidence_service)

    def execute(self, properties: list[Property]) -> list[PredictionResult]:
        return [self._predict_one.execute(p) for p in properties]
