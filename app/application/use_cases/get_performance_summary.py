from __future__ import annotations

from app.domain.ports.prediction_logger import PredictionLogger


class GetPerformanceSummary:
    def __init__(self, prediction_logger: PredictionLogger):
        self._logger = prediction_logger

    def execute(self, model_version: str | None = None) -> dict:
        return self._logger.summary(model_version=model_version)
