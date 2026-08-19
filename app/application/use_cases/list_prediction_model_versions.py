from __future__ import annotations

from app.domain.ports.prediction_logger import PredictionLogger


class ListPredictionModelVersions:
    """Versões de modelo com pelo menos 1 predição registrada — fonte do seletor "Geral/por modelo"
    da tela Performance & Error Matrix."""

    def __init__(self, prediction_logger: PredictionLogger):
        self._logger = prediction_logger

    def execute(self) -> list[str]:
        return self._logger.distinct_model_versions()
