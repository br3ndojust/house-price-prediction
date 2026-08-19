from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd

from app.domain.entities.prediction_confidence import PredictionConfidence
from app.domain.entities.property import Property
from app.domain.ports.confidence_service import ConfidenceService
from app.domain.ports.model_repository import ModelRepository


class GetPredictionConfidence:
    """Breakdown completo de confiança para 1 imóvel (não loga predição — mesma decisão de
    `ExplainPrediction`, é uma leitura/diagnóstico, não um novo evento de negócio)."""

    def __init__(self, model_repository: ModelRepository, confidence_service: ConfidenceService):
        self._models = model_repository
        self._confidence = confidence_service

    def execute(self, property_: Property) -> PredictionConfidence:
        if not self._confidence.is_ready():
            raise ValueError(
                "calibração de confiança não disponível — rode src/scripts/build_confidence_matrix.py"
            )
        raw_df = pd.DataFrame([dataclasses.asdict(property_)])
        model_features = self._models.build_model_features(raw_df)
        price = float(self._models.predict_from_features(model_features).iloc[0])
        price_log = pd.Series([np.log1p(price)])
        band = self._models.predict_price_band(price_log)[0]
        cluster = self._models.predict_property_cluster(model_features)[0]

        return self._confidence.score(model_features, band, int(cluster), property_.waterfront, property_.grade)


class GetConfidenceCalibrationSummary:
    def __init__(self, confidence_service: ConfidenceService):
        self._confidence = confidence_service

    def execute(self) -> dict:
        return self._confidence.calibration_summary()
