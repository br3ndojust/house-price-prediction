from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd

from app.application.dto.prediction_result import PredictionResult
from app.domain.entities.prediction import Prediction
from app.domain.entities.property import Property
from app.domain.ports.confidence_service import ConfidenceService
from app.domain.ports.model_repository import ModelRepository
from app.domain.ports.prediction_logger import PredictionLogger


class PredictPrice:
    def __init__(self, model_repository: ModelRepository, prediction_logger: PredictionLogger,
                 confidence_service: ConfidenceService | None = None):
        self._models = model_repository
        self._logger = prediction_logger
        self._confidence = confidence_service

    def execute(self, property_: Property) -> PredictionResult:
        raw_df = pd.DataFrame([dataclasses.asdict(property_)])
        model_features = self._models.build_model_features(raw_df)
        price = float(self._models.predict_from_features(model_features).iloc[0])

        price_log = pd.Series([np.log1p(price)])
        band = self._models.predict_price_band(price_log)[0]
        cluster = self._models.predict_property_cluster(model_features)[0]

        confidence = None
        if self._confidence is not None and self._confidence.is_ready():
            confidence = self._confidence.score(
                model_features, band, int(cluster), property_.waterfront, property_.grade
            )

        prediction = Prediction(
            property=property_,
            predicted_price=price,
            price_band=band,
            property_cluster=int(cluster),
            model_version=self._models.active_version(),
            confidence_score=confidence.score if confidence else None,
            confidence_category=confidence.category if confidence else None,
        )
        prediction_id = self._logger.log(prediction)
        return PredictionResult(
            prediction_id=prediction_id,
            predicted_price=price,
            price_band=band,
            property_cluster=int(cluster),
            model_version=prediction.model_version,
            confidence_score=prediction.confidence_score,
            confidence_category=prediction.confidence_category,
        )
