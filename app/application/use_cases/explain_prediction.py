from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd

from app.domain.entities.prediction_explanation import PredictionExplanation
from app.domain.entities.property import Property
from app.domain.ports.explainability_service import ExplainabilityService
from app.domain.ports.model_repository import ModelRepository


class ExplainPrediction:
    def __init__(self, model_repository: ModelRepository, explainability_service: ExplainabilityService):
        self._models = model_repository
        self._explainability = explainability_service

    def execute(self, property_: Property) -> PredictionExplanation:
        raw_df = pd.DataFrame([dataclasses.asdict(property_)])
        model_features = self._models.build_model_features(raw_df)
        price = float(self._models.predict_from_features(model_features).iloc[0])
        return self._explainability.explain_local(model_features, price)
