from __future__ import annotations

from typing import Protocol

import pandas as pd

from app.domain.entities.prediction_confidence import PredictionConfidence


class ConfidenceService(Protocol):
    def score(self, raw_df: pd.DataFrame, model_features_df: pd.DataFrame) -> PredictionConfidence:
        """Calcula o Confidence Score (0-100) + categoria para 1 imóvel, usando a calibração congelada
        em `artifacts/confidence_calibration.json` (TEST calibra, VAL verifica uma única vez —
        ver `src/evaluation/confidence.py`)."""
        ...

    def is_ready(self) -> bool:
        ...

    def calibration_summary(self) -> dict:
        """Metodologia + cortes de categoria + quartis de APE do TEST — para `GET
        /model/confidence-calibration`."""
        ...
