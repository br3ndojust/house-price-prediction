from __future__ import annotations

from typing import Protocol

import pandas as pd

from app.domain.entities.feature_importance import FeatureImportance
from app.domain.entities.prediction_explanation import PredictionExplanation


class ExplainabilityService(Protocol):
    def global_importance(self) -> list[FeatureImportance]:
        """Ranking global (gain-based, nativo do XGBoost)."""
        ...

    def explain_local(self, model_features_row: pd.DataFrame, predicted_price: float) -> PredictionExplanation:
        """SHAP local (shap.TreeExplainer) para uma única predição."""
        ...

    def feature_statistics(self) -> dict[str, dict]:
        """Correlação com `price` + quartis de cada feature oficial, sobre TRAIN."""
        ...

    def feature_sample(self, sample_size: int = 400) -> list[dict]:
        """Amostra de (feature..., price) de TRAIN — só para o gráfico de dispersão do portal."""
        ...
