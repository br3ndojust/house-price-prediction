from __future__ import annotations

from app.domain.entities.feature_importance import FeatureImportance
from app.domain.ports.explainability_service import ExplainabilityService


class GetGlobalFeatureImportance:
    def __init__(self, explainability_service: ExplainabilityService):
        self._explainability = explainability_service

    def execute(self) -> list[FeatureImportance]:
        return self._explainability.global_importance()
