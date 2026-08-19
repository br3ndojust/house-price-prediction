from __future__ import annotations

from app.domain.ports.explainability_service import ExplainabilityService


class GetFeatureStatistics:
    def __init__(self, explainability_service: ExplainabilityService):
        self._explainability = explainability_service

    def execute(self) -> dict[str, dict]:
        return self._explainability.feature_statistics()


class GetFeatureSample:
    def __init__(self, explainability_service: ExplainabilityService):
        self._explainability = explainability_service

    def execute(self, sample_size: int = 400) -> list[dict]:
        return self._explainability.feature_sample(sample_size)
