from __future__ import annotations

from app.domain.entities.drift_report import DriftReport
from app.domain.ports.drift_evaluator import DriftEvaluator


class EvaluateDrift:
    def __init__(self, drift_evaluator: DriftEvaluator):
        self._evaluator = drift_evaluator

    def execute(self) -> DriftReport:
        return self._evaluator.evaluate()


class ListDriftReports:
    def __init__(self, drift_evaluator: DriftEvaluator):
        self._evaluator = drift_evaluator

    def execute(self) -> list[DriftReport]:
        return self._evaluator.list_reports()


class GetLatestDriftReport:
    def __init__(self, drift_evaluator: DriftEvaluator):
        self._evaluator = drift_evaluator

    def execute(self) -> DriftReport | None:
        return self._evaluator.latest_report()
