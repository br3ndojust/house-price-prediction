from __future__ import annotations

from typing import Protocol

from app.domain.entities.drift_report import DriftReport


class DriftEvaluator(Protocol):
    def evaluate(self) -> DriftReport:
        """PSI por feature (predições recentes vs. TRAIN) + distância média no espaço de features
        (reusa src/evaluation/geographic_coverage.py::feature_space_coverage, fase 04). Persiste o
        relatório gerado."""
        ...

    def list_reports(self) -> list[DriftReport]:
        ...

    def latest_report(self) -> DriftReport | None:
        ...
