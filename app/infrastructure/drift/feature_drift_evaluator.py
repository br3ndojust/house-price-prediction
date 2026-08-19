"""Adapter do port DriftEvaluator — PSI por feature (predições recentes vs. TRAIN) + distância média
no espaço de features, reusando `src/evaluation/geographic_coverage.py::feature_space_coverage`
(fase 04, P5: sem duplicar lógica de cobertura)."""
from __future__ import annotations

import dataclasses
import uuid

import numpy as np
import pandas as pd
from sqlalchemy import select

from src.evaluation.geographic_coverage import feature_space_coverage

from app.core.config import settings
from app.domain.entities.drift_report import DriftReport
from app.domain.ports.model_repository import ModelRepository
from app.domain.ports.prediction_logger import PredictionLogger
from app.infrastructure.persistence.db import session_scope
from app.infrastructure.persistence.model_version_repository import SqlModelVersionRepository
from app.infrastructure.persistence.models import DriftReportRecord

_MODEL_FEATURES = [
    "bedrooms", "bathrooms", "sqft_living", "sqft_lot", "floors", "waterfront", "view",
    "condition", "grade", "sqft_above", "sqft_basement", "sqft_living15", "sqft_lot15",
    "property_age", "dist_to_seattle_center", "comps_knn_price", "local_grade_percentile",
]


def _psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index — quantis de `expected` (TRAIN) definem os baldes."""
    quantiles = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    if len(quantiles) < 3:
        return 0.0
    edges = quantiles.copy()
    edges[0], edges[-1] = -np.inf, np.inf

    exp_counts, _ = np.histogram(expected, bins=edges)
    act_counts, _ = np.histogram(actual, bins=edges)
    exp_pct = np.maximum(exp_counts / max(len(expected), 1), 1e-4)
    act_pct = np.maximum(act_counts / max(len(actual), 1), 1e-4)
    return float(np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct)))


def _to_entity(record: DriftReportRecord) -> DriftReport:
    return DriftReport(
        id=record.id,
        n_samples=record.n_samples,
        psi_by_feature=record.psi_json,
        overall_status=record.overall_status,
        mean_feature_space_distance=record.mean_feature_space_distance,
        prediction_drift=record.prediction_drift_json,
        model_version=record.model_version,
        created_at=record.created_at,
    )


class FeatureDriftEvaluator:
    def __init__(
        self,
        model_repository: ModelRepository,
        prediction_logger: PredictionLogger,
        model_version_repository: SqlModelVersionRepository | None = None,
    ):
        self._models = model_repository
        self._logger = prediction_logger
        self._versions = model_version_repository
        self._train_df: pd.DataFrame | None = None

    def _active_model_version(self) -> str | None:
        # `get_active()` reflete a versão registrada em `model_versions` (mesma exibida em Registro de
        # Modelo) — mais confiável que `ModelRepository.active_version()`, que deriva do nome do
        # arquivo .pkl carregado e pode divergir do id da versão (ex.: candidato promovido).
        if self._versions is None:
            return None
        active = self._versions.get_active()
        return active.version if active else None

    def _train_reference(self) -> pd.DataFrame:
        if self._train_df is None:
            df = pd.read_parquet(settings.data_dir / "trusted" / "features_contextual.parquet")
            self._train_df = df[df["split"] == "train"].reset_index(drop=True)
        return self._train_df

    def evaluate(self) -> DriftReport:
        recent = self._logger.list_recent(limit=1000)
        train = self._train_reference()

        if len(recent) < settings.drift_min_samples:
            report = DriftReport(
                id=str(uuid.uuid4()),
                n_samples=len(recent),
                psi_by_feature={},
                overall_status="insufficient_data",
                mean_feature_space_distance=None,
                model_version=self._active_model_version(),
            )
            self._persist(report)
            return report

        raw_df = pd.DataFrame([dataclasses.asdict(p.property) for p in recent])
        features_df = self._models.build_model_features(raw_df)

        psi_by_feature = {
            f: round(_psi(train[f].to_numpy(dtype=float), features_df[f].to_numpy(dtype=float)), 4)
            for f in _MODEL_FEATURES if f in train.columns and f in features_df.columns
        }
        max_psi = max(psi_by_feature.values()) if psi_by_feature else 0.0

        coverage = feature_space_coverage(train, features_df.reset_index(drop=True), _MODEL_FEATURES)
        mean_distance = float(coverage["nearest_train_neighbor_distance"].mean())

        train_prices = train["price"].to_numpy(dtype=float)
        recent_prices = np.array([p.predicted_price for p in recent], dtype=float)
        prediction_drift = {
            "train_mean": float(train_prices.mean()),
            "train_median": float(np.median(train_prices)),
            "recent_mean": float(recent_prices.mean()),
            "recent_median": float(np.median(recent_prices)),
            "mean_change_pct": float((recent_prices.mean() - train_prices.mean()) / train_prices.mean()),
            "psi": round(_psi(train_prices, recent_prices), 4),
        }

        report = DriftReport(
            id=str(uuid.uuid4()),
            n_samples=len(recent),
            psi_by_feature=psi_by_feature,
            overall_status=DriftReport.classify(max_psi),
            mean_feature_space_distance=mean_distance,
            prediction_drift=prediction_drift,
            model_version=self._active_model_version(),
        )
        self._persist(report)
        return report

    def _persist(self, report: DriftReport) -> None:
        with session_scope() as session:
            session.add(
                DriftReportRecord(
                    id=report.id,
                    n_samples=report.n_samples,
                    psi_json=report.psi_by_feature,
                    overall_status=report.overall_status,
                    mean_feature_space_distance=report.mean_feature_space_distance,
                    prediction_drift_json=report.prediction_drift,
                    model_version=report.model_version,
                    created_at=report.created_at,
                )
            )
        from app.infrastructure.observability.metrics import DRIFT_STATUS_GAUGE

        DRIFT_STATUS_GAUGE.labels(status=report.overall_status).set(1)

    def list_reports(self) -> list[DriftReport]:
        with session_scope() as session:
            stmt = select(DriftReportRecord).order_by(DriftReportRecord.created_at.desc())
            return [_to_entity(r) for r in session.scalars(stmt).all()]

    def latest_report(self) -> DriftReport | None:
        with session_scope() as session:
            stmt = select(DriftReportRecord).order_by(DriftReportRecord.created_at.desc()).limit(1)
            record = session.scalars(stmt).first()
            return _to_entity(record) if record else None
