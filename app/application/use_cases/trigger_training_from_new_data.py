"""Fase 16 (pós-fechamento) — dispara retraining com dado novo rotulado (venda real), somado ao TRAIN
oficial — docs/08_continuous_learning.md, seções 2-3. Nunca ativa o candidato sozinho (P6)."""
from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd

from src.pipelines.training_data import build_labeled_rows

from app.domain.entities.property import Property
from app.domain.entities.training_job import TrainingJob
from app.domain.ports.model_repository import ModelRepository
from app.domain.ports.prediction_logger import PredictionLogger
from app.domain.ports.training_service import TrainingService


def _labeled_rows_from(raw_df: pd.DataFrame, actual_prices: np.ndarray, models: ModelRepository,
                        test_size: float = 0.2) -> pd.DataFrame:
    materials = models.training_materials()
    if materials["price_breakpoints"] is None or materials["cluster_scaler"] is None:
        raise ValueError(
            "artefatos de segmentação (price_quartiles.json / property_cluster_model.pkl) não "
            "disponíveis — não é possível rotular banda/cluster do dado novo."
        )
    return build_labeled_rows(
        raw_df, actual_prices, materials["demographics"], materials["spatial_index"],
        materials["price_breakpoints"], materials["cluster_scaler"], materials["cluster_kmeans"],
        test_size=test_size,
    )


class TriggerTrainingFromFeedback:
    """Usa as predições que já receberam `POST /feedback` (preço de venda real) como dado novo."""

    def __init__(self, prediction_logger: PredictionLogger, model_repository: ModelRepository,
                 training_service: TrainingService):
        self._logger = prediction_logger
        self._models = model_repository
        self._training = training_service

    def execute(self, triggered_by: str | None = None,
                feature_cols: list[str] | None = None,
                metric: str | None = None,
                test_size: float = 0.2) -> tuple[TrainingJob, int]:
        predictions = self._logger.list_with_feedback()
        if not predictions:
            raise ValueError("nenhuma predição com valor real (feedback) registrada ainda")

        raw_df = pd.DataFrame([dataclasses.asdict(p.property) for p in predictions])
        actual_prices = np.array([p.actual_price for p in predictions], dtype=float)
        extra_rows = _labeled_rows_from(raw_df, actual_prices, self._models, test_size=test_size)

        job = self._training.create_job(triggered_by)
        self._training.stage_extra_rows(job.id, extra_rows)
        if feature_cols:
            self._training.stage_feature_cols(job.id, feature_cols)
        if metric:
            self._training.stage_metric(job.id, metric)
        return job, len(extra_rows)


class TriggerTrainingFromUpload:
    """Usa um lote de dado novo rotulado enviado via upload (CSV com atributos + preço real)."""

    def __init__(self, model_repository: ModelRepository, training_service: TrainingService):
        self._models = model_repository
        self._training = training_service

    def execute(self, raw_df: pd.DataFrame, actual_prices: np.ndarray, triggered_by: str | None = None,
                feature_cols: list[str] | None = None, metric: str | None = None,
                test_size: float = 0.2) -> tuple[TrainingJob, int]:
        if len(raw_df) == 0:
            raise ValueError("arquivo enviado não tem nenhuma linha válida")
        # valida cada linha com a mesma entidade de domínio usada em /predictions (P5, nunca duplica regra)
        for row in raw_df.to_dict(orient="records"):
            Property(**row)

        extra_rows = _labeled_rows_from(raw_df, actual_prices, self._models, test_size=test_size)
        job = self._training.create_job(triggered_by)
        self._training.stage_extra_rows(job.id, extra_rows)
        if feature_cols:
            self._training.stage_feature_cols(job.id, feature_cols)
        if metric:
            self._training.stage_metric(job.id, metric)
        return job, len(extra_rows)
