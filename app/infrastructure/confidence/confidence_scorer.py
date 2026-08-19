"""Adapter do port ConfidenceService — carrega `artifacts/confidence_calibration.json` (congelado por
`src/scripts/build_confidence_matrix.py`: TEST calibra, VAL verifica uma única vez) e reusa as MESMAS
funções de `src/evaluation/confidence.py` usadas na calibração (P5: nenhuma lógica duplicada) para
pontuar uma predição nova em tempo de requisição.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.evaluation.confidence import (
    apply_coverage_bucket,
    categorize,
    confidence_score,
    expected_error_for_row,
    hard_segment_flag,
    segment_keys,
)
from src.evaluation.geographic_coverage import FeatureSpaceIndex, apply_feature_space_distance, fit_feature_space_index

from app.core.config import settings
from app.domain.entities.prediction_confidence import PredictionConfidence


class ConfidenceScorer:
    def __init__(self, artifacts_dir=None, data_dir=None):
        self._artifacts_dir = artifacts_dir or settings.artifacts_dir
        self._data_dir = data_dir or settings.data_dir
        self._artifact: dict | None = None
        self._train_df: pd.DataFrame | None = None
        self._ecdf_reference: np.ndarray | None = None
        self._feature_space_index: FeatureSpaceIndex | None = None
        self._load()

    def _load(self) -> None:
        path = self._artifacts_dir / "confidence_calibration.json"
        if not path.exists():
            self._artifact = None
            return
        self._artifact = json.loads(path.read_text(encoding="utf-8"))
        self._ecdf_reference = np.array(self._artifact["ecdf_reference"])

        features_path = self._data_dir / "trusted" / "features_contextual.parquet"
        df = pd.read_parquet(features_path)
        self._train_df = df[df["split"] == "train"].reset_index(drop=True)
        # fit 1x por processo (não por requisição) — refitar a árvore de vizinhos a cada predição
        # era o que causava timeout em lotes grandes (o(n_lote) fits de uma árvore com ~15 mil pontos)
        self._feature_space_index = fit_feature_space_index(self._train_df, self._artifact["feature_cols"])

    def is_ready(self) -> bool:
        return self._artifact is not None

    def score(self, model_features_df: pd.DataFrame, price_band: str, property_cluster: int,
              waterfront: int, grade: int) -> PredictionConfidence:
        if not self.is_ready():
            raise ValueError(
                "artifacts/confidence_calibration.json não encontrado — rode "
                "`python src/scripts/build_confidence_matrix.py` antes de pedir confiança."
            )
        artifact = self._artifact

        distance = apply_feature_space_distance(self._feature_space_index, model_features_df)
        coverage_bucket = apply_coverage_bucket(distance, artifact["coverage_bucket_edges"])[0]

        row = pd.DataFrame([{
            "price_band": price_band, "property_cluster": property_cluster,
            "waterfront": waterfront, "grade": grade,
        }])
        keys = segment_keys(row, np.array([coverage_bucket]))
        expected = expected_error_for_row(
            keys["level0"].iloc[0], keys["level1"].iloc[0], keys["level2"].iloc[0],
            artifact["segment_lookup"],
        )

        score_value = float(confidence_score(expected["median_ape"], self._ecdf_reference)[0])
        category = categorize(score_value, artifact["category_thresholds"])

        return PredictionConfidence(
            score=round(score_value, 1),
            category=category,
            expected_ape=expected["median_ape"],
            expected_abs_error=expected["median_abs_error"],
            feature_space_distance=float(distance[0]),
            coverage_bucket=str(coverage_bucket),
            segment_n=expected["n"],
            low_sample=expected["low_sample"],
            hard_segment=bool(hard_segment_flag(row).iloc[0]),
            segment_level_used=expected["level_used"],
        )

    def calibration_summary(self) -> dict:
        if not self.is_ready():
            return {"ready": False}
        artifact = self._artifact
        return {
            "ready": True,
            "methodology": artifact["methodology"],
            "category_thresholds": artifact["category_thresholds"],
            "category_labels_high_to_low": artifact["category_labels_high_to_low"],
            "ape_breakpoints_from_test": artifact["ape_breakpoints_from_test"],
            "min_segment_n": artifact["min_segment_n"],
            "coverage_bucket_edges": artifact["coverage_bucket_edges"],
        }
