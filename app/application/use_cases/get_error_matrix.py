from __future__ import annotations

import numpy as np
import pandas as pd

from src.evaluation.error_matrix import compute_metrics, error_matrix_by

from app.domain.ports.prediction_logger import PredictionLogger


class GetErrorMatrix:
    """Error Matrix real (reusa src/evaluation/error_matrix.py, P5) sobre predições com feedback."""

    def __init__(self, prediction_logger: PredictionLogger):
        self._logger = prediction_logger

    def execute(self, model_version: str | None = None) -> dict:
        predictions = self._logger.list_with_feedback(model_version=model_version)
        if not predictions:
            return {
                "n": 0,
                "note": "nenhuma predição com feedback (preço real) registrada ainda",
                "global": None,
                "by_price_band": [],
                "by_property_cluster": [],
            }

        df = pd.DataFrame([
            {
                "y_true": p.actual_price,
                "y_pred": p.predicted_price,
                "price_band": p.price_band,
                "property_cluster": p.property_cluster,
            }
            for p in predictions
        ])
        global_metrics = compute_metrics(df["y_true"].to_numpy(), df["y_pred"].to_numpy())
        by_band = error_matrix_by(df, "price_band", "y_true", "y_pred")
        by_cluster = error_matrix_by(df, "property_cluster", "y_true", "y_pred")
        return {
            "n": len(df),
            "global": global_metrics,
            "by_price_band": by_band.to_dict(orient="records"),
            "by_property_cluster": by_cluster.to_dict(orient="records"),
        }
