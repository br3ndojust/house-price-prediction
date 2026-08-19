"""Fase 11 — pipeline fim-a-fim raw -> features -> model -> prediction (P5: paridade treino/produção).

Reusa as MESMAS funções de fase 01/05/06 usadas em treino — nenhuma lógica duplicada/reescrita para
produção (P5, P6: contrato de produção não pode divergir do experimental sem rastreabilidade).
"""
from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.data.merge import merge_demographics
from src.features.comparable import apply_comps_knn_price
from src.features.neighborhood import add_dist_to_seattle_center
from src.features.raw import add_property_age
from src.features.relative_position import apply_local_grade_percentile


def predict_price(raw_df: pd.DataFrame, model_bundle: dict, spatial_index, demographics: pd.DataFrame,
                   as_of_date: date | None = None) -> np.ndarray:
    """Prevê `price` em dólar para imóveis sem `id`/`date`/`price` (ex: `future_unseen_examples.csv`).

    `as_of_date`: data de referência para calcular `property_age` (imóveis futuros não têm data de
    venda real) — default hoje. Decisão de design explícita: em produção real, a API de inferência
    passaria a data da requisição; aqui, default determinístico para reprodutibilidade do teste.
    """
    as_of = as_of_date or datetime.now().date()
    df = raw_df.copy()
    df["date"] = as_of.strftime("%Y%m%dT000000")

    df = merge_demographics(df, demographics)
    df = add_property_age(df)
    df = add_dist_to_seattle_center(df)
    df["comps_knn_price"] = apply_comps_knn_price(df, spatial_index)
    df["local_grade_percentile"] = apply_local_grade_percentile(df, spatial_index)

    X = df[model_bundle["feature_cols"]]
    pred_log = model_bundle["model"].predict(X)
    return np.expm1(pred_log)


def load_production_artifacts(artifacts_dir: str = "artifacts", data_dir: str = "data"):
    model_bundle = joblib.load(Path(artifacts_dir) / "model_final.pkl")
    spatial = joblib.load(Path(artifacts_dir) / "spatial_index.pkl")
    demographics = pd.read_csv(Path(data_dir) / "raw" / "zipcode_demographics.csv")
    return model_bundle, spatial["index"], demographics
