"""Fase 07 — distância ao centróide do cluster físico. Reusa o `KMeans`/`StandardScaler` já fitados
só em TRAIN na fase 06 (`artifacts/property_cluster_model.pkl`) — não fita nada novo aqui (P1).

Motivação (extensão exploratória, `notebooks/06_market_property_segmentation.ipynb`): `property_cluster`
como categoria bruta carrega sinal real de erro (cluster raro teve erro médio 4.5x pior em `test`), mas
é instável sob o holdout geográfico por zipcode do split (fase 03) — a frequência do cluster raro varia
por acaso de qual zipcode caiu em qual partição. Uma distância contínua ao centróide preserva o sinal
sem depender do rótulo discreto raro."""
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd

from src.segmentation.property_clusters import PHYSICAL_PROFILE_COLUMNS


def add_property_cluster_distance(
    df: pd.DataFrame, property_cluster_model_path: str = "artifacts/property_cluster_model.pkl"
) -> pd.DataFrame:
    """Requer a coluna `property_cluster` já presente em `df` (fase 06, `house_segments.parquet`)."""
    df = df.copy()
    artifact = joblib.load(property_cluster_model_path)
    scaler, km = artifact["scaler"], artifact["kmeans"]

    X = scaler.transform(df[PHYSICAL_PROFILE_COLUMNS])
    assigned_centroids = km.cluster_centers_[df["property_cluster"].to_numpy()]
    df["property_cluster_distance"] = np.linalg.norm(X - assigned_centroids, axis=1)
    return df
