"""Fase 06 — feature CONTEXTUAL de acessibilidade. `add_dist_to_seattle_center` não tem `.fit` (ponto
de referência fixo, não aprendido de dado algum) — mesmo assim vive em CONTEXTUAL por ser
espacial/geográfica.

`fit_train_zip_index`/`apply_dist_to_nearest_train_zip` (fase 08) promovem pra feature real o
diagnóstico da fase 04 (`src/evaluation/geographic_coverage.py::nearest_train_zip_distance`, que só
media distância zip-a-zip agregada) — aqui a distância é por imóvel (própria lat/long da linha), não
por centróide do zipcode inteiro, fit só em TRAIN (P1)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

SEATTLE_CENTER_LAT = 47.6062
SEATTLE_CENTER_LONG = -122.3321


def add_dist_to_seattle_center(df: pd.DataFrame) -> pd.DataFrame:
    """Distância euclidiana (graus) a um ponto fixo do centro de Seattle. Mecanismo ACCESSIBILITY —
    não usa zipcode como identidade, só coordenadas contínuas (P1)."""
    df = df.copy()
    df["dist_to_seattle_center"] = np.sqrt(
        (df["lat"] - SEATTLE_CENTER_LAT) ** 2 + (df["long"] - SEATTLE_CENTER_LONG) ** 2
    )
    return df


def fit_train_zip_index(train_df: pd.DataFrame, zip_col: str = "zipcode") -> NearestNeighbors:
    """Centróide (lat/long médio) de cada zipcode de TRAIN, indexado para consulta por vizinho mais
    próximo. Fit só em TRAIN (P1) — nunca usa identidade de zipcode como feature, só a geometria."""
    centroids = train_df.groupby(zip_col)[["lat", "long"]].mean()
    return NearestNeighbors(n_neighbors=1).fit(centroids.values)


def apply_dist_to_nearest_train_zip(df: pd.DataFrame, train_zip_index: NearestNeighbors) -> np.ndarray:
    """Distância (graus) da lat/long do imóvel ao centróide do zipcode de TRAIN mais próximo."""
    dist, _ = train_zip_index.kneighbors(df[["lat", "long"]].values)
    return dist.ravel()
