"""Fases 02+04 — representação geográfica e cobertura de dado por zipcode.

Fase 02 usa só `zip_representation_summary` (descritivo, 100% dos dados, pré-split, P1). Fase 04 usa o
restante (comparação de distribuição, distância geográfica, cobertura no espaço de features, baldes de
escassez) — todas rodam pós-split, comparando partições já definidas, sem `.fit` de nada reaplicável.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


def zip_representation_summary(df: pd.DataFrame, zip_col: str = "zipcode",
                                price_col: str = "price") -> pd.DataFrame:
    """n de imóveis, preço médio/mediano/std/IQR por zipcode — descritivo, 100% dos dados (fase 02)."""
    g = df.groupby(zip_col)[price_col]
    out = g.agg(n="count", mean_price="mean", median_price="median", std_price="std").reset_index()
    q1 = g.quantile(0.25).reset_index(name="q1")
    q3 = g.quantile(0.75).reset_index(name="q3")
    out = out.merge(q1, on=zip_col).merge(q3, on=zip_col)
    out["iqr_price"] = out["q3"] - out["q1"]
    return out.sort_values("n").reset_index(drop=True)


def distribution_comparison(df: pd.DataFrame, split_col: str, cols: list[str]) -> pd.DataFrame:
    """Compara quantis (min/p25/mediana/p75/max) de cada coluna entre partições do split (fase 04)."""
    rows = []
    for col in cols:
        for part, sub in df.groupby(split_col, observed=True):
            q = sub[col].quantile([0, 0.25, 0.5, 0.75, 1.0])
            rows.append({
                "column": col, "split": part, "n": len(sub),
                "min": q.iloc[0], "p25": q.iloc[1], "median": q.iloc[2], "p75": q.iloc[3],
                "max": q.iloc[4], "mean": sub[col].mean(),
            })
    return pd.DataFrame(rows)


def zip_centroids(df: pd.DataFrame, zip_col: str = "zipcode") -> pd.DataFrame:
    return df.groupby(zip_col)[["lat", "long"]].mean().reset_index()


def nearest_train_zip_distance(train_df: pd.DataFrame, other_df: pd.DataFrame,
                                zip_col: str = "zipcode") -> pd.DataFrame:
    """Para cada zipcode de `other_df` (test/val), distância (graus) até o centróide do zipcode de
    TRAIN mais próximo — separa "zipcode não visto perto de zipcodes conhecidos" de "regime
    geográfico não visto" (fase 04)."""
    train_centroids = zip_centroids(train_df, zip_col)
    other_centroids = zip_centroids(other_df, zip_col)

    nn = NearestNeighbors(n_neighbors=1).fit(train_centroids[["lat", "long"]].values)
    dist, idx = nn.kneighbors(other_centroids[["lat", "long"]].values)

    out = other_centroids.copy()
    out["nearest_train_zipcode"] = train_centroids.iloc[idx.ravel()][zip_col].to_numpy()
    out["distance_to_nearest_train_zip"] = dist.ravel()
    return out.sort_values("distance_to_nearest_train_zip", ascending=False).reset_index(drop=True)


class FeatureSpaceIndex:
    """`.fit` (mean/std, NearestNeighbors) feito uma única vez em TRAIN (P1) — reusável para muitas
    consultas depois (fase 04 original só fazia 1 chamada por execução; a Matriz de Confiança em
    produção, `app/infrastructure/confidence/confidence_scorer.py`, consulta por requisição — refitar
    a árvore em ~15 mil linhas de TRAIN a cada predição é o que causava timeout em lotes grandes)."""

    def __init__(self, nn: NearestNeighbors, mean: pd.Series, std: pd.Series, feature_cols: list[str]):
        self.nn = nn
        self.mean = mean
        self.std = std
        self.feature_cols = feature_cols


def fit_feature_space_index(train_df: pd.DataFrame, feature_cols: list[str]) -> FeatureSpaceIndex:
    mean = train_df[feature_cols].mean()
    std = train_df[feature_cols].std().replace(0, 1)
    train_z = (train_df[feature_cols] - mean) / std
    nn = NearestNeighbors(n_neighbors=1).fit(train_z.values)
    return FeatureSpaceIndex(nn, mean, std, feature_cols)


def apply_feature_space_distance(index: FeatureSpaceIndex, other_df: pd.DataFrame) -> np.ndarray:
    other_z = (other_df[index.feature_cols] - index.mean) / index.std
    dist, _ = index.nn.kneighbors(other_z.values)
    return dist.ravel()


def feature_space_coverage(train_df: pd.DataFrame, other_df: pd.DataFrame,
                            feature_cols: list[str]) -> pd.DataFrame:
    """Para cada linha de `other_df`, distância normalizada (z-score) ao vizinho mais próximo em
    TRAIN no espaço de features — mede se test/val caem dentro da região coberta por TRAIN (fase 04).
    Fit + consulta num único call — para uso pontual (1 chamada por execução, src/scripts/notebooks).
    Para consultar repetidamente sobre o mesmo TRAIN, use `fit_feature_space_index` uma vez +
    `apply_feature_space_distance` por consulta, em vez desta função (evita refitar a árvore)."""
    index = fit_feature_space_index(train_df, feature_cols)
    distances = apply_feature_space_distance(index, other_df)

    out = other_df[[c for c in ["id", "zipcode"] if c in other_df.columns]].copy()
    out["nearest_train_neighbor_distance"] = distances
    return out


def data_scarcity_buckets(train_df: pd.DataFrame, zip_col: str = "zipcode",
                           low_max: int = 100, medium_max: int = 500) -> pd.DataFrame:
    """Classifica cada zipcode de TRAIN em LOW/MEDIUM/HIGH por volume de dado (fase 04)."""
    counts = train_df.groupby(zip_col).size().reset_index(name="n_properties")

    def bucket(n: int) -> str:
        if n < low_max:
            return "LOW"
        if n < medium_max:
            return "MEDIUM"
        return "HIGH"

    counts["scarcity_bucket"] = counts["n_properties"].apply(bucket)
    return counts
