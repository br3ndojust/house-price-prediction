"""Fase 06 — comparáveis de mercado via KNN espacial. `.fit` só em TRAIN (P1); target-derived
(usa price_log), por isso é a feature de maior risco de leakage do pipeln — self-match é excluído
explicitamente ao aplicar sobre as próprias linhas de treino (senão o vizinho mais próximo de uma
linha de treino seria ela mesma, com distância 0)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


class SpatialIndex:
    def __init__(self, nn: NearestNeighbors, price_log: np.ndarray, grade: np.ndarray, k: int):
        self.nn = nn
        self.price_log = price_log
        self.grade = grade
        self.k = k


def fit_spatial_index(train_df: pd.DataFrame, k: int) -> SpatialIndex:
    nn = NearestNeighbors(n_neighbors=k + 1).fit(train_df[["lat", "long"]].values)
    return SpatialIndex(
        nn=nn,
        price_log=train_df["price_log"].to_numpy(),
        grade=train_df["grade"].to_numpy(),
        k=k,
    )


def _neighbor_query(index: SpatialIndex, coords: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Retorna (idx, dist) dos k vizinhos por linha, excluindo self-match (distância ~0) quando
    presente."""
    dist, idx = index.nn.kneighbors(coords, n_neighbors=index.k + 1)
    self_mask = dist[:, 0] < 1e-9
    idx_out = np.where(self_mask[:, None], idx[:, 1:], idx[:, :-1])
    dist_out = np.where(self_mask[:, None], dist[:, 1:], dist[:, :-1])
    return idx_out, dist_out


def _neighbor_indices(index: SpatialIndex, coords: np.ndarray) -> np.ndarray:
    """Retorna os k índices vizinhos por linha, excluindo self-match (distância ~0) quando presente."""
    idx, _ = _neighbor_query(index, coords)
    return idx


def apply_comps_knn_price(df: pd.DataFrame, index: SpatialIndex) -> np.ndarray:
    coords = df[["lat", "long"]].values
    neighbor_idx = _neighbor_indices(index, coords)
    return index.price_log[neighbor_idx].mean(axis=1)


def apply_comps_knn_neighbor_distance(df: pd.DataFrame, index: SpatialIndex) -> np.ndarray:
    """Distância média (graus) aos k vizinhos usados em `comps_knn_price` — sinal explícito de quão
    confiável é aquele comp. Motivação (fase 08): `comps_knn_price` perde correlação forte de TRAIN
    (r=0.833) para TEST (r=0.601, gap -0.232) porque o índice é fit só em TRAIN e TEST cai em zipcodes
    nunca vistos (fase 03/04) — expor a distância deixa o modelo aprender a descontar o comp quando o
    vizinho mais próximo está longe, em vez de assumir peso uniforme."""
    coords = df[["lat", "long"]].values
    _, neighbor_dist = _neighbor_query(index, coords)
    return neighbor_dist.mean(axis=1)


def apply_local_price_dispersion(df: pd.DataFrame, index: SpatialIndex) -> np.ndarray:
    """Desvio-padrão do `price_log` entre os mesmos k vizinhos espaciais de `comps_knn_price` —
    heterogeneidade do mercado local (vizinhança com preço disperso é estruturalmente mais difícil de
    prever que uma homogênea). Reusa o mesmo índice fitado, sem novo `.fit`."""
    coords = df[["lat", "long"]].values
    neighbor_idx, _ = _neighbor_query(index, coords)
    return index.price_log[neighbor_idx].std(axis=1)
