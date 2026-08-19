"""Fase 05 — Técnica A: SMOGN (reimplementação lean, do zero — não usa o pacote `smogn` do PyPI, não
instalado/sem manutenção ativa, nem código de projeto anterior). Só opera sobre `train` (P1).

Algoritmo (Branco/Torgo/Ribeiro 2017, versão enxuta): imóveis em regiões raras do alvo (`price_log`
muito baixo ou muito alto) são super-representados via 2 mecanismos — interpolação SMOTE-like com um
vizinho também raro (mais realista), ou ruído gaussiano quando o vizinho mais próximo é "comum" (evita
inventar um ponto a meio caminho entre um extremo e o centro da distribuição, que seria irreal).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


def relevance_mask(target: pd.Series, low_q: float = 0.10, high_q: float = 0.90) -> pd.Series:
    """Região "rara" do alvo: cauda inferior ou superior (P1: fit dos quantis sempre em train)."""
    lo, hi = target.quantile([low_q, high_q])
    return (target <= lo) | (target >= hi)


def smogn_augment(train_df: pd.DataFrame, feature_cols: list[str], target_col: str,
                   k: int = 5, tail_boost: float = 1.0, low_q: float = 0.10, high_q: float = 0.90,
                   noise_scale: float = 0.05, random_state: int = 42) -> pd.DataFrame:
    """Retorna `train_df` + linhas sintéticas geradas a partir das regiões raras do alvo.

    Colunas fora de `feature_cols`/`target_col` (ex: `id`, `zipcode`) são herdadas da linha "base" —
    nenhum zipcode novo é fabricado (P1), a linha sintética representa uma variação de um imóvel real.
    """
    rng = np.random.RandomState(random_state)
    rare_mask = relevance_mask(train_df[target_col], low_q, high_q).to_numpy()
    rare_idx = np.flatnonzero(rare_mask)
    if len(rare_idx) == 0:
        return train_df.copy()

    n_synthetic = int(round(len(rare_idx) * tail_boost))

    X = train_df[feature_cols].to_numpy(dtype=float)
    mean, std = X.mean(axis=0), X.std(axis=0)
    std[std == 0] = 1.0
    Xz = (X - mean) / std

    nn = NearestNeighbors(n_neighbors=k + 1).fit(Xz)
    _, neighbor_idx = nn.kneighbors(Xz[rare_idx])  # inclui self na coluna 0

    synthetic_rows = []
    for _ in range(n_synthetic):
        pos = rng.randint(len(rare_idx))
        base_i = rare_idx[pos]
        candidates = neighbor_idx[pos, 1:]  # exclui self
        neighbor_i = candidates[rng.randint(len(candidates))]

        base_row = train_df.iloc[base_i].copy()
        if rare_mask[neighbor_i]:
            # vizinho tambem raro -> interpolacao SMOTE-like (mais realista, ambos sao "extremos")
            frac = rng.uniform(0.0, 1.0)
            new_features = X[base_i] + frac * (X[neighbor_i] - X[base_i])
            new_target = train_df[target_col].iloc[base_i] + frac * (
                train_df[target_col].iloc[neighbor_i] - train_df[target_col].iloc[base_i]
            )
        else:
            # vizinho comum -> ruido gaussiano em torno do ponto raro (nao interpola rumo ao centro)
            noise = rng.normal(0.0, noise_scale, size=len(feature_cols)) * std
            new_features = X[base_i] + noise
            target_std = train_df[target_col].std()
            new_target = train_df[target_col].iloc[base_i] + rng.normal(0.0, noise_scale * target_std)

        new_row = base_row.copy()
        for col, val in zip(feature_cols, new_features):
            new_row[col] = val
        new_row[target_col] = new_target
        if "id" in new_row:
            new_row["id"] = -1  # sintetico, nunca colide com id real
        synthetic_rows.append(new_row)

    synthetic_df = pd.DataFrame(synthetic_rows).reset_index(drop=True)
    return pd.concat([train_df, synthetic_df], ignore_index=True)
