"""Fase 05 — Técnica B (secundária): perturbação controlada de tamanho, com ajuste de `price` por
elasticidade local — nunca copia o preço original pro imóvel perturbado (isso introduziria ruído puro,
sinalizado como problema pelo usuário). Só opera sobre `train` (P1).

Simplificação consciente e documentada: a elasticidade é estimada globalmente em `train`
(`price_log ~ log(sqft_living)`), não por cluster físico — o cluster (fase 06) ainda não existe nesta
ordem do pipeline (a fase 05 roda antes da segmentação).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def estimate_size_elasticity(train_df: pd.DataFrame, size_col: str = "sqft_living",
                              target_col: str = "price_log") -> float:
    """Slope de `price_log ~ log(size_col)` em train — elasticidade preço×tamanho (P1: fit em train)."""
    log_size = np.log(train_df[size_col])
    slope, _ = np.polyfit(log_size, train_df[target_col], 1)
    return float(slope)


def controlled_perturbation_augment(train_df: pd.DataFrame, target_col: str = "price_log",
                                     size_col: str = "sqft_living", lot_col: str = "sqft_lot",
                                     size_pct_range: tuple[float, float] = (0.02, 0.05),
                                     lot_pct: float = 0.05, augment_fraction: float = 1.0,
                                     random_state: int = 42) -> pd.DataFrame:
    """Perturba `size_col`±2-5%/`lot_col`±5% de uma amostra de `train`, mantendo
    grade/condition/waterfront/view/zipcode fixos, e ajusta `price`/`price_log` proporcionalmente pela
    elasticidade local estimada (não copia o preço original — regra defensável, P4)."""
    rng = np.random.RandomState(random_state)
    elasticity = estimate_size_elasticity(train_df, size_col, target_col)

    n_sample = int(round(len(train_df) * augment_fraction))
    if augment_fraction >= 1.0:
        # perturba todas as linhas, na ordem original (evita embaralhar sem necessidade)
        sample_idx = np.resize(np.arange(len(train_df)), n_sample)
    else:
        sample_idx = rng.choice(len(train_df), size=n_sample, replace=False)

    synthetic = train_df.iloc[sample_idx].copy().reset_index(drop=True)
    size_pct = rng.uniform(size_pct_range[0], size_pct_range[1], size=n_sample) * rng.choice(
        [-1, 1], size=n_sample
    )
    lot_pct_signed = rng.uniform(-lot_pct, lot_pct, size=n_sample)

    old_size = synthetic[size_col].to_numpy()
    new_size = old_size * (1 + size_pct)
    synthetic[size_col] = new_size
    synthetic[lot_col] = synthetic[lot_col].to_numpy() * (1 + lot_pct_signed)

    log_ratio = np.log(new_size) - np.log(old_size)
    synthetic[target_col] = synthetic[target_col].to_numpy() + elasticity * log_ratio
    if "price" in synthetic.columns:
        synthetic["price"] = np.expm1(synthetic[target_col])
    if "id" in synthetic.columns:
        synthetic["id"] = -1

    return pd.concat([train_df, synthetic], ignore_index=True)
