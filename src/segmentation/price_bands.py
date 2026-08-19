"""Fase 03 — segmentação de mercado por preço (P1: fit só em train; P3: banda só nomeada se justificada)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def fit_price_quartiles(train_price_log: pd.Series) -> np.ndarray:
    """Breakpoints de quartil (25/50/75%) do price_log de TRAIN. TEST/VAL só aplicam (P1)."""
    return np.quantile(train_price_log, [0.25, 0.5, 0.75])


def apply_price_quartile(price_log: pd.Series, breakpoints: np.ndarray) -> pd.Series:
    labels = ["Q1", "Q2", "Q3", "Q4"]
    edges = [-np.inf, *breakpoints, np.inf]
    return pd.cut(price_log, bins=edges, labels=labels)


def band_separation_report(df: pd.DataFrame, quartile_col: str,
                            median_cols: list[str], rate_cols: list[str]) -> pd.DataFrame:
    """Mediana de `median_cols` (contínuas) + taxa (média) de `rate_cols` (binárias/raras) por
    quartil — usado para decidir se a banda semântica (Entry/Standard/Premium/Luxury) é
    estatisticamente justificável (P3), nunca forçada por nome. Mediana de uma binária rara (ex:
    waterfront) fica sempre 0 e não diferencia nada — por isso usa taxa, não mediana, para essas."""
    medians = df.groupby(quartile_col, observed=True)[median_cols].median()
    rates = df.groupby(quartile_col, observed=True)[rate_cols].mean().add_suffix("_rate")
    return medians.join(rates)


SEMANTIC_LABELS = {"Q1": "Entry", "Q2": "Standard", "Q3": "Premium", "Q4": "Luxury"}
