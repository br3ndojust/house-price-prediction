"""Fase 05 — features DERIVED: razão/interação entre colunas, sem `.fit` (P1: sem leakage)."""
from __future__ import annotations

import pandas as pd


def add_basement_features(df: pd.DataFrame) -> pd.DataFrame:
    """has_basement / basement_ratio. Mecanismo SIZE — composição do espaço, não só volume total."""
    df = df.copy()
    df["has_basement"] = (df["sqft_basement"] > 0).astype(int)
    df["basement_ratio"] = df["sqft_basement"] / df["sqft_living"].replace(0, pd.NA)
    df["basement_ratio"] = df["basement_ratio"].fillna(0)
    return df


def add_grade_condition_interaction(df: pd.DataFrame) -> pd.DataFrame:
    """grade * condition. Mecanismo QUALITY — testa se condition importa em interação com grade
    (fase 04: condition fraco isolado, r=0.057; leve correlação negativa com grade, -0.12)."""
    df = df.copy()
    df["grade_condition_interaction"] = df["grade"] * df["condition"]
    return df


def add_bathrooms_per_bedroom(df: pd.DataFrame) -> pd.DataFrame:
    """bathrooms / bedrooms. Mecanismo SIZE — a extensão exploratória da fase 04
    (`notebooks/04_geographic_coverage.ipynb`) achou `bathrooms` como a feature crua cujo gap de
    cobertura train->test mais correlaciona com erro real (r=0.25, maior entre as 16
    `BASELINE_FEATURES`); razão relativa tende a generalizar melhor que a contagem bruta sob o
    holdout geográfico completo do split (fase 03)."""
    df = df.copy()
    df["bathrooms_per_bedroom"] = df["bathrooms"] / df["bedrooms"].replace(0, 1)
    return df


def add_sqft_living_to_lot_ratio(df: pd.DataFrame) -> pd.DataFrame:
    """sqft_living / sqft_lot. Mecanismo SIZE — a extensão da fase 04 achou `sqft_lot`/`sqft_lot15`
    como as features com maior gap de distribuição relativa train->test (20-21%) depois de `floors`;
    `log_sqft_lot` (fase 05) já foi rejeitado na ablation olhando só MAE médio — esta razão ataca a
    mesma escassez por outro ângulo (tamanho relativo ao lote, não escala absoluta)."""
    df = df.copy()
    df["sqft_living_to_lot_ratio"] = df["sqft_living"] / df["sqft_lot"].replace(0, pd.NA)
    df["sqft_living_to_lot_ratio"] = df["sqft_living_to_lot_ratio"].fillna(0)
    return df
