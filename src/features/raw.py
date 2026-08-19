"""Fase 05 — features RAW: transformações row-wise diretas da fonte, sem `.fit` (P1: sem leakage)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def add_property_age(df: pd.DataFrame) -> pd.DataFrame:
    """Idade do imóvel na data da venda. Mecanismo AGE — depreciação/estilo construtivo."""
    df = df.copy()
    sale_year = pd.to_datetime(df["date"], format="%Y%m%dT%H%M%S").dt.year
    df["property_age"] = sale_year - df["yr_built"]
    return df


def add_renovation_features(df: pd.DataFrame) -> pd.DataFrame:
    """was_renovated / years_since_renovation. Mecanismo CONDITION."""
    df = df.copy()
    df["was_renovated"] = (df["yr_renovated"] > 0).astype(int)
    sale_year = pd.to_datetime(df["date"], format="%Y%m%dT%H%M%S").dt.year
    df["years_since_renovation"] = np.where(
        df["was_renovated"] == 1, sale_year - df["yr_renovated"], -1
    )
    return df


def add_log_sqft_lot(df: pd.DataFrame) -> pd.DataFrame:
    """log1p(sqft_lot). Mecanismo SIZE — skew bruto 10.7 (train), skew log 0.96."""
    df = df.copy()
    df["log_sqft_lot"] = np.log1p(df["sqft_lot"])
    return df
