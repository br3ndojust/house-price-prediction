"""Fase 01 — limpeza de data/raw/kc_house_data.csv (P1: RAW nunca sobrescrito)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def drop_implausible_rows(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Remove linhas fisicamente implausíveis (erro de digitação), não outliers de mercado reais.

    bedrooms=33 num imóvel de 1620 sqft_living (~49 sqft/quarto) é incompatível com qualquer
    configuração residencial real — mantido como erro de digitação (não como sinal de mercado, que é
    o caso de preços/áreas extremas mas fisicamente consistentes, que permanecem no dataset).
    """
    mask = df["bedrooms"] > 15
    dropped_ids = df.loc[mask, "id"].tolist()
    return df.loc[~mask].copy(), {"dropped_implausible_bedrooms_ids": dropped_ids}


def add_price_log(df: pd.DataFrame) -> pd.DataFrame:
    """price_log = log1p(price) — skewness bruto ~4.0 cai para ~0.43 em log (checado empiricamente)."""
    df = df.copy()
    df["price_log"] = np.log1p(df["price"])
    return df


def clean_kc_house_data(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    df, log = drop_implausible_rows(df)
    df = add_price_log(df)
    log["n_rows_out"] = len(df)
    log["n_resales"] = int(df["id"].duplicated(keep=False).sum())
    return df, log
