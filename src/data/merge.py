"""Fase 01 — merge de kc_house_data com zipcode_demographics. zipcode é só chave (P1: nunca feature)."""
from __future__ import annotations

import pandas as pd


def merge_demographics(house: pd.DataFrame, demographics: pd.DataFrame) -> pd.DataFrame:
    house_zips = set(house["zipcode"])
    demo_zips = set(demographics["zipcode"])
    missing = house_zips - demo_zips
    if missing:
        raise ValueError(f"zipcodes em house sem demografia correspondente: {missing}")
    if demographics["zipcode"].duplicated().any():
        raise ValueError("zipcode_demographics não é 1:1 por zipcode")

    merged = house.merge(demographics, on="zipcode", how="left", validate="many_to_one")
    assert len(merged) == len(house), "merge alterou o número de linhas"
    return merged
