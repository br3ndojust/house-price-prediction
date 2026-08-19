"""Fase 07 — checagem de leakage e determinismo de features (P1, P5)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def check_determinism(build_fn, df: pd.DataFrame, n_repeats: int = 2) -> bool:
    """Mesma entrada + mesma config -> mesmo resultado, sempre (P2: determinismo)."""
    results = [build_fn(df.copy()) for _ in range(n_repeats)]
    first = results[0]
    return all(
        (r.select_dtypes(include=[np.number]).fillna(-999) ==
         first.select_dtypes(include=[np.number]).fillna(-999)).all().all()
        for r in results[1:]
    )


def check_spatial_index_fit_size(spatial_index, expected_train_rows: int) -> bool:
    """Confirma estruturalmente que o índice espacial foi `.fit` só com linhas de TRAIN — os
    índices retornados pelo KNN são posicionais dentro do array de fit, então não podem apontar
    para nada fora de TRAIN por construção; aqui só confirmamos o tamanho do array de fit."""
    return len(spatial_index.price_log) == expected_train_rows


def train_test_correlation_gap(df: pd.DataFrame, feature: str, target: str,
                                suspicious_gap: float = 0.15) -> dict:
    """Gap de correlação TRAIN->TEST. Um gap NEGATIVO grande (TEST << TRAIN) é esperado para
    features target-derived e reflete dificuldade de generalização geográfica, não leakage. Um
    gap POSITIVO grande (TEST >> TRAIN) seria suspeito — sinal de possível leakage."""
    train = df[df["split"] == "train"]
    test = df[df["split"] == "test"]
    r_train = train[feature].corr(train[target])
    r_test = test[feature].corr(test[target])
    gap = r_test - r_train
    return {
        "feature": feature, "r_train": r_train, "r_test": r_test, "gap": gap,
        "suspicious_leakage": bool(gap > suspicious_gap),
    }
