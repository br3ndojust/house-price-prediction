"""Fase 06 — posição relativa: percentil de grade entre vizinhos espaciais. `.fit` (índice espacial)
só em TRAIN, reusa `SpatialIndex` de `src/features/comparable.py` (P1, P5: sem duplicar lógica de KNN)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.comparable import SpatialIndex, _neighbor_indices


def apply_local_grade_percentile(df: pd.DataFrame, index: SpatialIndex) -> np.ndarray:
    coords = df[["lat", "long"]].values
    neighbor_idx = _neighbor_indices(index, coords)
    neighbor_grades = index.grade[neighbor_idx]
    own_grade = df["grade"].to_numpy().reshape(-1, 1)
    return (neighbor_grades < own_grade).mean(axis=1)
