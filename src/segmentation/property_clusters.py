"""Fase 03 — clustering de perfil físico. Nunca usa price/lat/long/zipcode (P3: perfil físico e mercado
não se misturam), fit só em train (P1)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from src.features.raw import add_property_age

PHYSICAL_PROFILE_COLUMNS = [
    "sqft_living", "sqft_lot", "bedrooms", "bathrooms", "floors",
    "grade", "condition", "view", "waterfront", "property_age",
]

__all__ = ["add_property_age", "PHYSICAL_PROFILE_COLUMNS", "scan_property_cluster_k",
           "fit_property_cluster", "apply_property_cluster"]


def scan_property_cluster_k(train_df: pd.DataFrame, k_range: range,
                             random_state: int) -> pd.DataFrame:
    """Silhouette por k, calculado só em train — usado para escolher k antes de fixar o cluster (P1)."""
    scaler = StandardScaler()
    X = scaler.fit_transform(train_df[PHYSICAL_PROFILE_COLUMNS])
    rows = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = km.fit_predict(X)
        score = silhouette_score(X, labels)
        rows.append({"k": k, "silhouette": score})
    return pd.DataFrame(rows)


def fit_property_cluster(train_df: pd.DataFrame, k: int, random_state: int):
    scaler = StandardScaler()
    X = scaler.fit_transform(train_df[PHYSICAL_PROFILE_COLUMNS])
    km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    km.fit(X)
    return scaler, km


def apply_property_cluster(df: pd.DataFrame, scaler: StandardScaler, km: KMeans) -> np.ndarray:
    X = scaler.transform(df[PHYSICAL_PROFILE_COLUMNS])
    return km.predict(X)
