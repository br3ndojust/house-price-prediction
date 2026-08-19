"""Fase 07 — ablation: baseline -> +feature -> CV -> métrica por segmento -> decisão (P4).

Usa um XGBoost fixo (não tunado — tuning é decisão da fase 08) só para isolar o efeito de cada
grupo de features, sempre via GroupKFold por zipcode restrito a TRAIN (nunca partição única, P1/P4).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from xgboost import XGBRegressor

ABLATION_MODEL_PARAMS = dict(n_estimators=300, max_depth=4, learning_rate=0.05, random_state=42)


def run_ablation_cv(train_df: pd.DataFrame, feature_cols: list[str], target: str,
                     group_col: str, n_splits: int, random_state: int) -> dict:
    """Retorna MAE em dólar (global e por price_band) agregado sobre os folds."""
    gkf = GroupKFold(n_splits=n_splits)
    X = train_df[feature_cols]
    y = train_df[target]
    groups = train_df[group_col]

    fold_mae = []
    band_maes: dict[str, list[float]] = {}

    for train_idx, val_idx in gkf.split(X, y, groups=groups):
        model = XGBRegressor(**ABLATION_MODEL_PARAMS, n_jobs=-1)
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        pred_log = model.predict(X.iloc[val_idx])

        y_true_dollar = np.expm1(y.iloc[val_idx])
        y_pred_dollar = np.expm1(pred_log)
        mae = np.mean(np.abs(y_true_dollar - y_pred_dollar))
        fold_mae.append(mae)

        if "price_band" in train_df.columns:
            bands = train_df.iloc[val_idx]["price_band"]
            for band in bands.unique():
                mask = (bands == band).to_numpy()
                band_mae = np.mean(np.abs(y_true_dollar.to_numpy()[mask] - y_pred_dollar[mask]))
                band_maes.setdefault(band, []).append(band_mae)

    return {
        "mae_mean": float(np.mean(fold_mae)),
        "mae_std": float(np.std(fold_mae)),
        "mae_by_band_mean": {b: float(np.mean(v)) for b, v in band_maes.items()},
    }
