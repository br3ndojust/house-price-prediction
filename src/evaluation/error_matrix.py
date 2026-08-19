"""Fase 09 — Error Matrix: nunca reportar só métrica agregada (P3, P4)."""
from __future__ import annotations

import numpy as np
import pandas as pd

LOW_SAMPLE_THRESHOLD = 20


def r2_log_score(y_true_log: np.ndarray, y_pred_log: np.ndarray) -> float:
    """R² no espaço `price_log` (o alvo do modelo) — única fonte dessa fórmula (P5), reusada pela CV
    de TRAIN (`src/models/train.py`) e pelas avaliações em TEST/VAL (fases 11/13), pra dar um R²
    comparável entre as três partições."""
    ss_res = np.sum((y_true_log - y_pred_log) ** 2)
    ss_tot = np.sum((y_true_log - y_true_log.mean()) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0


def compute_metrics(y_true_dollar: np.ndarray, y_pred_dollar: np.ndarray) -> dict:
    err = y_true_dollar - y_pred_dollar
    abs_err = np.abs(err)
    ape = abs_err / np.maximum(y_true_dollar, 1)
    return {
        "n": int(len(y_true_dollar)),
        "mae": float(np.mean(abs_err)),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "median_ae": float(np.median(abs_err)),
        "mape": float(np.mean(ape)),
        "median_ape": float(np.median(ape)),
        "p90_ae": float(np.percentile(abs_err, 90)),
        "bias": float(np.mean(err)),  # positivo = modelo subestima (y_true > y_pred) em média
        "low_sample": bool(len(y_true_dollar) < LOW_SAMPLE_THRESHOLD),
    }


def error_matrix_by(df: pd.DataFrame, group_col: str, y_true_col: str,
                     y_pred_col: str) -> pd.DataFrame:
    rows = []
    for group, sub in df.groupby(group_col, observed=True):
        metrics = compute_metrics(sub[y_true_col].to_numpy(), sub[y_pred_col].to_numpy())
        metrics[group_col] = group
        rows.append(metrics)
    cols = [group_col, "n", "mae", "rmse", "median_ae", "mape", "median_ape", "p90_ae", "bias",
            "low_sample"]
    return pd.DataFrame(rows)[cols].sort_values("mae", ascending=False).reset_index(drop=True)


def characteristic_cuts(df: pd.DataFrame, y_true_col: str, y_pred_col: str) -> pd.DataFrame:
    definitions = {
        "waterfront=1": df["waterfront"] == 1,
        "grade>=10": df["grade"] >= 10,
        "view>0": df["view"] > 0,
        "renovated": df["yr_renovated"] > 0,
    }
    rows = []
    for name, mask in definitions.items():
        sub = df[mask]
        if len(sub) == 0:
            continue
        metrics = compute_metrics(sub[y_true_col].to_numpy(), sub[y_pred_col].to_numpy())
        metrics["characteristic"] = name
        rows.append(metrics)
    cols = ["characteristic", "n", "mae", "rmse", "median_ae", "mape", "median_ape", "p90_ae",
            "bias", "low_sample"]
    return pd.DataFrame(rows)[cols]
