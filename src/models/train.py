"""Fase 08 — seleção de modelo via GroupKFold por zipcode (P1: nunca partição única; P4: reporta
empate técnico explicitamente, nunca esconde atrás de um único número)."""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from src.evaluation.error_matrix import r2_log_score


def _param_grid(param_dict: dict) -> list[dict]:
    keys = list(param_dict)
    combos = itertools.product(*(param_dict[k] for k in keys))
    return [dict(zip(keys, combo)) for combo in combos]


def build_model(name: str, params: dict):
    if name == "ridge":
        return make_pipeline(StandardScaler(), Ridge(**params))
    if name == "xgboost":
        return XGBRegressor(**params, random_state=42, n_jobs=-1)
    raise ValueError(f"modelo desconhecido: {name}")


#: métricas disponíveis para ranquear a seleção de modelo — chave -> (campo mean, campo std, direção).
#: direção 1 = menor é melhor (mae/rmse/mape), -1 = maior é melhor (r2). MAE (dollar) é a métrica
#: oficial (P4: critério pré-registrado) — as demais existem para candidatos experimentais via
#: retraining (tela Treino do portal), nunca mudam o critério oficial de promoção.
METRIC_KEYS = {
    "mae": ("mae_mean", "mae_std", 1),
    "rmse": ("rmse_mean", "rmse_std", 1),
    "mape": ("mape_mean", "mape_std", 1),
    "r2": ("r2_log_mean", "r2_log_std", -1),
}


def cross_validate_candidate(train_df: pd.DataFrame, feature_cols: list[str], target: str,
                              group_col: str, model_name: str, params: dict,
                              n_splits: int, random_state: int) -> dict:
    gkf = GroupKFold(n_splits=n_splits)
    X = train_df[feature_cols]
    y = train_df[target]
    groups = train_df[group_col]

    fold_mae, fold_rmse, fold_mape, fold_r2 = [], [], [], []
    for train_idx, val_idx in gkf.split(X, y, groups=groups):
        model = build_model(model_name, params)
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        pred_log = model.predict(X.iloc[val_idx])

        y_true_log = y.iloc[val_idx].to_numpy()
        y_true_dollar = np.expm1(y_true_log)
        y_pred_dollar = np.expm1(pred_log)

        abs_error = np.abs(y_true_dollar - y_pred_dollar)
        fold_mae.append(np.mean(abs_error))
        fold_rmse.append(np.sqrt(np.mean((y_true_dollar - y_pred_dollar) ** 2)))
        fold_mape.append(np.mean(abs_error / y_true_dollar) * 100)
        fold_r2.append(r2_log_score(y_true_log, pred_log))

    return {
        "model": model_name, "params": params,
        "mae_mean": float(np.mean(fold_mae)), "mae_std": float(np.std(fold_mae)),
        "rmse_mean": float(np.mean(fold_rmse)), "rmse_std": float(np.std(fold_rmse)),
        "mape_mean": float(np.mean(fold_mape)), "mape_std": float(np.std(fold_mape)),
        "r2_log_mean": float(np.mean(fold_r2)), "r2_log_std": float(np.std(fold_r2)),
    }


def run_model_selection(train_df: pd.DataFrame, feature_cols: list[str], target: str,
                         group_col: str, model_grids: dict, n_splits: int,
                         random_state: int, metric: str = "mae") -> list[dict]:
    results = []
    for model_name, grid in model_grids.items():
        for params in _param_grid(grid):
            results.append(
                cross_validate_candidate(
                    train_df, feature_cols, target, group_col, model_name, params,
                    n_splits, random_state,
                )
            )
    mean_key, _, direction = METRIC_KEYS[metric]
    return sorted(results, key=lambda r: direction * r[mean_key])


def technical_tie_check(results: list[dict], metric: str = "mae") -> dict:
    """Gap entre o 1º e o 2º colocado vs. soma dos desvios-padrão dos folds — se o gap for menor,
    é empate técnico e deve ser reportado como tal (P4), nunca escondido atrás do argmax."""
    if len(results) < 2:
        return {"is_tie": False}
    mean_key, std_key, direction = METRIC_KEYS[metric]
    best, second = results[0], results[1]
    gap = direction * (second[mean_key] - best[mean_key])
    std_sum = best[std_key] + second[std_key]
    return {
        "is_tie": bool(gap < std_sum),
        "gap": gap,
        "std_sum": std_sum,
        "metric": metric,
        "best": {"model": best["model"], "params": best["params"], mean_key: best[mean_key]},
        "second": {"model": second["model"], "params": second["params"], mean_key: second[mean_key]},
    }
