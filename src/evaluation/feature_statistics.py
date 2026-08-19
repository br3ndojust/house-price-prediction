"""Estatísticas descritivas por feature (correlação com o alvo + quartis) — usado pela API/portal
para os gráficos de correlação, dispersão e quartis na tela Importância de Features (P5: cálculo
centralizado aqui, nunca duplicado no adapter da API)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def compute_feature_statistics(df: pd.DataFrame, feature_cols: list[str],
                                price_col: str = "price") -> dict[str, dict]:
    """Correlação de Pearson com `price_col` (bruta, não em log — mais interpretável pro usuário do
    portal) + quartis (min/q1/mediana/q3/max) de cada feature, sobre o `df` recebido (TRAIN)."""
    price = df[price_col].to_numpy(dtype=float)
    out: dict[str, dict] = {}
    for col in feature_cols:
        if col not in df.columns:
            continue
        values = df[col].to_numpy(dtype=float)
        corr = float(np.corrcoef(values, price)[0, 1]) if np.std(values) > 0 else 0.0
        q = np.quantile(values, [0, 0.25, 0.5, 0.75, 1.0])
        out[col] = {
            "correlation_with_price": round(corr, 4),
            "min": float(q[0]), "q1": float(q[1]), "median": float(q[2]),
            "q3": float(q[3]), "max": float(q[4]),
        }
    return out


def sample_feature_price_pairs(df: pd.DataFrame, feature_cols: list[str], price_col: str = "price",
                                sample_size: int = 400, random_state: int = 42) -> pd.DataFrame:
    """Amostra aleatória (TRAIN) de `feature_cols` + `price_col` — usada só pro gráfico de dispersão no
    portal, nunca para decisão de modelo (isso é fase 10, sobre o `df` inteiro)."""
    cols = [c for c in feature_cols if c in df.columns] + [price_col]
    n = min(sample_size, len(df))
    return df[cols].sample(n=n, random_state=random_state).reset_index(drop=True)


def compute_gain_importance(model, feature_cols: list[str]) -> dict[str, float]:
    """Importância bruta (não normalizada) de um modelo já treinado — gain nativo do XGBoost, ou
    `|coef_|` pra um pipeline linear (Ridge). Única fonte dessa lógica (P5) — reusada tanto pro
    ranking "ao vivo" do modelo ativo quanto pro snapshot persistido de qualquer modelo registrado."""
    if hasattr(model, "get_booster"):
        gain = model.get_booster().get_score(importance_type="gain")
        return {f: gain.get(f, 0.0) for f in feature_cols}
    if hasattr(model, "named_steps") and hasattr(model[-1], "coef_"):
        return dict(zip(feature_cols, np.abs(model[-1].coef_)))
    return dict.fromkeys(feature_cols, 0.0)
