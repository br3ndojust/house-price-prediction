"""Critério objetivo de promoção — compara o candidato recém-treinado contra o modelo ATIVO, no
mesmo `val`, quebrado por segmento (P3/P4: nunca decide promoção olhando só a métrica agregada).
Pré-registrado: candidato só é recomendado se não piorar o MAE em nenhum segmento além de uma
margem tolerável — uma melhora agregada que esconde piora num segmento é REJEITADA, não recomendada
(mesma disciplina descrita em docs/entregavel_04_aprendizado_continuo.md, seção 5)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.evaluation.error_matrix import LOW_SAMPLE_THRESHOLD, compute_metrics, error_matrix_by


def score_model_on_val(model_bundle: dict, val_df: pd.DataFrame, target: str = "price_log") -> dict:
    """Roda um bundle `{"model", "feature_cols"}` já treinado sobre `val` — usado tanto pro candidato
    quanto pro modelo ativo, pra comparação em igualdade de condições."""
    feature_cols = model_bundle["feature_cols"]
    y_true = np.expm1(val_df[target].to_numpy())
    y_pred = np.expm1(model_bundle["model"].predict(val_df[feature_cols]))
    scored = val_df.copy()
    scored["y_true"] = y_true
    scored["y_pred"] = y_pred
    return {
        "global": compute_metrics(y_true, y_pred),
        "by_price_band": error_matrix_by(scored, "price_band", "y_true", "y_pred").to_dict(orient="records"),
    }


def check_promotion_criterion(candidate_by_band: list[dict], active_by_band: list[dict],
                               tolerance: float = 0.02) -> dict:
    """`tolerance`: margem tolerável de piora de MAE por segmento (0.02 = 2%, mesmo valor usado como
    exemplo em `docs/entregavel_04_aprendizado_continuo.md`). Segmentos onde o modelo ativo tem
    amostra pequena (`< LOW_SAMPLE_THRESHOLD`) não entram na decisão — dado insuficiente pra
    comparação confiável (P4)."""
    active_by_key = {r["price_band"]: r for r in active_by_band}
    violations = []
    for row in candidate_by_band:
        band = row["price_band"]
        active_row = active_by_key.get(band)
        if active_row is None or active_row["n"] < LOW_SAMPLE_THRESHOLD:
            continue
        allowed = active_row["mae"] * (1 + tolerance)
        if row["mae"] > allowed:
            violations.append({
                "price_band": band,
                "candidate_mae": round(row["mae"], 2),
                "active_mae": round(active_row["mae"], 2),
                "allowed_mae": round(allowed, 2),
            })
    return {"recommended": len(violations) == 0, "tolerance": tolerance, "violations": violations}
