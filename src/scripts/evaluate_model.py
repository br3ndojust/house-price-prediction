"""Fase 11 — error_matrix: nunca só métrica agregada (P3, P4). Roda só sobre `test` — repetível
entre iterações do loop de hipótese (fase 12). `val` NÃO é tocado aqui (P1) — só na fase 13, uma
única vez, depois que tudo (augmentation/features/modelo/hipóteses) estiver travado."""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.evaluation.error_matrix import characteristic_cuts, compute_metrics, error_matrix_by, r2_log_score


def predict(model_bundle: dict, df: pd.DataFrame) -> np.ndarray:
    X = df[model_bundle["feature_cols"]]
    pred_log = model_bundle["model"].predict(X)
    return np.expm1(pred_log)


def run_test_evaluation_pipeline(
    candidate_artifact_path: str = "artifacts/model_candidate.pkl",
    features_parquet: str = "data/trusted/features_contextual.parquet",
) -> dict:
    """Fase 11, refatorada em função chamável (P5) — reusada tanto pelo `main()` (CLI) quanto pelo
    `app/infrastructure/training/pipeline_training_service.py` (retraining via API), pra dar o lado
    TEST da comparação treino/teste/val que sinaliza overfitting/underfitting. Roda só sobre `test` —
    nunca toca `val` aqui (P1: checagem final única fica só na fase 13). Precisa rodar sobre o
    artefato ainda fitado só em `train` (fase 10) — a fase 13 refita em `train`+`test` e sobrescreve o
    mesmo artefato, então isso tem que acontecer ANTES dela na mesma execução."""
    model_bundle = joblib.load(candidate_artifact_path)
    df = pd.read_parquet(features_parquet)

    test = df[df["split"] == "test"].copy()
    pred_log = model_bundle["model"].predict(test[model_bundle["feature_cols"]])
    test["y_true"] = np.expm1(test[model_bundle["target"]])
    test["y_pred"] = np.expm1(pred_log)

    global_metrics = compute_metrics(test["y_true"].to_numpy(), test["y_pred"].to_numpy())
    global_metrics["r2_log"] = r2_log_score(test[model_bundle["target"]].to_numpy(), pred_log)
    by_band = error_matrix_by(test, "price_band", "y_true", "y_pred")
    return {
        "split_used": "test",
        "global": global_metrics,
        "by_price_band": by_band.to_dict(orient="records"),
    }


def main() -> None:
    base = run_test_evaluation_pipeline()
    model_bundle = joblib.load("artifacts/model_candidate.pkl")
    df = pd.read_parquet("data/trusted/features_contextual.parquet")
    test = df[df["split"] == "test"].copy()
    test["y_true"] = np.expm1(test[model_bundle["target"]])
    test["y_pred"] = predict(model_bundle, test)

    by_cluster = error_matrix_by(test, "property_cluster", "y_true", "y_pred")
    by_zipcode = error_matrix_by(test, "zipcode", "y_true", "y_pred")
    by_characteristic = characteristic_cuts(test, "y_true", "y_pred")

    Path("reports").mkdir(exist_ok=True)
    error_matrix = {
        **base,
        "note": "val nao entra aqui (P1) - checagem final unica fica na fase 13.",
        "by_property_cluster": by_cluster.to_dict(orient="records"),
        "by_zipcode": by_zipcode.to_dict(orient="records"),
        "by_characteristic": by_characteristic.to_dict(orient="records"),
    }
    Path("reports/error_matrix.json").write_text(json.dumps(error_matrix, indent=2, default=str))
    print(json.dumps(error_matrix, indent=2, default=str))


if __name__ == "__main__":
    main()
