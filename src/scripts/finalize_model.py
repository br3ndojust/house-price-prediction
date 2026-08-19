"""Fase 13 — final_refit_and_val_check: refit único combinando `train`+`test`, checagem ÚNICA em
`val` (P1: val sagrado, tocado uma única vez, depois de tudo — augmentation, features, modelo,
hipóteses — estar travado)."""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.evaluation.error_matrix import compute_metrics, error_matrix_by, r2_log_score
from src.models.train import build_model


def run_final_refit_pipeline(
    candidate_artifact_path: str = "artifacts/model_candidate.pkl",
    features_parquet: str = "data/trusted/features_contextual.parquet",
    final_artifact_path: str = "artifacts/model_final.pkl",
) -> dict:
    """Fase 13, refatorada em função chamável (P5) — reusada pelo `main()` (CLI) e pelo
    `app/infrastructure/training/pipeline_training_service.py` (retraining via API).

    ATENÇÃO (P1): fora do fluxo experimental original (fases 00-14), `val` deixa de ser "tocado uma
    única vez" no sentido estrito — um job de retraining via API que rode esta função reavalia `val`
    a cada novo candidato, igual a qualquer pipeline de reavaliação contínua real (docs/08_continuous_
    learning.md, seção 4: "Error Matrix comparativo candidato vs. produção"). O contrato de honestidade
    é preservado de outra forma: nenhuma decisão de feature/modelo/hipótese é tomada olhando `val` —
    ele só mede o candidato já travado, nunca escolhe entre alternativas.
    """
    model_bundle = joblib.load(candidate_artifact_path)
    feature_cols = model_bundle["feature_cols"]
    target = model_bundle["target"]

    df = pd.read_parquet(features_parquet)
    train_test = df[df["split"].isin(["train", "test"])]
    val = df[df["split"] == "val"].copy()

    final_model = build_model(model_bundle["model_name"], model_bundle["params"])
    final_model.fit(train_test[feature_cols], train_test[target])

    Path(final_artifact_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"model": final_model, "model_name": model_bundle["model_name"], "params": model_bundle["params"],
         "feature_cols": feature_cols, "target": target,
         "trained_on": "train+test (refit final, fase 13)"},
        final_artifact_path,
    )

    val_pred_log = final_model.predict(val[feature_cols])
    val["y_true"] = np.expm1(val[target])
    val["y_pred"] = np.expm1(val_pred_log)

    global_metrics = compute_metrics(val["y_true"].to_numpy(), val["y_pred"].to_numpy())
    global_metrics["r2_log"] = r2_log_score(val[target].to_numpy(), val_pred_log)
    by_band = error_matrix_by(val, "price_band", "y_true", "y_pred")
    by_cluster = error_matrix_by(val, "property_cluster", "y_true", "y_pred")

    return {
        "split_used": "val",
        "trained_on": "train+test combinados (refit final)",
        "n_train_test": len(train_test),
        "global": global_metrics,
        "by_price_band": by_band.to_dict(orient="records"),
        "by_property_cluster": by_cluster.to_dict(orient="records"),
        "final_artifact_path": final_artifact_path,
        "model_name": model_bundle["model_name"],
        "params": model_bundle["params"],
    }


def main() -> None:
    out = run_final_refit_pipeline()
    out["note"] = (
        "Tocado uma unica vez nesta execucao completa - depois de augmentation (fase 05, "
        "revertida na fase 12), features (fase 07-09), modelo (fase 10) e hipoteses (fase "
        "12) estarem todos travados. Nao usado para nenhuma decisao anterior (P1)."
    )
    Path("reports/val_final_check.json").write_text(json.dumps(out, indent=2, default=str))
    print(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    main()
