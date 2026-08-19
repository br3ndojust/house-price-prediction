"""Fase 10 — model_selection: grid Ridge+XGBoost via GroupKFold(zipcode), TRAIN (aumentado, fase 05)
only (P1, P4)."""
import json
import sys
from pathlib import Path

import joblib
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.models.train import build_model, run_model_selection, technical_tie_check


def run_model_selection_pipeline(
    features_parquet: str = "data/trusted/features_contextual.parquet",
    model_cfg_path: str = "configs/model.yaml",
    feature_metadata_path: str = "data/trusted/feature_metadata.json",
    candidate_artifact_path: str = "artifacts/model_candidate.pkl",
    feature_cols_override: list[str] | None = None,
    metric: str = "mae",
) -> dict:
    """Fase 10, refatorada em função chamável (P5) — reusada tanto pelo `main()` (CLI) quanto pelo
    `app/infrastructure/training/pipeline_training_service.py` (retraining via API, sem duplicar
    lógica nem subprocess).

    `feature_cols_override`: subconjunto das 20 features oficiais escolhido pelo usuário na tela de
    Treino do portal (fase 16) — quando informado, substitui `feature_meta["model_features"]` só
    nesta execução; nunca reescreve `data/trusted/feature_metadata.json` (o conjunto oficial
    permanece travado, P6 — isso só gera um candidato para avaliação, igual qualquer outro).

    `metric`: métrica usada para ranquear os candidatos deste treino (`mae` | `rmse` | `mape` | `r2`,
    ver `src.models.train.METRIC_KEYS`) — `mae` é a métrica oficial (P4); escolher outra só afeta
    qual candidato este job elege como vencedor, nunca o critério oficial de promoção."""
    model_cfg = yaml.safe_load(Path(model_cfg_path).read_text(encoding="utf-8"))
    feature_meta = json.loads(Path(feature_metadata_path).read_text())
    feature_cols = feature_cols_override if feature_cols_override else feature_meta["model_features"]
    target = feature_meta["target"]

    df = pd.read_parquet(features_parquet)
    train = df[df["split"] == "train"].reset_index(drop=True)

    grids = {
        "ridge": {"alpha": model_cfg["models"]["ridge"]["alpha"]},
        "xgboost": {
            "n_estimators": model_cfg["models"]["xgboost"]["n_estimators"],
            "max_depth": model_cfg["models"]["xgboost"]["max_depth"],
            "learning_rate": model_cfg["models"]["xgboost"]["learning_rate"],
        },
    }

    results = run_model_selection(
        train, feature_cols, target, model_cfg["cv"]["group_column"], grids,
        model_cfg["cv"]["n_splits"], model_cfg["cv"]["random_state"], metric=metric,
    )
    tie = technical_tie_check(results, metric=metric)

    best = results[0]
    final_model = build_model(best["model"], best["params"])
    final_model.fit(train[feature_cols], train[target])

    Path(candidate_artifact_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"model": final_model, "model_name": best["model"], "params": best["params"],
         "feature_cols": feature_cols, "target": target},
        candidate_artifact_path,
    )

    return {"cv_results": results, "technical_tie_check": tie, "winner": best,
            "candidate_artifact_path": candidate_artifact_path, "metric_used": metric}


def main() -> None:
    out = run_model_selection_pipeline()
    Path("reports/model_selection_report.json").write_text(json.dumps(out, indent=2, default=str))
    print(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    main()
