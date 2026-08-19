"""Fase 12 (iteração 1, delimitada) — hipótese motivada pelo Error Matrix da fase 11.

HIPÓTESE: a decisão de augmentation da fase 05 (Técnica B adotada por melhorar `GroupKFold` MAE dentro
de `train` em 20,5%) piorou o MAE em `test` (zipcode não visto) em 8,8% (fase 11). Reverter para o
`train` original (sem augmentation), rodando o pipeline COMPLETO (segmentação + features + modelo, não
só o diagnóstico simplificado da fase 05), deve reduzir o MAE em `test`.

CRITÉRIO DE ACEITE (definido antes do experimento, P4): aqui, diferente da fase 05, `test` é
DELIBERADAMENTE a métrica decisiva — a fase 11 mostrou que a métrica de CV em `train` não é proxy
confiável pra generalização geográfica neste caso, e o próprio propósito desta hipótese é checar
generalização. Reverte para sem augmentation se isso reduzir o MAE global em `test`. Uma única
iteração — critério de parada do roadmap, seção 10.
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.augmentation.perturbation import controlled_perturbation_augment
from src.evaluation.error_matrix import compute_metrics, error_matrix_by
from src.features.comparable import apply_comps_knn_price, fit_spatial_index
from src.features.neighborhood import add_dist_to_seattle_center
from src.features.raw import add_property_age
from src.features.relative_position import apply_local_grade_percentile
from src.models.train import build_model
from src.segmentation.price_bands import SEMANTIC_LABELS, apply_price_quartile, fit_price_quartiles
from src.segmentation.property_clusters import (
    PHYSICAL_PROFILE_COLUMNS,
    apply_property_cluster,
    fit_property_cluster,
    scan_property_cluster_k,
)

FEATURE_COLS = [
    "bedrooms", "bathrooms", "sqft_living", "sqft_lot", "floors", "waterfront", "view",
    "condition", "grade", "sqft_above", "sqft_basement", "sqft_living15", "sqft_lot15",
    "medn_hshld_incm_amt", "hous_val_amt", "per_bchlr",
    "property_age", "dist_to_seattle_center", "comps_knn_price", "local_grade_percentile",
]
TARGET = "price_log"
K_SPATIAL = 30


def build_pipeline(train_raw: pd.DataFrame, test_raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Roda segmentação (fit em train) + features contextuais (fit em train) — mesma lógica de
    `src/scripts/run_segmentation.py`/`src/scripts/generate_features.py`, aplicada a um par (train, test)."""
    train = add_property_age(train_raw.copy())
    test = add_property_age(test_raw.copy())

    breakpoints = fit_price_quartiles(train[TARGET])
    for df_ in (train, test):
        df_["price_quartile"] = apply_price_quartile(df_[TARGET], breakpoints)
        df_["price_band"] = df_["price_quartile"].map(SEMANTIC_LABELS)

    k_scan = scan_property_cluster_k(train, range(2, 7), random_state=42)
    best_k = int(k_scan.loc[k_scan["silhouette"].idxmax(), "k"])
    scaler, km = fit_property_cluster(train, best_k, random_state=42)
    for df_ in (train, test):
        df_["property_cluster"] = apply_property_cluster(df_, scaler, km)

    train = add_dist_to_seattle_center(train)
    test = add_dist_to_seattle_center(test)

    index = fit_spatial_index(train, k=K_SPATIAL)
    train["comps_knn_price"] = apply_comps_knn_price(train, index)
    train["local_grade_percentile"] = apply_local_grade_percentile(train, index)
    test["comps_knn_price"] = apply_comps_knn_price(test, index)
    test["local_grade_percentile"] = apply_local_grade_percentile(test, index)

    return train, test


def evaluate_variant(train_raw: pd.DataFrame, test_raw: pd.DataFrame, winner_params: dict) -> dict:
    train, test = build_pipeline(train_raw, test_raw)

    model = build_model(winner_params["model"], winner_params["params"])
    model.fit(train[FEATURE_COLS], train[TARGET])

    test = test.copy()
    test["y_true"] = np.expm1(test[TARGET])
    test["y_pred"] = np.expm1(model.predict(test[FEATURE_COLS]))

    global_metrics = compute_metrics(test["y_true"].to_numpy(), test["y_pred"].to_numpy())
    by_band = error_matrix_by(test, "price_band", "y_true", "y_pred")
    return {"global": global_metrics, "by_band": by_band.to_dict(orient="records")}


def run_revert_augmentation_hypothesis_pipeline(
    house_clean_parquet: str = "data/processed/house_clean.parquet",
    split_assignment_parquet: str = "data/processed/split_assignment.parquet",
    model_selection_report_path: str = "reports/model_selection_report.json",
) -> dict:
    """Fase 12, refatorada em função chamável (P5) — reusada tanto por `main()` (CLI) quanto pelo
    notebook `notebooks/12_hypothesis_revert_augmentation_iter2.ipynb`."""
    house = pd.read_parquet(house_clean_parquet)
    split = pd.read_parquet(split_assignment_parquet)
    full = house.merge(split[["id", "date", "split"]], on=["id", "date"], how="left")

    train_no_aug = full[full["split"] == "train"].reset_index(drop=True)
    # regenerado pela mesma funcao/parametros da fase 05 (nao le train_for_pipeline.parquet, que pode
    # ja ter sido revertido por uma execucao anterior desta fase) - reproducibilidade determinista.
    train_adopted = controlled_perturbation_augment(train_no_aug, augment_fraction=0.5, random_state=42)
    test = full[full["split"] == "test"].reset_index(drop=True)

    winner_params = json.loads(Path(model_selection_report_path).read_text(encoding="utf-8"))["winner"]

    results = {
        "no_augmentation": evaluate_variant(train_no_aug, test, winner_params),
        "adopted_technique_b": evaluate_variant(train_adopted, test, winner_params),
    }

    mae_no_aug = results["no_augmentation"]["global"]["mae"]
    mae_adopted = results["adopted_technique_b"]["global"]["mae"]
    revert = mae_no_aug < mae_adopted

    return {
        "hypothesis": "reverter para train sem augmentation reduz MAE em test (pipeline completo)",
        "acceptance_criteria": "test e deliberadamente decisivo aqui (ver docstring) - reverte se "
                                "MAE sem augmentation < MAE com Tecnica B",
        "results": results,
        "mae_no_augmentation": mae_no_aug,
        "mae_adopted_technique_b": mae_adopted,
        "decision": "REVERT_TO_NO_AUGMENTATION" if revert else "KEEP_TECHNIQUE_B",
        "stopping_rule_applied": "fase12 encerrada apos 1 iteracao (roadmap secao 10)",
    }


def main() -> None:
    out = run_revert_augmentation_hypothesis_pipeline()
    Path("reports/fase12_experiment.json").write_text(json.dumps(out, indent=2, default=str))
    print(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    main()
