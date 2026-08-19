"""Fase 04 — geographic_coverage: o split tem zipcode/região sub-representada em train? Isso
correlaciona com erro? Produz o veredito (go/no-go) para o experimento de augmentation da fase 05.

`val` deliberadamente NÃO entra nesta análise (mesmo sendo só medição de distância/cobertura, sem
target) — reforça P1 (val só é tocado uma vez, na fase 13). Só train x test.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from xgboost import XGBRegressor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.evaluation.geographic_coverage import (
    data_scarcity_buckets,
    distribution_comparison,
    feature_space_coverage,
    nearest_train_zip_distance,
)
from src.validation.ablation import ABLATION_MODEL_PARAMS

BASELINE_FEATURES = [
    "bedrooms", "bathrooms", "sqft_living", "sqft_lot", "floors", "waterfront", "view",
    "condition", "grade", "sqft_above", "sqft_basement", "sqft_living15", "sqft_lot15",
    "medn_hshld_incm_amt", "hous_val_amt", "per_bchlr",
]


def run_geographic_coverage_pipeline(
    house_clean_parquet: str = "data/processed/house_clean.parquet",
    split_assignment_parquet: str = "data/processed/split_assignment.parquet",
) -> dict:
    """Fase 04, refatorada em função chamável (P5) — reusada tanto por `main()` (CLI) quanto por
    notebook."""
    house = pd.read_parquet(house_clean_parquet)
    split = pd.read_parquet(split_assignment_parquet)
    df = house.merge(split[["id", "date", "split"]], on=["id", "date"], how="left")
    assert len(df) == len(house)

    train = df[df["split"] == "train"].reset_index(drop=True)
    test = df[df["split"] == "test"].reset_index(drop=True)

    # 1. Distribuicao comparativa (so train x test, val fica de fora, ver docstring)
    dev = df[df["split"].isin(["train", "test"])]
    dist_cols = ["price_log", "sqft_living", "grade", "lat", "long", "waterfront", "view"]
    dist_comparison = distribution_comparison(dev, "split", dist_cols)

    # 2. Distancia geografica: zip de test -> zip de train mais proximo
    geo_distance = nearest_train_zip_distance(train, test)

    # 3. Cobertura no espaco de features (fit em train)
    coverage = feature_space_coverage(train, test, BASELINE_FEATURES)

    # 4. Baldes de escassez por zip em train
    scarcity = data_scarcity_buckets(train)
    low_volume_zips = scarcity[scarcity["scarcity_bucket"] == "LOW"]["zipcode"].tolist()

    # 5. Baseline rapido (features basicas, sem engenharia ainda) treinado em train, erro em test
    model = XGBRegressor(**ABLATION_MODEL_PARAMS, n_jobs=-1)
    model.fit(train[BASELINE_FEATURES], train["price_log"])
    pred_log = model.predict(test[BASELINE_FEATURES])
    abs_err_dollar = np.abs(np.expm1(test["price_log"]) - np.expm1(pred_log))

    test_diag = test[["id", "zipcode"]].copy()
    test_diag["abs_error"] = abs_err_dollar.values
    test_diag = test_diag.merge(
        geo_distance[["zipcode", "distance_to_nearest_train_zip"]], on="zipcode", how="left"
    )
    test_diag["feature_space_distance"] = coverage["nearest_train_neighbor_distance"].values

    corr_geo_dist, p_geo_dist = pearsonr(test_diag["distance_to_nearest_train_zip"], test_diag["abs_error"])
    corr_feat_dist, p_feat_dist = pearsonr(test_diag["feature_space_distance"], test_diag["abs_error"])

    # 6. Arvore de decisao (definida com o usuario antes de rodar a fase 05)
    train_has_underrepresented = len(low_volume_zips) > 0
    error_correlates_with_scarcity = bool(
        (corr_geo_dist > 0.2 and p_geo_dist < 0.05) or (corr_feat_dist > 0.2 and p_feat_dist < 0.05)
    )
    augmentation_hypothesis = "STRONG" if (train_has_underrepresented and error_correlates_with_scarcity) \
        else "WEAK"

    out = {
        "distribution_comparison": dist_comparison.to_dict(orient="records"),
        "geo_distance_test_zips": geo_distance.to_dict(orient="records"),
        "n_low_volume_zips_in_train": len(low_volume_zips),
        "low_volume_zips_in_train": low_volume_zips,
        "scarcity_buckets_counts": scarcity["scarcity_bucket"].value_counts().to_dict(),
        "baseline_diagnostic_model": "XGBoost (ABLATION_MODEL_PARAMS), basic features, no augmentation",
        "correlation_error_vs_geo_distance": {"r": corr_geo_dist, "p": p_geo_dist},
        "correlation_error_vs_feature_space_distance": {"r": corr_feat_dist, "p": p_feat_dist},
        "decision_tree": {
            "train_has_underrepresented_regions": train_has_underrepresented,
            "error_correlates_with_scarcity_signals": error_correlates_with_scarcity,
            "augmentation_hypothesis": augmentation_hypothesis,
        },
        "note": (
            "val excluido desta analise (P1 - val tocado uma unica vez, na fase 13). Veredito orienta "
            "o criterio de aceite da fase 05, mas as 2 tecnicas de augmentation sao testadas de "
            "qualquer forma (requisito explicito do usuario)."
        ),
    }
    return out


def main() -> None:
    out = run_geographic_coverage_pipeline()
    Path("reports/geographic_coverage.json").write_text(json.dumps(out, indent=2, default=str))
    print(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    main()
