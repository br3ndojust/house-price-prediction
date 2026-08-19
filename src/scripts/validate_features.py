"""Fase 09 — feature_validation: leakage + ablation antes de promover features (P2, P4)."""
import json
import sys
from pathlib import Path

import joblib
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.validation.ablation import run_ablation_cv
from src.validation.leakage import check_spatial_index_fit_size, train_test_correlation_gap

BASELINE_FEATURES = [
    "bedrooms", "bathrooms", "sqft_living", "sqft_lot", "floors", "waterfront", "view",
    "condition", "grade", "sqft_above", "sqft_basement", "sqft_living15", "sqft_lot15",
    "medn_hshld_incm_amt", "hous_val_amt", "per_bchlr",
]
RAW_DERIVED_FEATURES = [
    "was_renovated", "years_since_renovation", "has_basement", "basement_ratio",
    "grade_condition_interaction", "log_sqft_lot",
]
CONTEXTUAL_FEATURES = ["dist_to_seattle_center", "comps_knn_price", "local_grade_percentile"]

# candidatas novas (fases 07/08) — dirigidas pelas extensões exploratórias das fases 04/06, nunca
# testadas em ablation formal ainda. Separadas de RAW_DERIVED_FEATURES/CONTEXTUAL_FEATURES (que já
# passaram por essa decisão) pra não misturar um conjunto já decidido com um ainda em teste.
NEW_RAW_DERIVED_FEATURES = ["bathrooms_per_bedroom", "sqft_living_to_lot_ratio", "property_cluster_distance"]
NEW_CONTEXTUAL_FEATURES = ["dist_to_nearest_train_zip", "comps_knn_neighbor_distance", "local_price_dispersion"]

TARGET = "price_log"
GROUP_COL = "zipcode"


def run_feature_validation_pipeline(
    model_cfg_path: str = "configs/model.yaml",
    features_parquet: str = "data/trusted/features_contextual.parquet",
    spatial_index_path: str = "artifacts/spatial_index.pkl",
    report_path: str = "reports/ablation_fase09.json",
) -> dict:
    """Fase 09, refatorada em função chamável (P5) — reusada tanto por `main()` (CLI) quanto por
    notebook (`09_feature_validation.ipynb`)."""
    model_cfg = yaml.safe_load(Path(model_cfg_path).read_text(encoding="utf-8"))["cv"]
    df = pd.read_parquet(features_parquet)
    train = df[df["split"] == "train"].reset_index(drop=True)

    # --- Leakage estrutural: spatial index fit só com linhas de TRAIN ---
    spatial = joblib.load(spatial_index_path)
    leakage_checks = {
        "spatial_index_fit_size_matches_train": check_spatial_index_fit_size(
            spatial["index"], len(train)
        ),
    }
    for feature in CONTEXTUAL_FEATURES:
        leakage_checks[feature] = train_test_correlation_gap(df, feature, TARGET)
    for feature in NEW_CONTEXTUAL_FEATURES:
        leakage_checks[feature] = train_test_correlation_gap(df, feature, TARGET)

    # --- Ablation: baseline -> +raw_derived -> +contextual -> +tudo ---
    best_known = BASELINE_FEATURES + CONTEXTUAL_FEATURES + ["property_age"]
    feature_sets = {
        "baseline": BASELINE_FEATURES,
        "baseline+raw_derived": BASELINE_FEATURES + RAW_DERIVED_FEATURES,
        "baseline+contextual": BASELINE_FEATURES + CONTEXTUAL_FEATURES,
        "baseline+raw_derived+contextual": BASELINE_FEATURES + RAW_DERIVED_FEATURES + CONTEXTUAL_FEATURES,
        # property_age (fase 03) nunca foi testada formalmente para o modelo até esta fase (P4)
        "baseline+contextual+property_age": best_known,
        # candidatas novas (fases 07/08) testadas em cima do melhor conjunto já decidido (best_known),
        # não do baseline puro — o que importa é se melhoram o que já ganhou, não isoladas (P4)
        "best_known+new_raw_derived": best_known + NEW_RAW_DERIVED_FEATURES,
        "best_known+new_contextual": best_known + NEW_CONTEXTUAL_FEATURES,
        "best_known+all_new_candidates": best_known + NEW_RAW_DERIVED_FEATURES + NEW_CONTEXTUAL_FEATURES,
    }
    ablation_results = {}
    for name, cols in feature_sets.items():
        ablation_results[name] = run_ablation_cv(
            train, cols, TARGET, GROUP_COL,
            n_splits=model_cfg["n_splits"], random_state=model_cfg["random_state"],
        )

    out = {"leakage_checks": leakage_checks, "ablation": ablation_results}
    Path(report_path).write_text(json.dumps(out, indent=2, default=str))
    return out


def main() -> None:
    out = run_feature_validation_pipeline()
    print(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    main()
