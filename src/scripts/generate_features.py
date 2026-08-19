"""Fases 07+08 — feature engineering RAW/DERIVED (07) e CONTEXTUAL (08). `.fit` (índice espacial)
só em TRAIN (P1) — `train` aqui já é o escolhido pela fase 05 (augmentation)."""
import json
import sys
from pathlib import Path

import joblib
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.features.comparable import (
    apply_comps_knn_neighbor_distance,
    apply_comps_knn_price,
    apply_local_price_dispersion,
    fit_spatial_index,
)
from src.features.derived import (
    add_basement_features,
    add_bathrooms_per_bedroom,
    add_grade_condition_interaction,
    add_sqft_living_to_lot_ratio,
)
from src.features.neighborhood import (
    add_dist_to_seattle_center,
    apply_dist_to_nearest_train_zip,
    fit_train_zip_index,
)
from src.features.physical_typicality import add_property_cluster_distance
from src.features.raw import add_log_sqft_lot, add_renovation_features
from src.features.relative_position import apply_local_grade_percentile


def run_raw_derived_features_pipeline(
    house_segments_parquet: str = "data/trusted/house_segments.parquet",
    output_parquet: str = "data/trusted/features_raw_derived.parquet",
    property_cluster_model_path: str = "artifacts/property_cluster_model.pkl",
) -> dict:
    """Fase 07, refatorada em função chamável (P5) — reusada tanto por `main()` (CLI) quanto por
    notebook (`07_raw_derived_features.ipynb`). RAW/DERIVED (row-wise, sem fit) + `property_cluster_distance`
    (reusa artefato já fitado em TRAIN na fase 06, não fita nada novo aqui)."""
    df = pd.read_parquet(house_segments_parquet)

    # --- Fase 05: RAW/DERIVED (row-wise, sem fit) ---
    df = add_renovation_features(df)
    df = add_basement_features(df)
    df = add_grade_condition_interaction(df)
    df = add_log_sqft_lot(df)

    # --- Candidatas novas (extensão dirigida pelos achados das fases 04/06) ---
    df = add_bathrooms_per_bedroom(df)
    df = add_sqft_living_to_lot_ratio(df)
    df = add_property_cluster_distance(df, property_cluster_model_path)

    Path(output_parquet).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_parquet, index=False)

    new_features = [
        "was_renovated", "years_since_renovation", "has_basement", "basement_ratio",
        "grade_condition_interaction", "log_sqft_lot",
        "bathrooms_per_bedroom", "sqft_living_to_lot_ratio", "property_cluster_distance",
    ]
    return {"n_rows": len(df), "n_columns": df.shape[1], "new_features": new_features,
            "output_parquet": output_parquet}


def run_contextual_features_pipeline(
    raw_derived_parquet: str = "data/trusted/features_raw_derived.parquet",
    comparable_cfg_path: str = "configs/comparable.yaml",
    features_contextual_parquet: str = "data/trusted/features_contextual.parquet",
    spatial_index_path: str = "artifacts/spatial_index.pkl",
) -> dict:
    """Fase 08, refatorada em função chamável (P5) — reusada tanto por `main()` (CLI) quanto por
    notebook (`08_contextual_features.ipynb`). CONTEXTUAL (fit espacial só em train). Lê o parquet
    de RAW/DERIVED do disco (em vez de receber o df em memória) para poder rodar de forma
    independente da fase 07, caso `run_raw_derived_features_pipeline` já tenha rodado antes."""
    cfg = yaml.safe_load(Path(comparable_cfg_path).read_text(encoding="utf-8"))["comparable"]
    k = cfg["k"]

    df = pd.read_parquet(raw_derived_parquet)

    # --- Fase 06: CONTEXTUAL (fit espacial só em train) ---
    df = add_dist_to_seattle_center(df)

    train = df[df["split"] == "train"]
    index = fit_spatial_index(train, k=k)
    df["comps_knn_price"] = apply_comps_knn_price(df, index)
    df["local_grade_percentile"] = apply_local_grade_percentile(df, index)

    # --- Candidatas novas (extensão dirigida pelos achados das fases 04/08) ---
    train_zip_index = fit_train_zip_index(train)
    df["dist_to_nearest_train_zip"] = apply_dist_to_nearest_train_zip(df, train_zip_index)
    df["comps_knn_neighbor_distance"] = apply_comps_knn_neighbor_distance(df, index)
    df["local_price_dispersion"] = apply_local_price_dispersion(df, index)

    df.to_parquet(features_contextual_parquet, index=False)
    joblib.dump({"index": index, "k": k, "train_zip_index": train_zip_index}, spatial_index_path)

    new_features = [
        "dist_to_seattle_center", "comps_knn_price", "local_grade_percentile",
        "dist_to_nearest_train_zip", "comps_knn_neighbor_distance", "local_price_dispersion",
    ]
    return {"n_rows": len(df), "n_columns": df.shape[1], "new_features": new_features,
            "features_contextual_parquet": features_contextual_parquet,
            "spatial_index_path": spatial_index_path}


def main() -> None:
    raw_derived_out = run_raw_derived_features_pipeline()
    contextual_out = run_contextual_features_pipeline()

    new_features = raw_derived_out["new_features"] + contextual_out["new_features"]
    summary = {
        "n_rows": contextual_out["n_rows"],
        "n_columns": contextual_out["n_columns"],
        "new_features": new_features,
    }
    Path("reports/feature_generation_report.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
