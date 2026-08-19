"""Fase 06 — market_property_segmentation: bandas de preço + cluster de perfil físico.

`train` vem de `data/trusted/train_for_pipeline.parquet` (decisão da fase 05 — original ou aumentado,
conforme o experimento de augmentation), nunca do split bruto diretamente."""
import json
import sys
from pathlib import Path

import joblib
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.segmentation.price_bands import (
    SEMANTIC_LABELS,
    apply_price_quartile,
    band_separation_report,
    fit_price_quartiles,
)
from src.segmentation.property_clusters import (
    add_property_age,
    apply_property_cluster,
    fit_property_cluster,
    scan_property_cluster_k,
)


def run_segmentation_pipeline(
    house_clean_parquet: str = "data/processed/house_clean.parquet",
    split_assignment_parquet: str = "data/processed/split_assignment.parquet",
    train_for_pipeline_parquet: str = "data/trusted/train_for_pipeline.parquet",
    house_segments_parquet: str = "data/trusted/house_segments.parquet",
    price_quartiles_json: str = "data/trusted/price_quartiles.json",
    property_cluster_model_path: str = "artifacts/property_cluster_model.pkl",
) -> dict:
    """Fase 06, refatorada em função chamável (P5) — reusada tanto por `main()` (CLI) quanto por
    notebook."""
    house = pd.read_parquet(house_clean_parquet)
    split = pd.read_parquet(split_assignment_parquet)
    # merge por (id, date), nunca id isolado — id sozinho repete em revendas (fase 00/01)
    n_before = len(house)
    full = house.merge(split[["id", "date", "split"]], on=["id", "date"], how="left")
    assert len(full) == n_before, f"merge alterou o numero de linhas: {n_before} -> {len(full)}"

    # train vem do artefato da fase 05 (augmentation), test/val vem do split bruto (sempre reais)
    train_chosen = pd.read_parquet(train_for_pipeline_parquet)
    test_val = full[full["split"] != "train"]
    df = pd.concat([train_chosen, test_val], ignore_index=True)
    df = add_property_age(df)

    train = df[df["split"] == "train"]

    # --- Price bands (fit só em train) ---
    breakpoints = fit_price_quartiles(train["price_log"])
    df["price_quartile"] = apply_price_quartile(df["price_log"], breakpoints)
    df["price_band"] = df["price_quartile"].map(SEMANTIC_LABELS)

    separation = band_separation_report(
        train.assign(price_quartile=apply_price_quartile(train["price_log"], breakpoints)),
        "price_quartile", median_cols=["grade", "sqft_living"], rate_cols=["waterfront", "view"],
    )

    # --- Property clusters (fit só em train, sem price/lat/long/zipcode) ---
    k_scan = scan_property_cluster_k(train, range(2, 7), random_state=42)
    best_k = int(k_scan.loc[k_scan["silhouette"].idxmax(), "k"])
    scaler, km = fit_property_cluster(train, best_k, random_state=42)
    df["property_cluster"] = apply_property_cluster(df, scaler, km)

    Path(house_segments_parquet).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(house_segments_parquet, index=False)

    Path(price_quartiles_json).write_text(
        json.dumps({"breakpoints": breakpoints.tolist(), "labels": SEMANTIC_LABELS}, indent=2),
        encoding="utf-8",
    )
    joblib.dump({"scaler": scaler, "kmeans": km}, property_cluster_model_path)

    from src.segmentation.property_clusters import PHYSICAL_PROFILE_COLUMNS
    cluster_profile = df.groupby("property_cluster")[PHYSICAL_PROFILE_COLUMNS].median()

    summary = {
        "price_band_separation": separation.to_dict(),
        "k_scan": k_scan.to_dict(orient="records"),
        "best_k": best_k,
        "price_band_counts": df["price_band"].value_counts().to_dict(),
        "property_cluster_counts": df["property_cluster"].value_counts().to_dict(),
        "property_cluster_profile_median": cluster_profile.to_dict(orient="index"),
    }
    return summary


def main() -> None:
    summary = run_segmentation_pipeline()
    Path("reports/segmentation_report.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
