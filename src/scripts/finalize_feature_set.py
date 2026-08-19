"""Fase 07 — grava o conjunto final de features do modelo (só as `active` no registry) para a fase 08."""
import json
from pathlib import Path

MODEL_FEATURES = [
    "bedrooms", "bathrooms", "sqft_living", "sqft_lot", "floors", "waterfront", "view",
    "condition", "grade", "sqft_above", "sqft_basement", "sqft_living15", "sqft_lot15",
    "medn_hshld_incm_amt", "hous_val_amt", "per_bchlr",
    "property_age", "dist_to_seattle_center", "comps_knn_price", "local_grade_percentile",
    # as 6 candidatas novas das fases 07/08 foram testadas e REJEITADAS na fase 09 (revalidação com
    # mais folds: melhoria de ~1,4-1,5%, abaixo do corte de 5% definido como critério — ver
    # promote_features.py). Não entram aqui.
]
TARGET = "price_log"


def run_finalize_feature_set_pipeline(
    feature_metadata_path: str = "data/trusted/feature_metadata.json",
) -> dict:
    """Refatorada em função chamável (P5) — reusada tanto por `main()` (CLI) quanto por notebook."""
    Path(feature_metadata_path).write_text(
        json.dumps({"model_features": MODEL_FEATURES, "target": TARGET}, indent=2)
    )
    return {
        "feature_metadata_path": feature_metadata_path,
        "model_features": MODEL_FEATURES,
        "target": TARGET,
        "n_features": len(MODEL_FEATURES),
    }


def main() -> None:
    out = run_finalize_feature_set_pipeline()
    print(f"{out['n_features']} features oficiais gravadas em {out['feature_metadata_path']}")


if __name__ == "__main__":
    main()
