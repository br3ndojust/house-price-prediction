"""Fase 02 — EDA completo sobre 100% dos dados, ANTES do split (P1: descritivo, sem `.fit`).

Substitui a antiga fase 04 (que rodava só em `train`, depois do split) — mesma análise de
correlação/outlier/padrão geográfico, agora sobre o dataset inteiro, mais descoberta de representação
por zipcode (usada como insumo para a fase 04, `geographic_coverage`)."""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.evaluation.geographic_coverage import zip_representation_summary


def run_eda_pipeline(
    house_clean_parquet_path: str = "data/processed/house_clean.parquet",
    zip_representation_summary_path: str = "reports/zip_representation_summary.csv",
    eda_summary_path: str = "reports/eda_summary.json",
) -> dict:
    """Fase 02, refatorada em função chamável (P5) — reusada tanto por `main()` (CLI) quanto por
    notebook."""
    df = pd.read_parquet(house_clean_parquet_path)

    numeric_cols = [
        "price_log", "bedrooms", "bathrooms", "sqft_living", "sqft_lot", "floors", "waterfront",
        "view", "condition", "grade", "sqft_above", "sqft_basement",
        "sqft_living15", "sqft_lot15", "medn_hshld_incm_amt", "hous_val_amt", "per_bchlr",
    ]
    corr = df[numeric_cols].corr()["price_log"].drop("price_log").sort_values(key=abs, ascending=False)

    zip_summary = zip_representation_summary(df)
    low_volume_zips = zip_summary[zip_summary["n"] < 100]

    summary = {
        "n_rows": len(df),
        "n_zipcodes": int(df["zipcode"].nunique()),
        "correlation_with_price_log": corr.to_dict(),
        "zip_summary_min_n": int(zip_summary["n"].min()),
        "zip_summary_max_n": int(zip_summary["n"].max()),
        "n_zipcodes_below_100_properties": int(len(low_volume_zips)),
        "low_volume_zipcodes": low_volume_zips["zipcode"].tolist(),
    }

    Path("reports").mkdir(exist_ok=True)
    zip_summary.to_csv(zip_representation_summary_path, index=False)
    Path(eda_summary_path).write_text(json.dumps(summary, indent=2, default=str))
    return summary


def main() -> None:
    out = run_eda_pipeline()
    print(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    main()
