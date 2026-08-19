"""Fase 01 — data_understanding: limpeza + merge -> data/processed/house_clean.parquet."""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.clean import clean_kc_house_data
from src.data.merge import merge_demographics


def run_dataset_build_pipeline(
    kc_house_data_csv: str = "data/raw/kc_house_data.csv",
    zipcode_demographics_csv: str = "data/raw/zipcode_demographics.csv",
    house_clean_parquet_path: str = "data/processed/house_clean.parquet",
    build_dataset_log_path: str = "reports/build_dataset_log.json",
) -> dict:
    """Fase 01, refatorada em função chamável (P5) — reusada tanto por `main()` (CLI) quanto por
    notebook."""
    house = pd.read_csv(kc_house_data_csv)
    demo = pd.read_csv(zipcode_demographics_csv)

    house_clean, clean_log = clean_kc_house_data(house)
    merged = merge_demographics(house_clean, demo)

    Path("data/processed").mkdir(parents=True, exist_ok=True)
    merged.to_parquet(house_clean_parquet_path, index=False)

    clean_log["n_rows_in"] = len(house)
    clean_log["n_columns_out"] = merged.shape[1]
    Path(build_dataset_log_path).write_text(
        json.dumps(clean_log, indent=2), encoding="utf-8"
    )
    return clean_log


def main() -> None:
    out = run_dataset_build_pipeline()
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
