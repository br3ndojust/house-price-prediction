"""Fase 02 — canonical_split: GroupShuffleSplit por zipcode -> split_metadata.json. val é sagrado."""
import json
import sys
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.validation.split import canonical_split, split_summary


def run_split_pipeline(
    split_cfg_path: str = "configs/split.yaml",
    house_clean_parquet_path: str = "data/processed/house_clean.parquet",
    split_assignment_parquet_path: str = "data/processed/split_assignment.parquet",
    split_metadata_path: str = "data/processed/split_metadata.json",
) -> dict:
    """Fase 02, refatorada em função chamável (P5) — reusada tanto por `main()` (CLI) quanto por
    notebook."""
    cfg = yaml.safe_load(Path(split_cfg_path).read_text(encoding="utf-8"))["split"]
    group_column = yaml.safe_load(Path(split_cfg_path).read_text(encoding="utf-8"))["group_column"]
    random_state = yaml.safe_load(Path(split_cfg_path).read_text(encoding="utf-8"))["random_state"]

    df = pd.read_parquet(house_clean_parquet_path)
    split = canonical_split(df, group_column, cfg["train"], cfg["test"], cfg["val"], random_state)

    # id sozinho não é chave única (revendas, ver fase 00/01) — carrega "date" junto para que o
    # merge downstream (fase 03+) use (id, date) como chave, nunca id isolado.
    df_out = df[["id", "date", group_column]].copy()
    df_out["split"] = split.values
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    df_out.to_parquet(split_assignment_parquet_path, index=False)

    summary = split_summary(df, split, group_column)

    # Check: resales of the same property (same id) never span multiple split parts.
    resale_check = (
        pd.DataFrame({"id": df["id"], "split": split})
        .groupby("id")["split"].nunique()
    )
    resales_spanning_splits = int((resale_check > 1).sum())
    summary["resales_spanning_splits"] = resales_spanning_splits

    metadata = {
        "config": {"split": cfg, "group_column": group_column, "random_state": random_state},
        "summary": summary,
    }
    Path(split_metadata_path).write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return metadata


def main() -> None:
    out = run_split_pipeline()
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
