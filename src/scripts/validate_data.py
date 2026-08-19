"""Fase 00 — contrato de schema/nulos/cardinalidade dos 3 CSVs de entrada. Ver P1."""
import json
import sys
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.contracts import (
    ContractReport,
    check_future_unseen,
    check_kc_house_data,
    check_zipcode_demographics,
)


def run_data_validation_pipeline(
    data_contract_cfg_path: str = "configs/data_contract.yaml",
) -> dict:
    """Fase 00, refatorada em função chamável (P5) — reusada tanto por `main()` (CLI) quanto por
    notebook."""
    cfg = yaml.safe_load(Path(data_contract_cfg_path).read_text(encoding="utf-8"))
    report = ContractReport()

    kc = pd.read_csv(cfg["kc_house_data"]["path"])
    check_kc_house_data(kc, cfg["kc_house_data"], report)

    demo = pd.read_csv(cfg["zipcode_demographics"]["path"])
    check_zipcode_demographics(demo, cfg["zipcode_demographics"], report)

    future = pd.read_csv(cfg["future_unseen_examples"]["path"])
    check_future_unseen(future, cfg["future_unseen_examples"], report)

    out = {
        "ok": report.ok,
        "violations": [v.__dict__ for v in report.violations],
        "row_counts": {"kc_house_data": len(kc), "zipcode_demographics": len(demo),
                        "future_unseen_examples": len(future)},
    }
    return out


def main() -> int:
    out = run_data_validation_pipeline()
    Path("reports").mkdir(exist_ok=True)
    Path("reports/data_validation_report.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8"
    )
    print(json.dumps(out, indent=2))
    return 0 if out["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
