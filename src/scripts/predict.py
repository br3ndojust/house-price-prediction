"""Fase 14 — demo de inferência sobre `future_unseen_examples.csv` via `src/pipelines/inference.py`
(paridade treino/produção, P5). Usa `artifacts/model_final.pkl` (refit `train`+`test`, fase 13)."""
import json
import sys
from datetime import date
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.pipelines.inference import load_production_artifacts, predict_price

DEMO_AS_OF_DATE = date(2026, 8, 16)  # data de referencia fixa, reprodutibilidade do audit


def run_inference_demo_pipeline(
    raw_csv_path: str = "data/raw/future_unseen_examples.csv",
    predictions_csv_path: str = "reports/future_unseen_predictions.csv",
    summary_report_path: str = "reports/inference_demo_report.json",
    as_of_date: date = DEMO_AS_OF_DATE,
) -> dict:
    """Fase 14, refatorada em função chamável (P5) — reusada tanto por `main()` (CLI) quanto por
    notebook, sem duplicar a lógica de paridade treino/produção via `src/pipelines/inference.py`."""
    raw = pd.read_csv(raw_csv_path)
    model_bundle, spatial_index, demographics = load_production_artifacts()

    predictions = predict_price(raw, model_bundle, spatial_index, demographics, as_of_date)

    out = raw.copy()
    out["predicted_price"] = predictions
    Path(predictions_csv_path).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(predictions_csv_path, index=False)

    summary = {
        "n_predictions": len(predictions),
        "as_of_date": as_of_date.isoformat(),
        "predicted_price_summary": {
            "min": float(predictions.min()), "median": float(pd.Series(predictions).median()),
            "mean": float(predictions.mean()), "max": float(predictions.max()),
        },
    }
    Path(summary_report_path).write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    summary = run_inference_demo_pipeline()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
