"""Fase 10 (iteração 1, delimitada) — hipótese motivada pelo Error Matrix da fase 09.

Baseline de produção real: k=30 (`configs/comparable.yaml`, usado em toda a fase 05-09 — nota: os
notebooks/relatórios das fases 06/07/09 tinham um erro de digitação afirmando "k=15" na narrativa; o
código sempre leu o config corretamente, só o texto estava errado, corrigido nesta fase, ver AUDIT_LOG).

HIPÓTESE: comps_knn_price/local_grade_percentile usam k=30 vizinhos espaciais. Imóveis de
luxo/waterfront são geograficamente esparsos (fase 03: só 163 imóveis no cluster físico de luxo em todo
o dataset) — um k ainda maior pode estabilizar mais a estimativa nesses casos esparsos, melhorando o
segmento que mais precisa dela (fase 09: MAE 4x pior em waterfront). Testamos k menor (15, mais local) e
maior (50, mais suavizado) contra o baseline de produção (30).

CRITÉRIO DE ACEITE (definido ANTES do experimento, P4): promover k diferente do baseline só se MAE de
`Luxury` em TEST cair >=5% E MAE global não piorar mais que 1%. Uma única iteração — não um loop aberto
perseguindo luxo (critério de parada do roadmap, seção 10).
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.evaluation.error_matrix import compute_metrics, error_matrix_by
from src.features.comparable import apply_comps_knn_price, fit_spatial_index
from src.features.neighborhood import add_dist_to_seattle_center
from src.features.relative_position import apply_local_grade_percentile

CANDIDATE_K_VALUES = [15, 30, 50]
BASELINE_K = 30  # producao real (configs/comparable.yaml)


def build_variant(df: pd.DataFrame, k: int) -> pd.DataFrame:
    train = df[df["split"] == "train"]
    index = fit_spatial_index(train, k=k)
    out = df.copy()
    out["comps_knn_price"] = apply_comps_knn_price(out, index)
    out["local_grade_percentile"] = apply_local_grade_percentile(out, index)
    return out


def run_k_sweep_hypothesis_pipeline(
    house_segments_parquet: str = "data/trusted/house_segments.parquet",
    feature_metadata_path: str = "data/trusted/feature_metadata.json",
    model_selection_report_path: str = "reports/model_selection_report.json",
    ledger_path: str = "reports/experiments/ledger.csv",
) -> dict:
    """Fase 10, refatorada em função chamável (P5) — reusada tanto por `main()` (CLI) quanto pelo
    notebook `notebooks/12_hypothesis_k_sweep_iter1.ipynb`."""
    feature_meta = json.loads(Path(feature_metadata_path).read_text(encoding="utf-8"))
    feature_cols = feature_meta["model_features"]
    target = feature_meta["target"]

    base = pd.read_parquet(house_segments_parquet)
    base = add_dist_to_seattle_center(base)

    winner_params = json.loads(Path(model_selection_report_path).read_text(encoding="utf-8"))["winner"]

    ledger_rows = []
    comparison = {}
    baseline_global_mae = None
    baseline_luxury_mae = None

    for k in CANDIDATE_K_VALUES:
        variant = build_variant(base, k)
        train = variant[variant["split"] == "train"]
        test = variant[variant["split"] == "test"].copy()

        model = XGBRegressor(**winner_params["params"], random_state=42, n_jobs=-1)
        model.fit(train[feature_cols], train[target])

        test["y_true"] = np.expm1(test[target])
        test["y_pred"] = np.expm1(model.predict(test[feature_cols]))

        global_metrics = compute_metrics(test["y_true"].to_numpy(), test["y_pred"].to_numpy())
        by_band = error_matrix_by(test, "price_band", "y_true", "y_pred")
        luxury_row = by_band[by_band["price_band"] == "Luxury"].iloc[0].to_dict()

        comparison[f"k={k}"] = {"global_mae": global_metrics["mae"], "luxury_mae": luxury_row["mae"]}
        ledger_rows.append({
            "experiment_id": f"fase10_k{k}",
            "dataset_version": "features_contextual_v1",
            "feature_set_version": f"spatial_k{k}",
            "hypothesis_id": "fase10_h1_k_neighbors",
            "model": "xgboost",
            "hyperparameters": json.dumps(winner_params["params"]),
            "split_version": "split_v1",
            "metrics": json.dumps(
                {"global_mae": global_metrics["mae"], "luxury_mae": luxury_row["mae"]}
            ),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "git_commit": "",
            "status": "candidate",
        })

        if k == BASELINE_K:
            baseline_global_mae = global_metrics["mae"]
            baseline_luxury_mae = luxury_row["mae"]

    assert baseline_global_mae is not None, f"BASELINE_K={BASELINE_K} precisa estar em CANDIDATE_K_VALUES"

    # --- decisão baseada no critério de aceite definido antes do experimento ---
    decision = {}
    for k in CANDIDATE_K_VALUES:
        if k == BASELINE_K:
            continue
        g = comparison[f"k={k}"]["global_mae"]
        l = comparison[f"k={k}"]["luxury_mae"]
        luxury_improvement = (baseline_luxury_mae - l) / baseline_luxury_mae
        global_degradation = (g - baseline_global_mae) / baseline_global_mae
        meets_criteria = luxury_improvement >= 0.05 and global_degradation <= 0.01
        decision[f"k={k}"] = {
            "luxury_improvement_pct": luxury_improvement, "global_degradation_pct": global_degradation,
            "meets_acceptance_criteria": bool(meets_criteria),
        }

    any_accepted = any(d["meets_acceptance_criteria"] for d in decision.values())
    out = {
        "hypothesis": "k diferente do baseline (30) reduz erro em Luxury sem piorar erro global",
        "acceptance_criteria": "Luxury MAE cai >=5% E MAE global nao piora mais que 1%",
        "baseline_k30": {"global_mae": baseline_global_mae, "luxury_mae": baseline_luxury_mae},
        "comparison": comparison,
        "decision_per_k": decision,
        "final_decision": "PROMOTE" if any_accepted else "REJECT_KEEP_K30",
        "stopping_rule_applied": "fase10 encerrada apos 1 iteracao (roadmap secao 10) - "
                                  "nao itera indefinidamente perseguindo o segmento de luxo",
    }

    Path(ledger_path).parent.mkdir(exist_ok=True, parents=True)
    ledger_df = pd.read_csv(ledger_path)
    ledger_df = pd.concat([ledger_df, pd.DataFrame(ledger_rows)], ignore_index=True)
    ledger_df.to_csv(ledger_path, index=False)

    return out


def main() -> None:
    out = run_k_sweep_hypothesis_pipeline()
    Path("reports/fase10_experiment.json").write_text(
        json.dumps(out, indent=2, default=str), encoding="utf-8"
    )
    print(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    main()
