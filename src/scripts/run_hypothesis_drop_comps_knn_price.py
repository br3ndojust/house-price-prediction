"""Fase 15 — Hipótese: remover `comps_knn_price` melhora o modelo?

Motivação: `comps_knn_price` domina 40,5% do gain total (gain-based, XGBoost) — mais que o dobro da
2ª feature (`sqft_living`, 19,1%) e quase tanto quanto as outras 19 features somadas (~59,5%). Essa
discrepância motiva a pergunta: será que o modelo generaliza melhor sem ela (menos dependência de uma
única feature target-derived, fase 06)?

Critério pré-registrado (P4 — definido ANTES de rodar, nunca ajustado depois do resultado):
- **TRAIN, GroupKFold(3)** (mesmo método fases 09/10, hiperparâmetros oficiais do modelo final —
  XGBoost `n_estimators=400, max_depth=3, learning_rate=0.05`): MAE do candidato (19 features) não
  pode piorar em relação ao baseline (20 features).
- **TEST** (modelo fit só em TRAIN, aplicado em TEST nunca visto — mesmo método fase 11/12): métrica
  DECISIVA para generalização (mesmo precedente da fase 12 iteração 2, que usou TEST deliberadamente
  quando a pergunta é sobre generalização, não sobre CV em TRAIN). MAE precisa **cair ≥5%** para adotar.
- **Adoção exige as duas condições** — TRAIN não piora E TEST melhora ≥5%. Resultado misto (ex: TEST
  melhora mas TRAIN piora) é reportado como inconclusivo, não adotado sem uma nova rodada de análise.
- Quebra por `price_band` (P3) — nunca decide só pelo MAE global; uma melhora agregada que esconde
  piora em `Luxury` é motivo de rejeição, mesmo que o número global passe no critério (mesma disciplina
  do critério de substituição de `docs/08_continuous_learning.md`, seção 4).

Não é uma fase que decide sozinha — se ADOTADA, a hipótese só vira modelo oficial depois de refazer
fase 10 (seleção) + fase 13 (refit final + checagem em `val`) + fase 14 (contrato) com o conjunto de
19 features, feito por `src/scripts/promote_hypothesis_drop_comps_knn_price.py`.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.evaluation.error_matrix import compute_metrics, error_matrix_by
from src.models.train import build_model, cross_validate_candidate

TARGET = "price_log"
GROUP_COL = "zipcode"
DROP_FEATURE = "comps_knn_price"
TEST_DECISIVE_IMPROVEMENT = 0.05  # 5%, pré-registrado (pedido do usuário)


def run_drop_comps_knn_price_hypothesis_pipeline(
    features_parquet: str = "data/trusted/features_contextual.parquet",
    candidate_artifact_path: str = "artifacts/model_candidate.pkl",
    output_report_path: str = "reports/hypothesis_drop_comps_knn_price.json",
) -> dict:
    """Fase 15, refatorada em função chamável (P5) — reusada tanto por `main()` (CLI) quanto pelo
    notebook `12_hypothesis_drop_comps_knn_price_iter3.ipynb`.

    Critério pré-registrado (P4 — definido ANTES de rodar, nunca ajustado depois do resultado): ver
    docstring do módulo para o critério completo."""
    candidate_bundle = joblib.load(candidate_artifact_path)
    model_name = candidate_bundle["model_name"]
    params = candidate_bundle["params"]
    baseline_features = candidate_bundle["feature_cols"]
    assert DROP_FEATURE in baseline_features, f"{DROP_FEATURE} não está no conjunto oficial atual"
    candidate_features = [f for f in baseline_features if f != DROP_FEATURE]

    df = pd.read_parquet(features_parquet)
    train = df[df["split"] == "train"].reset_index(drop=True)
    test = df[df["split"] == "test"].reset_index(drop=True)

    feature_sets = {"baseline_20_features": baseline_features, "sem_comps_knn_price_19_features": candidate_features}

    # ---- TRAIN, GroupKFold(3) — mesmo método fases 09/10 --------------------------------------
    cv_results = {
        name: cross_validate_candidate(train, cols, TARGET, GROUP_COL, model_name, params, n_splits=3, random_state=42)
        for name, cols in feature_sets.items()
    }

    # ---- TEST — fit só em TRAIN, aplica em TEST nunca visto (mesmo método fase 11) -------------
    test_results = {}
    for name, cols in feature_sets.items():
        model = build_model(model_name, params)
        model.fit(train[cols], train[TARGET])
        y_true = np.expm1(test[TARGET].to_numpy())
        y_pred = np.expm1(model.predict(test[cols]))
        test_df = test.copy()
        test_df["y_true"], test_df["y_pred"] = y_true, y_pred
        test_results[name] = {
            "global": compute_metrics(y_true, y_pred),
            "by_price_band": error_matrix_by(test_df, "price_band", "y_true", "y_pred").to_dict(orient="records"),
        }

    # ---- comparação + decisão --------------------------------------------------------------
    base_cv_mae = cv_results["baseline_20_features"]["mae_mean"]
    cand_cv_mae = cv_results["sem_comps_knn_price_19_features"]["mae_mean"]
    cv_change_pct = (cand_cv_mae - base_cv_mae) / base_cv_mae

    base_test_mae = test_results["baseline_20_features"]["global"]["mae"]
    cand_test_mae = test_results["sem_comps_knn_price_19_features"]["global"]["mae"]
    test_change_pct = (cand_test_mae - base_test_mae) / base_test_mae  # negativo = melhora (MAE caiu)

    train_did_not_worsen = cv_change_pct <= 0
    test_improved_5pct = test_change_pct <= -TEST_DECISIVE_IMPROVEMENT

    by_band_base = {r["price_band"]: r["mae"] for r in test_results["baseline_20_features"]["by_price_band"]}
    by_band_cand = {r["price_band"]: r["mae"] for r in test_results["sem_comps_knn_price_19_features"]["by_price_band"]}
    band_deltas = {
        band: {
            "mae_baseline": by_band_base[band], "mae_candidato": by_band_cand.get(band),
            "change_pct": (by_band_cand[band] - by_band_base[band]) / by_band_base[band] if band in by_band_cand else None,
        }
        for band in by_band_base
    }
    any_band_worsened_a_lot = any(
        d["change_pct"] is not None and d["change_pct"] > 0.10  # mesmo espírito da margem tolerável do docs/08 (~2%), aqui mais folgado por ser 1 feature só
        for d in band_deltas.values()
    )

    rejection_reasons = []
    if not train_did_not_worsen:
        rejection_reasons.append(
            f"TRAIN GroupKFold(3) MAE piorou {cv_change_pct:+.2%} (critério: não pode piorar)"
        )
    if not test_improved_5pct:
        near_miss = " (near miss — quase bateu o critério, mesmo padrão da fase 12 iter.1)" if test_change_pct < 0 else ""
        rejection_reasons.append(
            f"TEST MAE mudou {test_change_pct:+.2%}, não atingiu a queda decisiva de "
            f"-{TEST_DECISIVE_IMPROVEMENT:.0%}{near_miss}"
        )
    if any_band_worsened_a_lot:
        worsened = {b: d["change_pct"] for b, d in band_deltas.items() if d["change_pct"] and d["change_pct"] > 0.10}
        rejection_reasons.append(f"banda(s) pioraram >10%: {worsened}")

    decision = "ADOPT_DROP_COMPS_KNN_PRICE" if not rejection_reasons else "REJECT_KEEP_COMPS_KNN_PRICE"

    out = {
        "hypothesis": "Remover comps_knn_price (feature dominante, 40.5% do gain) melhora o modelo.",
        "pre_registered_criterion": {
            "train_groupkfold3": "MAE do candidato não pode piorar em relação ao baseline",
            "test_holdout": f"MAE do candidato precisa cair >= {TEST_DECISIVE_IMPROVEMENT:.0%} (métrica decisiva)",
            "by_price_band": "nenhuma banda pode piorar > 10% (P3 — nunca só o número global)",
        },
        "model": {"algorithm": model_name, "params": params},
        "feature_counts": {"baseline": len(baseline_features), "candidato": len(candidate_features)},
        "train_groupkfold3": {
            "baseline_mae_mean": base_cv_mae, "candidato_mae_mean": cand_cv_mae,
            "change_pct": cv_change_pct, "train_did_not_worsen": train_did_not_worsen,
            "cv_results_full": cv_results,
        },
        "test_holdout": {
            "baseline_mae": base_test_mae, "candidato_mae": cand_test_mae,
            "change_pct": test_change_pct, "test_improved_5pct": test_improved_5pct,
            "by_price_band": band_deltas, "any_band_worsened_a_lot": any_band_worsened_a_lot,
            "full_results": test_results,
        },
        "decision": decision,
        "rejection_reasons": rejection_reasons,
    }
    Path(output_report_path).write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    return out


def main() -> None:
    out = run_drop_comps_knn_price_hypothesis_pipeline()
    cv_change_pct = out["train_groupkfold3"]["change_pct"]
    base_cv_mae = out["train_groupkfold3"]["baseline_mae_mean"]
    cand_cv_mae = out["train_groupkfold3"]["candidato_mae_mean"]
    base_test_mae = out["test_holdout"]["baseline_mae"]
    cand_test_mae = out["test_holdout"]["candidato_mae"]
    test_change_pct = out["test_holdout"]["change_pct"]
    band_deltas = out["test_holdout"]["by_price_band"]
    decision = out["decision"]

    print(json.dumps({k: v for k, v in out.items() if k not in ("train_groupkfold3", "test_holdout")}, indent=2, default=str))
    print(f"\nTRAIN CV: baseline={base_cv_mae:,.0f}  candidato={cand_cv_mae:,.0f}  change={cv_change_pct:+.2%}")
    print(f"TEST:     baseline={base_test_mae:,.0f}  candidato={cand_test_mae:,.0f}  change={test_change_pct:+.2%}")
    print(f"Por banda: {json.dumps(band_deltas, indent=2, default=str)}")
    print(f"\nDECISÃO: {decision}")


if __name__ == "__main__":
    main()
