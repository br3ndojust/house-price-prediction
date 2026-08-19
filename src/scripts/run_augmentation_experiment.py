"""Fase 05 — augmentation_experiment: baseline (sem augmentation, treinado completo) vs. Técnica A
(SMOGN) vs. Técnica B (perturbação controlada). Só `train` é aumentado (P1); `test` é sempre o real,
íntegro, usado como cheque de zipcode não visto (relatado, não decide sozinho); `val` não entra aqui.

CRITÉRIO DE ACEITE (definido antes do experimento, P4): decisão principal é por MAE médio em
`GroupKFold(3)/zipcode` DENTRO do train (aumentado ou não) — nunca por `test` isolado. Adota a técnica
vencedora só se reduzir o MAE de CV em >=2% sobre o baseline (evita adotar ruído). Reporta também o
MAE em `test` global e no subconjunto "atípico" de `test` (distância a train no espaço de features acima
da mediana, identificado na fase 04) como evidência complementar, não decisiva.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.augmentation.perturbation import controlled_perturbation_augment
from src.augmentation.smogn import smogn_augment
from src.evaluation.error_matrix import compute_metrics
from src.evaluation.geographic_coverage import feature_space_coverage
from src.validation.ablation import run_ablation_cv
from xgboost import XGBRegressor
from src.validation.ablation import ABLATION_MODEL_PARAMS

FEATURE_COLS = [
    "bedrooms", "bathrooms", "sqft_living", "sqft_lot", "floors", "waterfront", "view",
    "condition", "grade", "sqft_above", "sqft_basement", "sqft_living15", "sqft_lot15",
    "medn_hshld_incm_amt", "hous_val_amt", "per_bchlr",
]
TARGET = "price_log"
GROUP_COL = "zipcode"


def evaluate_on_test(train: pd.DataFrame, test: pd.DataFrame, hard_test_idx: np.ndarray) -> dict:
    model = XGBRegressor(**ABLATION_MODEL_PARAMS, n_jobs=-1)
    model.fit(train[FEATURE_COLS], train[TARGET])
    pred = np.expm1(model.predict(test[FEATURE_COLS]))
    true = np.expm1(test[TARGET])
    overall = compute_metrics(true.to_numpy(), pred)
    hard = compute_metrics(true.to_numpy()[hard_test_idx], pred[hard_test_idx])
    return {"overall": overall, "hard_subset": hard}


def run_augmentation_experiment_pipeline(
    model_cfg_path: str = "configs/model.yaml",
    house_clean_parquet: str = "data/processed/house_clean.parquet",
    split_assignment_parquet: str = "data/processed/split_assignment.parquet",
    augmentation_cfg_path: str = "configs/augmentation.yaml",
    train_for_pipeline_parquet: str = "data/trusted/train_for_pipeline.parquet",
) -> dict:
    """Fase 05, refatorada em função chamável (P5) — reusada tanto por `main()` (CLI) quanto por
    notebook."""
    model_cfg = yaml.safe_load(Path(model_cfg_path).read_text(encoding="utf-8"))["cv"]

    house = pd.read_parquet(house_clean_parquet)
    split = pd.read_parquet(split_assignment_parquet)
    df = house.merge(split[["id", "date", "split"]], on=["id", "date"], how="left")
    train = df[df["split"] == "train"].reset_index(drop=True)
    test = df[df["split"] == "test"].reset_index(drop=True)

    # subconjunto "atipico" de test (achado da fase 04): distancia a train no espaco de features
    # acima da mediana -> onde augmentation deveria ajudar mais, se a hipotese estiver certa
    coverage = feature_space_coverage(train, test, FEATURE_COLS)
    hard_test_idx = np.flatnonzero(
        coverage["nearest_train_neighbor_distance"] >= coverage["nearest_train_neighbor_distance"].median()
    )

    variants = {
        "baseline_no_augmentation": train,
        "technique_a_smogn": smogn_augment(
            train, FEATURE_COLS, TARGET, k=5, tail_boost=1.0, random_state=42
        ),
        "technique_b_controlled_perturbation": controlled_perturbation_augment(
            train, augment_fraction=0.5, random_state=42
        ),
    }

    results = {}
    for name, variant_train in variants.items():
        cv = run_ablation_cv(
            variant_train, FEATURE_COLS, TARGET, GROUP_COL,
            n_splits=model_cfg["n_splits"], random_state=model_cfg["random_state"],
        )
        test_eval = evaluate_on_test(variant_train, test, hard_test_idx)
        results[name] = {
            "n_train_rows": len(variant_train),
            "cv_mae_mean": cv["mae_mean"], "cv_mae_std": cv["mae_std"],
            "test_mae_overall": test_eval["overall"]["mae"],
            "test_mae_hard_subset": test_eval["hard_subset"]["mae"],
        }

    baseline_cv_mae = results["baseline_no_augmentation"]["cv_mae_mean"]
    decision = {}
    for name in ["technique_a_smogn", "technique_b_controlled_perturbation"]:
        improvement = (baseline_cv_mae - results[name]["cv_mae_mean"]) / baseline_cv_mae
        decision[name] = {
            "cv_mae_improvement_pct": improvement,
            "meets_acceptance_criteria": bool(improvement >= 0.02),
        }

    best_name = min(results, key=lambda n: results[n]["cv_mae_mean"])
    cv_recommended = best_name if (
        best_name != "baseline_no_augmentation"
        and decision.get(best_name, {}).get("meets_acceptance_criteria", False)
    ) else "baseline_no_augmentation"

    # A variante MATERIALIZADA segue configs/augmentation.yaml, nao o vencedor mecanico do criterio
    # de CV (ver fase 12: o criterio de CV recomendou Tecnica B e isso piorou a generalizacao real -
    # a decisao final e humana/versionada, nao um efeito colateral de rodar este script de novo).
    aug_cfg = yaml.safe_load(Path(augmentation_cfg_path).read_text(encoding="utf-8"))
    adopted_key = {
        "none": "baseline_no_augmentation",
        "technique_a_smogn": "technique_a_smogn",
        "technique_b_controlled_perturbation": "technique_b_controlled_perturbation",
    }[aug_cfg["adopted"]]

    out = {
        "acceptance_criteria": "criterio de CV: reduz MAE de GroupKFold(3) dentro do train em >=2% "
                                "sobre baseline sem augmentation",
        "results": results,
        "decision_per_technique": decision,
        "cv_recommended": cv_recommended,
        "adopted": adopted_key,
        "adopted_source": "configs/augmentation.yaml (decisao humana/versionada, ver fase 12 no "
                           "AUDIT_LOG.md - pode divergir do cv_recommended de proposito)",
    }
    chosen_train = variants[adopted_key]
    Path(train_for_pipeline_parquet).parent.mkdir(parents=True, exist_ok=True)
    chosen_train.to_parquet(train_for_pipeline_parquet, index=False)

    return out


def main() -> None:
    out = run_augmentation_experiment_pipeline()
    Path("reports/augmentation_experiment.json").write_text(json.dumps(out, indent=2, default=str))
    print(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    main()
