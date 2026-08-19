"""Fase 14 — promotion_contract: contrato de produção, distinto do experimental (P6).

Documento/pacote atômico, não substituição real de produção (não há produção real neste desafio) —
recomendação de promoção humana, nunca execução automática (P6). Usa `artifacts/model_final.pkl`
(refit `train`+`test`, fase 13) — não o `model_candidate.pkl` intermediário da fase 10.
"""
import json
from datetime import date
from pathlib import Path

import yaml


def run_production_contract_pipeline(
    feature_metadata_path: str = "data/trusted/feature_metadata.json",
    model_selection_report_path: str = "reports/model_selection_report.json",
    val_check_path: str = "reports/val_final_check.json",
    comparable_cfg_path: str = "configs/comparable.yaml",
    split_cfg_path: str = "configs/split.yaml",
    output_contract_path: str = "artifacts/production_contract.yaml",
) -> dict:
    """Fase 14, refatorada em função chamável (P5) — reusada tanto por `main()` (CLI) quanto por
    notebook/API que precise do contrato de produção sem passar por subprocess.

    Documento/pacote atômico, não substituição real de produção (não há produção real neste desafio) —
    recomendação de promoção humana, nunca execução automática (P6)."""
    feature_meta = json.loads(Path(feature_metadata_path).read_text(encoding="utf-8"))
    model_selection = json.loads(
        Path(model_selection_report_path).read_text(encoding="utf-8")
    )
    val_check = json.loads(Path(val_check_path).read_text(encoding="utf-8"))
    winner = model_selection["winner"]
    comparable_cfg = yaml.safe_load(
        Path(comparable_cfg_path).read_text(encoding="utf-8")
    )["comparable"]
    split_cfg = yaml.safe_load(Path(split_cfg_path).read_text(encoding="utf-8"))

    contract = {
        "contract_version": "1.0.0",
        "generated_at": date.today().isoformat(),
        "status": "RECOMMENDED_FOR_PROMOTION",
        "promotion_rule": (
            "Recomendacao gerada pelo pipeline (make evaluate). A substituicao de producao exige "
            "aprovacao humana explicita - o pipeline nunca promove sozinho (P6)."
        ),
        "model": {
            "algorithm": winner["model"],
            "hyperparameters": winner["params"],
            "artifact": "artifacts/model_final.pkl",
            "trained_on": "train+test combinados (refit final, fase 13)",
            "cv_mae_dollar": winner["mae_mean"],
            "cv_r2_log": winner["r2_log_mean"],
            "val_mae_dollar": val_check["global"]["mae"],
            "val_mape": val_check["global"]["mape"],
            "val_note": "val tocado uma unica vez, apos tudo (augmentation, features, modelo, "
                        "hipoteses) estar travado - fase 13.",
        },
        "feature_schema": {
            "model_features": feature_meta["model_features"],
            "target": feature_meta["target"],
            "n_features": len(feature_meta["model_features"]),
            "feature_registry": "data/processing/feature_registry.csv",
        },
        "preprocessing": {
            "spatial_index": {
                "artifact": "artifacts/spatial_index.pkl",
                "k_neighbors": comparable_cfg["k"],
                "fit_on": "train split only (P1)",
            },
            "demographics_merge": "data/raw/zipcode_demographics.csv (1:1 por zipcode)",
        },
        "dataset_version": {
            "raw_files": ["kc_house_data.csv", "zipcode_demographics.csv"],
            "processed_artifact": "data/processed/house_clean.parquet",
            "trusted_artifact": "data/trusted/features_contextual.parquet",
        },
        "split_version": {
            "method": "GroupShuffleSplit duplo por zipcode",
            "config": split_cfg,
            "metadata": "data/processed/split_metadata.json",
        },
        "validation_contract": {
            "data_contract": "configs/data_contract.yaml",
            "error_matrix_test": "reports/error_matrix.json",
            "val_final_check": "reports/val_final_check.json",
        },
        "known_limitations": [
            "Erro 2-4x maior em segmentos Luxury/waterfront/grade>=10 comparado ao mercado de massa "
            "(reports/error_matrix.json, reports/val_final_check.json) - qualquer promocao real "
            "precisa de SLA de erro diferenciado por segmento, nao um unico SLA global "
            "(ver docs/07_deploy_strategy.md).",
            "Composicao de zipcodes de TEST/VAL tem variancia alta (10/11 zipcodes cada, de 70 "
            "totais) - metrica agregada sensivel a qual zipcode especifico cai em cada particao "
            "(fase 11).",
            "Data augmentation (fase 05) melhorou CV dentro de train mas piorou generalizacao "
            "geografica real - decisao revertida na fase 12. Licao: metrica de proxy (CV em train) "
            "nem sempre reflete a metrica de interesse real (generalizacao); qualquer decisao "
            "futura de augmentation precisa ser validada contra zipcode nao visto, nao so CV.",
            "Sem recalibracao por early stopping em train+test alem do refit simples da fase 13 "
            "(simplificacao consciente do escopo enxuto desta execucao).",
        ],
        "dependencies": {
            "python": ">=3.11",
            "requirements_file": "requirements.txt",
        },
    }

    Path(output_contract_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_contract_path).write_text(
        yaml.dump(contract, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    return contract


def main() -> None:
    contract = run_production_contract_pipeline()
    print(yaml.dump(contract, allow_unicode=True, sort_keys=False))


if __name__ == "__main__":
    main()
