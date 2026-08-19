"""Fase 09 — atualiza status no feature_registry.csv com base na decisão de ablation (P2, P4).

Decisão (ver reports/ablation_fase09.json e reports/phase_reports/09_feature_validation.md), recomputada
sobre o `train` aumentado da fase 05:
- contextual (dist_to_seattle_center, comps_knn_price, local_grade_percentile) + property_age: ACTIVE
  (MAE cai de 95.276 -> 75.741, melhora em TODAS as bandas de preço, não só agregado — P4).
- raw_derived (was_renovated, years_since_renovation, has_basement, basement_ratio,
  grade_condition_interaction, log_sqft_lot): REJECTED — sozinho tem ganho marginal (95.276 -> 94.263),
  mas PIORA quando combinado com contextual (77.325 -> 77.442) — o que importa é o conjunto final, não a
  contribuição isolada. XGBoost já captura essas interações/transformações via splits nativos.
  Complexidade não pagou aluguel (P2).

Decisão nova (candidatas das fases 07/08, dirigidas pelas extensões exploratórias das fases 04/06),
revisada após validação com mais folds (critério do usuário: só segue se melhoria > 5%):
- GroupKFold(3) inicial: best_known 77.198 -> best_known+as 6 novas 76.442 (-0,97%). Diferença pequena
  frente ao desvio-padrão agregado (~7.000), mas pareada por fold (mesmos splits, GroupKFold é
  determinístico) a direção é consistente nos 3 folds (t=5.27, p=0.034) — não é ruído puro, mas é
  pequena.
- Revalidado com mais folds pra estimativa mais estável (49 zipcodes em train permitem):
  GroupKFold(7) = +1,50% (p=0.10); GroupKFold(10) = +1,44% (p=0.019). Efeito real, mas o TAMANHO fica
  em ~1,4-1,5% em toda granularidade de fold testada — não passa do corte de 5% definido pelo usuário
  como critério de decisão pra esta rodada.
- REJECTED as 6 candidatas novas por não atingirem o corte de melhoria mínima (5%), apesar do efeito
  ser estatisticamente real e consistente em direção. Decisão de threshold de negócio, não de ausência
  de sinal — registry mantém a hipótese e os números pra referência futura.
"""
from pathlib import Path

import pandas as pd

REGISTRY_PATH = Path("data/processing/feature_registry.csv")

ACTIVE = {"dist_to_seattle_center", "comps_knn_price", "local_grade_percentile", "property_age"}
REJECTED = {
    "was_renovated", "years_since_renovation", "has_basement", "basement_ratio",
    "grade_condition_interaction", "log_sqft_lot",
    "bathrooms_per_bedroom", "sqft_living_to_lot_ratio", "property_cluster_distance",
    "dist_to_nearest_train_zip", "comps_knn_neighbor_distance", "local_price_dispersion",
}


def run_promote_features_pipeline(
    registry_path: str = "data/processing/feature_registry.csv",
) -> dict:
    """Refatorada em função chamável (P5) — reusada tanto por `main()` (CLI) quanto por notebook
    (`09_feature_validation.ipynb`), aplicando a decisão de ablation ao registry."""
    registry_path_p = Path(registry_path)
    registry = pd.read_csv(registry_path_p)
    registry.loc[registry["feature"].isin(ACTIVE), "status"] = "active"
    registry.loc[registry["feature"].isin(REJECTED), "status"] = "rejected"
    registry.to_csv(registry_path_p, index=False)
    return {
        "registry_path": registry_path,
        "active_features": sorted(ACTIVE),
        "rejected_features": sorted(REJECTED),
        "summary_table": registry[["feature", "status"]].to_string(index=False),
    }


def main() -> None:
    out = run_promote_features_pipeline()
    print(out["summary_table"])


if __name__ == "__main__":
    main()
