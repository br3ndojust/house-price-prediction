"""Registra hipóteses no feature_registry.csv ANTES da implementação (P2, sem exceção).

Idempotente: recarrega o CSV, adiciona só features cujo nome ainda não existe.
"""
from pathlib import Path

import pandas as pd

REGISTRY_PATH = Path("data/processing/feature_registry.csv")

ROWS = [
    # retroativo: property_age foi implementado na fase 03 (necessário para o cluster físico),
    # antes desta rotina de registro existir. Registrado aqui com honestidade sobre a ordem real
    # de execução (P7) — é um desvio pontual do fluxo "registry antes de tudo", documentado no
    # audit log da fase 05, não escondido.
    dict(feature="property_age", layer="DERIVED", family="AGE", source="date,yr_built",
         target_derived=False, spatial=False,
         hypothesis="Idade do imóvel na data da venda correlaciona com depreciação/estilo construtivo",
         phase="03", first_iteration="fase03_segmentation", status="active"),

    dict(feature="was_renovated", layer="DERIVED", family="CONDITION", source="yr_renovated",
         target_derived=False, spatial=False,
         hypothesis="Reforma recente sinaliza melhoria de condição não capturada por yr_built sozinho",
         phase="05", first_iteration="fase05_raw_derived", status="candidate"),

    dict(feature="years_since_renovation", layer="DERIVED", family="CONDITION",
         source="yr_renovated,date", target_derived=False, spatial=False,
         hypothesis="Efeito de reforma decai com o tempo — quanto mais recente, maior o prêmio de preço",
         phase="05", first_iteration="fase05_raw_derived", status="candidate"),

    dict(feature="has_basement", layer="DERIVED", family="SIZE", source="sqft_basement",
         target_derived=False, spatial=False,
         hypothesis="Presença de porão é um mecanismo de layout distinto de área total",
         phase="05", first_iteration="fase05_raw_derived", status="candidate"),

    dict(feature="basement_ratio", layer="DERIVED", family="SIZE",
         source="sqft_basement,sqft_living", target_derived=False, spatial=False,
         hypothesis="Proporção de área no porão captura composição do espaço, não só volume total",
         phase="05", first_iteration="fase05_raw_derived", status="candidate"),

    dict(feature="grade_condition_interaction", layer="DERIVED", family="QUALITY",
         source="grade,condition", target_derived=False, spatial=False,
         hypothesis=("Fase 04 mostrou condition fraco isolado (r=0.057) e leve correlação negativa "
                     "com grade (-0.12) — a interação pode capturar o efeito real de condição em "
                     "imóveis de grade equivalente, que condition sozinho esconde"),
         phase="05", first_iteration="fase05_raw_derived", status="candidate"),

    dict(feature="log_sqft_lot", layer="DERIVED", family="SIZE", source="sqft_lot",
         target_derived=False, spatial=False,
         hypothesis="sqft_lot tem skew bruto 10.7 (checado em train); log1p reduz para 0.96, "
                     "estabilizando a relação com price_log como já validado para o próprio alvo",
         phase="05", first_iteration="fase05_raw_derived", status="candidate"),

    dict(feature="dist_to_seattle_center", layer="CONTEXTUAL", family="ACCESSIBILITY",
         source="lat,long", target_derived=False, spatial=True,
         hypothesis=("Fase 04 mostrou gradiente espacial forte centrado na área urbana — distância "
                     "euclidiana a um ponto fixo (centro de Seattle) captura acessibilidade sem "
                     "memorizar identidade administrativa de zipcode"),
         phase="06", first_iteration="fase06_contextual", status="candidate"),

    dict(feature="comps_knn_price", layer="CONTEXTUAL", family="COMPARABLE_MARKET",
         source="lat,long,price_log", target_derived=True, spatial=True,
         hypothesis=("Fase 04 mostrou hous_val_amt (demografia do zipcode) quase tão preditivo quanto "
                     "atributos físicos — preço médio dos k=30 vizinhos espaciais mais próximos em "
                     "TRAIN deve capturar sinal de mercado local mais fino que a média por zipcode"),
         phase="06", first_iteration="fase06_contextual", status="candidate"),

    dict(feature="local_grade_percentile", layer="CONTEXTUAL", family="RELATIVE_POSITION",
         source="lat,long,grade", target_derived=False, spatial=True,
         hypothesis=("Grade absoluto não diz se um imóvel é bem ou mal avaliado frente aos vizinhos "
                     "imediatos — percentil de grade entre os k=30 vizinhos espaciais em TRAIN testa "
                     "se posição relativa importa além do valor absoluto"),
         phase="06", first_iteration="fase06_contextual", status="candidate"),

    # candidatas novas (fases 07/08) — dirigidas pelas extensões exploratórias das fases 04/06,
    # nunca hardcoded como suposição: cada hipótese cita o número real que a motivou.
    dict(feature="bathrooms_per_bedroom", layer="DERIVED", family="SIZE",
         source="bathrooms,bedrooms", target_derived=False, spatial=False,
         hypothesis=("Extensão exploratória da fase 04 achou bathrooms como a feature crua cujo gap de "
                     "cobertura train->test mais correlaciona com erro real (r=0.25, maior entre as 16 "
                     "BASELINE_FEATURES) — razão relativa tende a generalizar melhor que contagem "
                     "bruta sob o holdout geográfico completo do split (fase 03)"),
         phase="07", first_iteration="fase07_raw_derived", status="candidate"),

    dict(feature="sqft_living_to_lot_ratio", layer="DERIVED", family="SIZE",
         source="sqft_living,sqft_lot", target_derived=False, spatial=False,
         hypothesis=("Extensão da fase 04 achou sqft_lot/sqft_lot15 com maior gap de distribuição "
                     "relativa train->test (20-21%) depois de floors; log_sqft_lot (fase 05) foi "
                     "rejeitado na ablation olhando só MAE médio — esta razão ataca a mesma escassez "
                     "por outro ângulo (tamanho relativo ao lote, não escala absoluta)"),
         phase="07", first_iteration="fase07_raw_derived", status="candidate"),

    dict(feature="property_cluster_distance", layer="DERIVED", family="PHYSICAL_TYPICALITY",
         source=("property_cluster,sqft_living,sqft_lot,bedrooms,bathrooms,floors,grade,condition,"
                  "view,waterfront,property_age"),
         target_derived=False, spatial=False,
         hypothesis=("Extensão exploratória da fase 06 achou que o property_cluster raro (perfil "
                     "físico de luxo, fitado só em train) tem erro médio 4.5x pior em test, mas sua "
                     "frequência por partição varia por acaso de qual zipcode caiu em qual grupo "
                     "(split fase 03) — distância contínua ao centróide (reusa KMeans/scaler já "
                     "fitados em train) preserva o sinal de erro sem herdar essa instabilidade "
                     "categórica"),
         phase="07", first_iteration="fase07_raw_derived", status="candidate"),

    dict(feature="dist_to_nearest_train_zip", layer="CONTEXTUAL", family="ACCESSIBILITY",
         source="lat,long (fit de centróides de zipcode em train)",
         target_derived=False, spatial=True,
         hypothesis=("Promove pra feature real o diagnóstico da fase 04 (nearest_train_zip_distance, "
                     "só usado como leitura), calculado por imóvel em vez de por zipcode agregado — "
                     "dist_to_seattle_center (ponto fixo) tem correlação fraca isolada (-0.188); esta "
                     "é dinâmica e reflete cobertura real do dado de train"),
         phase="08", first_iteration="fase08_contextual", status="candidate"),

    dict(feature="comps_knn_neighbor_distance", layer="CONTEXTUAL", family="COMPARABLE_MARKET",
         source="lat,long (via SpatialIndex fitado em train)", target_derived=False, spatial=True,
         hypothesis=("A própria fase 08 achou comps_knn_price com gap forte train->test (r=0.833 -> "
                     "0.601, -0.232) por causa do holdout geográfico completo (fase 03/04) — expor a "
                     "distância média aos k=30 vizinhos usados no comp deixa o modelo aprender a "
                     "descontar comps_knn_price quando o vizinho mais próximo está longe"),
         phase="08", first_iteration="fase08_contextual", status="candidate"),

    dict(feature="local_price_dispersion", layer="CONTEXTUAL", family="COMPARABLE_MARKET",
         source="lat,long,price_log (via SpatialIndex fitado em train)",
         target_derived=True, spatial=True,
         hypothesis=("Desvio-padrão do price_log entre os mesmos k=30 vizinhos espaciais de "
                     "comps_knn_price (reusa o índice, sem novo fit) — heterogeneidade do mercado "
                     "local como sinal de incerteza, complementa comps_knn_price/local_grade_percentile"),
         phase="08", first_iteration="fase08_contextual", status="candidate"),
]


def run_register_features_pipeline(
    registry_path: str = "data/processing/feature_registry.csv",
) -> dict:
    """Refatorada em função chamável (P5) — reusada tanto por `main()` (CLI) quanto por notebook
    (`09_feature_validation.ipynb`, passo 1, antes da validação em si)."""
    registry_path_p = Path(registry_path)
    registry = pd.read_csv(registry_path_p)
    new_rows = [r for r in ROWS if r["feature"] not in set(registry["feature"])]
    if new_rows:
        registry = pd.concat([registry, pd.DataFrame(new_rows)], ignore_index=True)
        registry.to_csv(registry_path_p, index=False)
    return {"n_new_rows": len(new_rows), "n_total_rows": len(registry), "registry_path": registry_path}


def main() -> None:
    out = run_register_features_pipeline()
    print(f"{out['n_new_rows']} novas linhas registradas, {out['n_total_rows']} total")


if __name__ == "__main__":
    main()
