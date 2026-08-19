# Fase 09 — feature_validation (Notebook: notebooks/09_feature_validation.ipynb)

## Objetivo
As features novas (fases 07/08) têm leakage, são redundantes ou são estáveis o suficiente para entrar
no modelo final?

## Metodologia / por quê
Ablation formal decide o que sobrevive, não correlação bruta isolada — a fase 07 já mostrou que
correlação bruta é só triagem, insuficiente para decisão. O notebook encadeia 4 funções chamáveis em
processo, nesta ordem exata:
1. `run_register_features_pipeline` (`scripts/register_features.py`) — registra hipóteses no
   `feature_registry.csv` antes da validação em si. É um script historicamente **órfão**: nenhum
   notebook o chamava antes desta refatoração. Seu próprio docstring documenta uma exceção honesta —
   `property_age` foi implementado na fase 03 (necessário para o cluster físico), antes desta rotina de
   registro existir, e está registrado aqui retroativamente, sem esconder a ordem real de execução.
2. `run_feature_validation_pipeline` (`scripts/validate_features.py`) — leakage estrutural (spatial
   index fit-size) + gap de correlação TRAIN→TEST por feature contextual, e ablation via `GroupKFold(3)`
   por zipcode (só em TRAIN), em MAE (dólar), comparando baseline, +raw_derived, +contextual e
   combinações.
3. `run_promote_features_pipeline` (`scripts/promote_features.py`) — aplica a decisão da ablation ao
   registry (`active`/`rejected`).
4. `run_finalize_feature_set_pipeline` (`scripts/finalize_feature_set.py`) — grava o conjunto oficial de
   features `active` em `feature_metadata.json`, consumido pela fase 10.

## Decisões-chave
- Ordem fixa register → validate → promote → finalize; cada passo só roda depois do anterior, dentro do
  mesmo processo do notebook (sem `subprocess`).
- `data/processing/feature_registry.csv` é o artefato governante — toda promoção/rejeição de feature
  passa por ele, nunca é decidida só no notebook ou só no relatório de ablation.
- `register_features.py` é chamado pela primeira vez a partir de um notebook nesta refatoração; a
  exceção histórica do `property_age` (registrado retroativamente, fase 03) é preservada no relato, não
  suavizada.
- Ablation usa um XGBoost fixo (não tunado — tuning é decisão da fase 10), sempre via `GroupKFold` por
  zipcode restrito a TRAIN, nunca partição única.
- `raw_derived` (6 features da fase 07) rejeitado: ganho isolado marginal e piora quando combinado com
  contextual — complexidade não pagou pela melhora.
- Refatoração (P5): as 4 funções são chamadas em processo; a execução é idempotente — como o pipeline já
  tinha rodado antes, o registry já chegou com `active`/`rejected` decididos, e esta execução reconfirma
  o mesmo estado (visível na tabela antes x depois no notebook).

## Entradas
- `data/trusted/features_contextual.parquet`
- `artifacts/spatial_index.pkl`
- `configs/model.yaml`
- `data/processing/feature_registry.csv`

## Saídas
- `data/processing/feature_registry.csv` (atualizado)
- `reports/ablation_fase09.json`
- `data/trusted/feature_metadata.json`
- `reports/figures/09_ablation.png`

## Resultado
**Registro:** 0 novas linhas nesta execução (26 linhas totais) — registry já existia de execução
anterior; `property_age` confirmado com `phase=3`, preservando a exceção histórica.
**Leakage:** `spatial_index_fit_size_matches_train=True`; nenhuma das 3 features contextuais dispara
`suspicious_leakage` (todas `False`) — `comps_knn_price` tem o maior gap (-0,232), na direção esperada
de generalização geográfica, não vazamento.
**Ablation (MAE $, GroupKFold(3), TRAIN):** baseline 107.515±14.836; +raw_derived 107.672±14.051 (~igual,
dentro do desvio); +contextual 79.520±7.063 (-27,0%); +raw_derived+contextual 80.463±8.066 (pior que só
contextual); **+contextual+`property_age` 77.198±6.956 (-28,2% vs. baseline, melhor conjunto)**.
**Promoção:** `active` — `dist_to_seattle_center`, `comps_knn_price`, `local_grade_percentile`,
`property_age`. `rejected` — `was_renovated`, `years_since_renovation`, `has_basement`,
`basement_ratio`, `grade_condition_interaction`, `log_sqft_lot`.
**Finalização:** 20 features oficiais gravadas em `data/trusted/feature_metadata.json`. Números
idênticos aos da execução anterior à reestruturação.

## Ver também
- Config: `configs/model.yaml`
- Relatório de fase: `reports/phase_reports/09_feature_validation.md`
