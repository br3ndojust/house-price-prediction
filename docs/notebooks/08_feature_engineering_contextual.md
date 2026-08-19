# Fase 08 — feature_engineering_contextual (Notebook: notebooks/08_feature_engineering_contextual.ipynb)

## Objetivo
Que mecanismo de localização/comparável/posição relativa ainda falta representar?

## Metodologia / por quê
Hipótese: um índice espacial de vizinhos (KNN, `k=30`, fit **só** em `train`) captura sinal de mercado
local mais fino que a média por zipcode — a fase 04 já havia mostrado gradiente espacial forte centrado
na área urbana e `hous_val_amt` (demografia do zipcode) quase tão preditivo quanto atributos físicos.
`run_contextual_features_pipeline` (`scripts/generate_features.py`) lê o parquet de RAW/DERIVED do disco
(saída da fase 07) — em vez de receber o `df` em memória — para poder rodar de forma independente da
fase 07 caso o parquet já exista. Sobre esse `df`: distância euclidiana a um ponto fixo (centro de
Seattle), depois `fit_spatial_index` só nas linhas `split == "train"`, e as features derivadas do índice
(`comps_knn_price`, `local_grade_percentile`) são aplicadas a todo o dataset via `apply_*`, nunca
refitadas em test/val.

## Decisões-chave
- Índice espacial (`fit_spatial_index`) ajustado exclusivamente em `train` — condição estrutural para
  não vazar `test`/`val` no cálculo de vizinhos; checado formalmente na fase 09
  (`check_spatial_index_fit_size`).
- `k=30`, `bed_tolerance=1`, `bath_tolerance=0.5`, `grade_tolerance=1` vêm de `configs/comparable.yaml`,
  não hardcoded no script.
- `comps_knn_price` é `target_derived=True` no registry — usa `price_log` de vizinhos em TRAIN, por isso
  é a feature mais sensível a leakage estrutural entre as 3 contextuais.
- Pós fase 12: índice refit sobre `train` sem augmentation (15.100 linhas).
- Refatoração (P5): função chamada em processo; artefatos (`features_contextual.parquet`,
  `spatial_index.pkl`) continuam sendo gravados pela própria função, não pelo notebook.

## Entradas
- `data/trusted/features_raw_derived.parquet`
- `configs/comparable.yaml`

## Saídas
- `data/trusted/features_contextual.parquet`
- `artifacts/spatial_index.pkl`
- `reports/figures/08_contextual.png`

## Resultado
`run_contextual_features_pipeline()` retornou `n_rows=21612`, `n_columns=62`, 3 `new_features`. TRAIN
com 15.100 linhas, TEST com 2.644. Correlação com `price_log`: `comps_knn_price` r_train=0,833 ->
r_test=0,601 (gap -0,232, generalização geográfica esperada, não leakage); `local_grade_percentile`
r_train=0,436 -> r_test=0,504 (gap +0,069); `dist_to_seattle_center` r_train=-0,188 -> r_test=-0,190
(gap -0,002, estável). Números idênticos aos da execução anterior à reestruturação.

## Ver também
- Config: `configs/comparable.yaml`
- Relatório de fase: `reports/phase_reports/08_feature_engineering_contextual.md`
