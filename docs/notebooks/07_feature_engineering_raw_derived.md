# Fase 07 — feature_engineering_raw_derived (Notebook: notebooks/07_feature_engineering_raw_derived.ipynb)

## Objetivo
Que mecanismos físicos/temporais/qualidade ainda não estão representados nas features brutas?

## Metodologia / por quê
Hipótese: parte do sinal está em interação/transformação de colunas existentes, não em variáveis
independentes novas. `run_raw_derived_features_pipeline` (`scripts/generate_features.py`) lê
`data/trusted/house_segments.parquet` (saída da fase 06, já sobre o `train` sem augmentation — decisão
da fase 12) e aplica transformações **row-wise**, sem nenhum `.fit`: reforma (`was_renovated`,
`years_since_renovation`), porão (`has_basement`, `basement_ratio`), interação grade×condition
(`grade_condition_interaction`) e `log1p` de `sqft_lot`. Por não terem estado ajustado, essas features
não têm risco de leakage estrutural — a única checagem necessária é correlação bruta como triagem
inicial; a decisão real de manter/descartar cada uma é da ablation formal (fase 09).

## Decisões-chave
- Todas as transformações são row-wise (sem `.fit` em nenhum subconjunto), então train/test/val recebem
  a mesma função determinística — não há risco de vazar estatística de um split para outro nesta fase.
- `log_sqft_lot` existe porque `sqft_lot` bruto tem skew ~10,7 (checado em `train`); `log1p` reduz para
  ~0,96, no mesmo espírito da transformação já validada para o próprio alvo (`price_log`).
- `grade_condition_interaction` testa a hipótese da fase 04 de que `condition` isolado é fraco
  (r=0,057), mas pode carregar sinal real quando combinado com `grade`.
- Todas as 6 features entram no registry (`data/processing/feature_registry.csv`) como `candidate` —
  status final só é decidido na fase 09.
- Refatoração (P5): a função é chamada diretamente em processo pelo notebook (`import` + chamada), em
  vez de `subprocess.run` sobre o script — mesmo código, mesmo artefato gravado em disco, execução
  transparente e depurável célula a célula.

## Entradas
- `data/trusted/house_segments.parquet`

## Saídas
- `data/trusted/features_raw_derived.parquet`
- `reports/figures/07_raw_derived.png`

## Resultado
`run_raw_derived_features_pipeline()` retornou `n_rows=21612`, `n_columns=59`, 6 `new_features`. Em
correlação bruta com `price_log` no TRAIN (n=15.100): `grade_condition_interaction` 0,530 (a mais
forte, isolada), `has_basement` 0,193, `log_sqft_lot` 0,162, `basement_ratio` 0,152, `was_renovated`
0,124, `years_since_renovation` 0,073 (a mais fraca). Números idênticos aos da execução anterior à
reestruturação em função chamável — comportamento zero-diferença confirmado.

## Ver também
- Relatório de fase: `reports/phase_reports/07_feature_engineering_raw_derived.md`
