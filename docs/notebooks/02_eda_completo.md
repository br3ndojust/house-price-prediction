# Fase 02 — eda_completo (Notebook: notebooks/02_eda_completo.ipynb)

## Objetivo
Que correlações, distribuições, outliers e padrões geográficos existem — e como o dado se distribui
por zipcode, antes de qualquer decisão de split?

## Metodologia / por quê
`run_eda_pipeline` roda sobre 100% dos dados (`data/processed/house_clean.parquet`), antes de qualquer
split — descritivo, sem `.fit` (P1). Substitui a antiga fase 04 do projeto anterior (que rodava só em
`train`, depois do split) por uma EDA que enxerga o dataset inteiro antes de qualquer decisão de
particionamento. Calcula correlação de cada feature numérica com `price_log`, e usa
`src/evaluation/geographic_coverage.py::zip_representation_summary` para descobrir a representação
(contagem de imóveis, preço mediano) por zipcode — insumo direto para a fase 04
(`geographic_coverage`).

## Decisões-chave
- EDA roda sobre 100% dos dados, não só em `train` — decisões descritivas não podem depender de uma
  partição ainda não definida.
- Nenhuma decisão de feature/modelo é tomada nesta fase — é só leitura/evidência (P1).
- Zipcodes com `n<100` imóveis são sinalizados explicitamente como achado formal (não intuição), e
  motivam a fase 04 a checar se essa escassez correlaciona com erro do modelo.

## Entradas
- `data/processed/house_clean.parquet`

## Saídas
- `reports/eda_summary.json`
- `reports/zip_representation_summary.csv`
- `reports/figures/02_correlation_price_log.png`
- `reports/figures/02_geo_price_pattern.png`
- `reports/figures/02_zip_volume_distribution.png`

## Resultado
- 21.612 linhas, 70 zipcodes.
- Top 5 correlações com `price_log`: `grade` (0,704), `sqft_living` (0,695), `hous_val_amt` (0,631),
  `sqft_living15` (0,619), `per_bchlr` (0,611).
- Gradiente espacial confirmado — razão de ~8,05x entre a mediana de preço do zipcode mais caro e do
  mais barato.
- `n` de imóveis por zipcode varia de 50 a 601; 3 zipcodes têm menos de 100 imóveis (98039, 98148,
  98024). `98039` (Medina) é simultaneamente o zipcode mais caro e um dos mais escassos em volume —
  achado que motiva a fase 04.

## Ver também
- Relatório de fase: `reports/phase_reports/02_eda_completo.md`
