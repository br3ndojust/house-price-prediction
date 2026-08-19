# Fase 01 — Entendimento dos dados (data_understanding)

**Pergunta:** o que os dados representam e estão prontos para uso?

**Hipótese:** variáveis físicas do imóvel + demografia do zipcode descrevem a formação de preço, uma vez
limpas inconsistências pontuais.

**Execução:** `src/data/clean.py`, `src/data/merge.py`, `scripts/build_dataset.py`, narrado em
`notebooks/01_data_understanding.ipynb`. Testado em `tests/test_clean_merge.py` (5 casos).

## Achados / decisões (cada uma verificada nos dados reais, não herdada)

| Decisão | Evidência |
|---|---|
| Remover 1 linha (`id=2402100895`, `bedrooms=33`) | `sqft_living=1620` → ~49 sqft/quarto, implausível; erro de digitação, não sinal de mercado |
| Manter 353 linhas de revenda (177 imóveis, `id` repetido) | Cada linha é venda real em data distinta — mantidas como observações independentes |
| `price_log = log1p(price)` como alvo | skew bruto 4.02 → skew log 0.43 (ver `reports/figures/01_price_log_transform.png`) |
| Merge 1:1 com `zipcode_demographics.csv` | 70/70 zipcodes cobertos nos dois lados, sem órfãos |

![price vs price_log](../figures/01_price_log_transform.png)

## Validado vs. descartado

- **Validado:** `log1p(price)` como alvo; revendas como observações válidas.
- **Descartado:** tratar `bedrooms=33` como outlier de mercado a manter (é erro de digitação).

## Decisão

`data/processed/house_clean.parquet` pronto (21.612 linhas × 48 colunas) para split canônico (fase 02).

## Riscos abertos

Checar na fase 02 se revendas do mesmo imóvel sempre caem no mesmo lado do split (esperado, já que o split
é por `zipcode`, mas não verificado formalmente ainda).

## Artefatos

`data/processed/house_clean.parquet`, `reports/build_dataset_log.json`,
`reports/figures/01_price_log_transform.png`, `notebooks/01_data_understanding.ipynb`.
