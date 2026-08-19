# Fase 00 — validate_data (Notebook: notebooks/00_validate_data.ipynb)

## Objetivo
Os dados brutos batem com o contrato esperado (schema, nulos, cardinalidade)?

## Metodologia / por quê
`run_data_validation_pipeline` carrega os 3 CSVs de entrada e roda os contratos definidos em
`configs/data_contract.yaml` via `src/data/contracts.py`: schema de colunas esperado, unicidade de
chave (usando `(id, date)` como grão real — `id` sozinho não é chave única porque há revendas da mesma
propriedade em datas diferentes), nulos, ranges plausíveis (bathrooms, floors, grade, condition) e
ausência de vazamento (`price`/`id`/`date`) em `future_unseen_examples.csv`. É a primeira fase do
roteiro (P1) — nada é limpo ou transformado aqui, só validado; decisões de limpeza ficam para a fase 01.

## Decisões-chave
- Chave de unicidade tratada como `(id, date)`, não `id` isolado — 353 linhas compartilham `id`
  repetido por serem revendas da mesma propriedade em datas diferentes, um achado real do contrato,
  não um bug.
- Violações têm severidade (`error` bloqueia o pipeline, `warning` é só informativo) — permite
  registrar anomalias conhecidas (ex: outlier de digitação) sem travar a fase seguinte.
- `bedrooms>15` (1 linha, `id=2402100895`) é sinalizado aqui como warning, mas a decisão de removê-la
  é adiada para a fase 01 (fase 00 só constata, não decide limpeza).

## Entradas
- `data/raw/kc_house_data.csv`
- `data/raw/zipcode_demographics.csv`
- `data/raw/future_unseen_examples.csv`
- `configs/data_contract.yaml`

## Saídas
- `reports/data_validation_report.json`
- `reports/figures/00_row_counts.png`

## Resultado
- Contrato passou: `ok: True`.
- `kc_house_data.csv`: 21.613 linhas; `zipcode_demographics.csv`: 70 linhas;
  `future_unseen_examples.csv`: 100 linhas.
- Única violação: 1 linha com `bedrooms>15` (`id=2402100895`), severidade `warning`, não bloqueante.

## Ver também
- Config: `configs/data_contract.yaml`
- Relatório de fase: `reports/phase_reports/00_validate_data.md`
