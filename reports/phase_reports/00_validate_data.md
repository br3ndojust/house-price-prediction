# Fase 00 — Validação de contrato de dados

**Pergunta:** os dados brutos batem com o contrato esperado (schema, nulos, cardinalidade)?

**Hipótese:** os 3 CSVs seguem um schema estável o suficiente para servir de base a um pipeline
reprodutível, com poucas exceções pontuais conhecidas.

**Execução:** `scripts/validate_data.py` (usa `src/data/contracts.py`, contrato em
`configs/data_contract.yaml`), narrado em `notebooks/00_validate_data.ipynb`. Testado em
`tests/test_data_contracts.py` (7 casos, incluindo os 3 arquivos reais).

## Achados

- `kc_house_data.csv`: 21.613 linhas, 21 colunas, schema ok, 0 nulos.
- **Achado real não previsto no roadmap**: `id` sozinho **não é chave única** — 353 linhas (177 imóveis)
  compartilham `id` repetido com `date`/`price` diferentes, ou seja, são **revendas da mesma propriedade**
  dentro do período coberto pelo dataset. O contrato foi ajustado para usar `(id, date)` como grão real
  (0 duplicatas nesse par). Isso é um insight para a fase 01: decidir se revendas contam como observações
  independentes ou precisam de tratamento especial (ex: leakage entre train/test se a mesma propriedade
  cair nos dois lados — mitigado pelo split ser por `zipcode`, não por `id`, mas vale checagem futura).
- 1 linha com `bedrooms=33` (id `2402100895`) — outlier de digitação já conhecido do blueprint, confirmado
  de forma independente aqui (não copiado, checado nos dados reais). Sinalizado como warning, decisão de
  limpeza adiada para fase 01.
- `zipcode_demographics.csv`: 70 zipcodes, sem nulos, sem duplicatas.
- `future_unseen_examples.csv`: 100 linhas, sem `price`/`id`/`date` (ok — sem vazamento).

## Validado vs. descartado

- **Validado:** schema dos 3 arquivos, ausência de vazamento em `future_unseen_examples.csv`.
- **Descartado:** usar `id` como chave única (não é real neste dataset) — corrigido para `(id, date)`.

## Decisão

Contrato passa (`ok: true`, `reports/data_validation_report.json`). Pipeline liberado para fase 01.

## Artefatos

`configs/data_contract.yaml`, `src/data/contracts.py`, `scripts/validate_data.py`,
`tests/test_data_contracts.py`, `reports/data_validation_report.json`, `notebooks/00_validate_data.ipynb`.
