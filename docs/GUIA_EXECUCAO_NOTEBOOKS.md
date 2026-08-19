# Guia de Execução — do zero, só com notebooks

Objetivo deste guia: permitir que um avaliador clone o repositório e reproduza **todo** o pipeline —
dados brutos → modelo final → contrato de produção → API servindo predições — rodando apenas os
notebooks em `notebooks/`, na ordem, sem precisar tocar em `scripts/`/`Makefile`. Cada notebook chama
a mesma função `run_<fase>_pipeline()` que o script CLI equivalente usa (zero divergência de lógica —
ver [`docs/notebooks/`](notebooks/) para a documentação de cada fase individualmente).

## 1. Pré-requisitos

```bash
git clone <repo>
cd house_pricing_final
python -m venv .venv && .venv\Scripts\activate      # Windows (PowerShell/cmd)
# ou: source .venv/bin/activate                       # Linux/Mac
pip install -r requirements.txt
```

`requirements.txt` já inclui `jupyter`, `nbconvert` e `nbformat` — nenhuma instalação extra é
necessária para rodar os notebooks. Python 3.11+ recomendado (testado em 3.13).

Os 3 arquivos brutos do desafio já vêm versionados em `data/raw/` (`kc_house_data.csv`,
`zipcode_demographics.csv`, `future_unseen_examples.csv`) — nada para baixar à parte.

## 2. Como rodar os notebooks

**Opção A — interativo (recomendado para avaliar de verdade):**
```bash
jupyter lab    # ou: jupyter notebook
```
Abra `notebooks/00_validate_data.ipynb` e rode todas as células (`Run All`); repita em ordem até
`15_confidence_matrix.ipynb`. Cada notebook é independente na leitura (lê os artefatos que o anterior
gravou em disco), então dá para inspecionar/pausar em qualquer fase antes de seguir.

**Opção B — linha de comando (para rodar tudo de uma vez, sem abrir o Jupyter):**
```bash
for nb in notebooks/*.ipynb; do
  python -m nbconvert --to notebook --execute --inplace "$nb" --ExecutePreprocessor.timeout=1200
done
```
(PowerShell: `Get-ChildItem notebooks\*.ipynb | ForEach-Object { python -m nbconvert --to notebook --execute --inplace $_.FullName --ExecutePreprocessor.timeout=1200 }`)

Isso executa e sobrescreve cada `.ipynb` in-place com outputs frescos (gráficos, tabelas, métricas
reais) — é a prova de que o projeto é replicável do zero, não só narrado.

**Modo clean-room (opcional, prova reprodutibilidade total):** antes do passo acima, apague os
artefatos gerados — `data/processed/`, `data/trusted/`, `artifacts/*.pkl`, `artifacts/*.yaml`,
`artifacts/*.json`, `reports/*.json`, `reports/*.csv` (exceto `data/raw/` e `configs/`, que são
entrada) — e rode os 18 notebooks em ordem. Se tudo terminar sem erro e os números baterem com os já
documentados em `docs/notebooks/*.md`, a reprodutibilidade está provada.

## 3. Ordem de execução e o que avaliar em cada fase

| # | Notebook | O que avaliar |
|---|---|---|
| 00 | `00_validate_data.ipynb` | Contrato de schema dos 3 CSVs — nulos, cardinalidade, ranges. |
| 01 | `01_data_understanding.ipynb` | Limpeza + merge físico↔demográfico. Dicionário de variáveis. |
| 02 | `02_eda_completo.ipynb` | Correlações, outliers, padrões geográficos — **Entregável 1**. |
| 03 | `03_canonical_split.ipynb` | Split por zipcode (não aleatório) — por que garante generalização. |
| 04 | `04_geographic_coverage.ipynb` | Zipcodes sub-representados em `train`; motiva a fase 05. |
| 05 | `05_augmentation_experiment.ipynb` | Teste de augmentation — decisão adotada/rejeitada, com critério. |
| 06 | `06_market_property_segmentation.ipynb` | Bandas de preço + clusters de perfil físico. |
| 07 | `07_feature_engineering_raw_derived.ipynb` | Features físicas/temporais derivadas. |
| 08 | `08_feature_engineering_contextual.ipynb` | Features espaciais (comparáveis KNN, percentil local). |
| 09 | `09_feature_validation.ipynb` | Leakage check + ablation — **por que cada feature foi promovida** (Entregável 2a). |
| 10 | `10_model_selection.ipynb` | Ridge vs XGBoost via GroupKFold — **escolha do modelo** (Entregável 2b). |
| 11 | `11_error_matrix.ipynb` | Erro por segmento em `test` — onde o modelo falha, honestamente. |
| 12 | `12_hypothesis_*_iterN.ipynb` (3 notebooks) | Loop de hipóteses motivado pelo Error Matrix — cada um com critério de aceite definido antes do resultado. |
| 13 | `13_final_refit_and_val_check.ipynb` | **O número que importa**: MAE em `val`, tocado uma única vez — resposta definitiva de generalização (Entregável 2c). |
| 14 | `14_promotion_contract.ipynb` | Contrato de produção + demo de inferência. |
| 15 | `15_confidence_matrix.ipynb` | Score de confiança por predição, calibrado em `test`/verificado em `val`. |

**Se o objetivo é só responder "o modelo é bom?"**, os dois notebooks decisivos são o **11**
(onde erra, por quanto, em qual segmento) e o **13** (MAE em `val` — a métrica que não foi otimizada
contra, a prova real de generalização). `reports/AUDIT_LOG.md` e
`reports/stakeholder_summary.html` resumem esses números em linguagem de negócio.

## 4. A partir de qual etapa dá para testar a API

**A API sobe e responde predições assim que o notebook 13 terminar.** Ela carrega, obrigatoriamente
no boot: `artifacts/model_final.pkl` (gravado pelo notebook 13), `artifacts/spatial_index.pkl`
(gravado pelo notebook 08) e `data/raw/zipcode_demographics.csv` (já existe desde o clone). Sem
esses três, o `uvicorn` nem sobe.

```bash
pip install -r app/requirements.txt
uvicorn app.main:app --reload --port 8000
# Swagger UI: http://localhost:8000/docs
```

Teste rápido de predição:
```bash
curl -X POST http://localhost:8000/api/v1/predictions -H "Content-Type: application/json" -d "{\"bedrooms\":3,\"bathrooms\":2.0,\"sqft_living\":1800,\"sqft_lot\":5000,\"floors\":1.0,\"waterfront\":0,\"view\":0,\"condition\":3,\"grade\":7,\"sqft_above\":1800,\"sqft_basement\":0,\"yr_built\":1990,\"yr_renovated\":0,\"zipcode\":98101,\"lat\":47.6,\"long\":-122.3,\"sqft_living15\":1800,\"sqft_lot15\":5000}"
```

**O que ainda falta nesse ponto (14/15 completam, mas API já funciona sem eles):**
- Sem o **notebook 14**: `GET /model/info` não mostra os metadados do contrato de produção (volta
  `{}` nesse campo) — o resto da API funciona normal.
- Sem o **notebook 15**: toda predição volta com `confidence_score`/`confidence_category` como
  `null`, e `POST /predictions/confidence` retorna erro pedindo pra rodar essa fase primeiro. Fora
  isso, `POST /api/v1/predictions` funciona normalmente.

Ou seja: **13 → API básica funciona. 14 → contrato completo. 15 → score de confiança completo.**
Rodar os 3 antes de testar a API dá a experiência completa (recomendado), mas tecnicamente dá pra
começar a testar logo após o 13.

## 5. Sanity check automatizado

```bash
python -m pytest tests/ -q
```
85 testes cobrindo dados, features, segmentação, modelo, pipeline e confiança — independente dos
notebooks (testam `src/` diretamente), servem como checagem rápida de que nada quebrou.
