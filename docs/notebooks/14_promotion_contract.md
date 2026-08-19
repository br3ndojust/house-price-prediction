# Fase 14 — promotion_contract (Notebook: notebooks/14_promotion_contract.ipynb)

## Objetivo
O candidato final está pronto para virar um contrato de produção auditável, e a inferência em
produção reproduz fielmente o pipeline de treino?

## Metodologia / por quê
Passo 1 (`scripts/write_production_contract.py::run_production_contract_pipeline`): monta
`artifacts/production_contract.yaml`, um "documento/pacote atômico, não substituição real de produção
(não há produção real neste desafio) — recomendação de promoção humana, nunca execução automática
(P6)". Usa `artifacts/model_final.pkl` (refit `train`+`test`, fase 13), não o `model_candidate.pkl`
intermediário da fase 10, e agrega schema de features, hiperparâmetros, métricas de CV/val,
pré-processamento (índice espacial), versões de dataset/split, contrato de validação e limitações
conhecidas explícitas.

Passo 2 (`scripts/predict.py::run_inference_demo_pipeline`): demo de inferência sobre
`data/raw/future_unseen_examples.csv` via `src/pipelines/inference.py::{load_production_artifacts,
predict_price}` — "paridade treino/produção (P5)": reusa as MESMAS funções de merge/feature
engineering usadas em treino, sem lógica duplicada/reescrita para produção. `property_age` é calculado
a partir de uma data de referência fixa (2026-08-16), decisão de design explícita para
reprodutibilidade do audit.

## Decisões-chave
- Contrato de produção é gerado a partir de `model_final.pkl` (pós fase 13), nunca do candidato
  intermediário.
- `status: RECOMMENDED_FOR_PROMOTION` é sempre uma recomendação — a promoção real exige aprovação
  humana explícita, o pipeline nunca promove sozinho (P6).
- Demo de inferência reusa o pipeline de produção real (`src/pipelines/inference.py`), não uma cópia
  simplificada, garantindo que o que é testado aqui é o que rodaria em produção (P5).

## Entradas
`data/trusted/feature_metadata.json`, `reports/model_selection_report.json`,
`reports/val_final_check.json`, `configs/comparable.yaml`, `configs/split.yaml` (passo 1);
`data/raw/future_unseen_examples.csv`, `artifacts/model_final.pkl`, `artifacts/spatial_index.pkl`,
`data/raw/zipcode_demographics.csv` (passo 2, via `load_production_artifacts`).

## Saídas
`artifacts/production_contract.yaml`, `reports/future_unseen_predictions.csv`,
`reports/inference_demo_report.json`, `reports/figures/14_inference_demo.png`.

## Resultado
Contrato gerado com `status: RECOMMENDED_FOR_PROMOTION`. Modelo XGBoost
(`n_estimators=400, max_depth=3, learning_rate=0.05`), treinado em `train`+`test`, CV MAE
US$ 76.905,72, CV R² log 0,8873, **MAE em `val` US$ 72.516,91**, MAPE 14,13% (os mesmos números
travados na fase 13). 20 features, índice espacial `k=30` (fit só em `train`), e 4 limitações
conhecidas explícitas (erro 2-4x maior em Luxury/waterfront/grade alto; sensibilidade de TEST/VAL à
composição de zipcode; lição da reversão de augmentation fase 05→12; ausência de recalibração por
early stopping além do refit da fase 13).

Demo de inferência: 100 imóveis de `future_unseen_examples.csv`, preço previsto com mediana
US$ 425.208,63, mínimo US$ 172.360,16, máximo US$ 2.466.707,50 — faixa plausível para King County.

## Ver também
- Relatório de fase: `reports/phase_reports/14_promotion_contract.md`
