# Fase 10 — Model selection (Notebook: notebooks/10_model_selection.ipynb)

## Objetivo
Qual modelo generaliza melhor sob validação geográfica?

## Metodologia / por quê
Grid enxuto Ridge + XGBoost via `GroupKFold(zipcode)` — nunca `train_test_split` simples como
critério principal, pois zipcode é a unidade de generalização real deste problema (P1). Hipótese
de fundo: escolha de features pesa mais que escolha de algoritmo. O `technical_tie_check` compara o
gap entre 1º e 2º colocado contra a soma dos desvios-padrão dos folds — quando o gap é menor, o
resultado é reportado explicitamente como empate técnico (P4: nunca esconder incerteza atrás do
argmax). O modelo final é refeito sobre `train` completo (sem augmentation, decisão da fase 05
revertida na fase 12) com os hiperparâmetros vencedores.

## Decisões-chave
- `GroupKFold` agrupado por `zipcode` (não K-fold aleatório) para simular generalização geográfica
  real (P1).
- Métrica oficial de ranking: MAE em dólares (P4, critério pré-registrado); RMSE/MAPE/R²(log)
  calculados só como apoio, nunca mudam o critério oficial de promoção.
- Empate técnico é reportado como campo explícito do relatório (`technical_tie_check`), não
  escondido atrás do argmax.
- Modelo final é treinado sobre `train` inteiro (não sobre um único fold) e persistido como
  candidato — a decisão de promoção final só acontece na fase 13/14.

## Entradas
- `data/trusted/features_contextual.parquet`
- `configs/model.yaml`
- `data/trusted/feature_metadata.json`

## Saídas
- `artifacts/model_candidate.pkl`
- `reports/model_selection_report.json`
- `reports/figures/10_model_selection.png`

## Resultado
Vencedor: **XGBoost** (`n_estimators=400, max_depth=3, learning_rate=0.05`), MAE CV
**76.905,72 ± 6.648,79**, R²(log)=0,8873. Empate técnico com o 2º colocado (XGBoost
`n_estimators=200, max_depth=3, learning_rate=0.1`, MAE 77.253,46) — gap de 347,73 é bem menor que a
soma dos desvios-padrão dos folds (13.152,15), então `technical_tie_check` reporta `is_tie=True`.
Ridge (alpha=10.0) fica logo atrás com MAE 78.286,83 — mesmo padrão da execução original: escolha de
algoritmo importa menos que escolha de features. 11 candidatos avaliados no total (3 valores de
alpha para Ridge, 8 combinações de hiperparâmetros para XGBoost).

## Ver também
- Config: `configs/model.yaml`
- Relatório de fase: `reports/phase_reports/10_model_selection.md`
