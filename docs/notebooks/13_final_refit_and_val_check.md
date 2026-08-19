# Fase 13 — final_refit_and_val_check (Notebook: notebooks/13_final_refit_and_val_check.ipynb)

## Objetivo
O candidato final (features/modelo/hipóteses todos travados nas fases 02-12) generaliza para dado
nunca visto — a checagem única e sagrada em `val`?

## Metodologia / por quê
Passo 1 (`scripts/finalize_model.py::run_final_refit_pipeline`): refit único combinando `train`+`test`
(maximiza dado usado no modelo final), depois checagem ÚNICA em `val` — "P1: val sagrado, tocado uma
única vez, depois de tudo (augmentation, features, modelo, hipóteses) estar travado". Fora do fluxo
experimental original, um job de retraining via API que rode esta função reavalia `val` a cada novo
candidato, igual a qualquer pipeline de reavaliação contínua real — o contrato de honestidade é
preservado de outra forma: nenhuma decisão de feature/modelo/hipótese olha para `val`, ele só mede o
candidato já travado, nunca escolhe entre alternativas. A função é determinística (random_state fixo),
por isso é seguro reexecutá-la neste notebook via `nbconvert` para fins de reprodutibilidade — isso é
diferente de "tocar val repetidas vezes está sempre bem"; nenhuma decisão deste projeto usa o
resultado desta reexecução para ajustar algo.

Passo 2 (`scripts/analyze_val_predictions.py::run_val_prediction_analysis_pipeline`): análise pós-hoc
das previsões em `val` — erro por região/tipo de imóvel/faixa de preço, e bandas de confiança empírica
(conformal simples): calibração via `model_candidate.pkl` (só viu `train`) aplicado em `test` (dado
nunca visto por esse modelo) → quantis de resíduo por banda de preço; validação aplicando esses
quantis às previsões de `model_final.pkl` em `val` (dado nunca usado para calibrar), medindo cobertura
real. Este script era órfão (nenhum notebook o chamava) e é adotado aqui como passo 2. **Não é uma
fase de decisão (P4)** — não muda nenhum artefato de modelo/feature, é leitura sobre artefatos já
travados (fases 09-14), para reporte visual.

## Decisões-chave
- `val` é tocado exatamente uma vez neste notebook (passo 1) — nenhuma fase anterior usou `val` para
  decidir feature, modelo, ou hipótese.
- `artifacts/model_final.pkl` (refit em `train`+`test`) passa a ser o candidato final, usado por todas
  as fases seguintes (14, 15) em vez de `model_candidate.pkl`.
- Passo 2 é explicitamente não-decisório: apenas mede se a banda de confiança calibrada em `test` se
  sustenta em `val`, sem reajustar nada com base no resultado.

## Entradas
`artifacts/model_candidate.pkl`, `data/trusted/features_contextual.parquet`,
`reports/ablation_fase09.json` (passo 2).

## Saídas
`artifacts/model_final.pkl`, `reports/val_final_check.json`,
`reports/prediction_confidence_analysis.json`, `reports/figures/13_val_final_check.png`,
`reports/figures/13_val_actual_vs_predicted.png`.

## Resultado
Checagem final em `val` (n=3.868, tocado UMA ÚNICA VEZ): MAE global **US$ 72.516,91**, RMSE
US$ 112.596,18, MAPE 14,13%, median APE 11,15%, bias +US$ 19.535,53 (subestimação leve em média).
Melhor que o MAE em `test` (fase 11) — composição de zipcodes de `val` não inclui um equivalente ao
zipcode difícil que dominava `test`, e o refit final usa mais dado (17.744 linhas de `train`+`test`).
Por banda de preço, `Luxury` continua com erro várias vezes maior que `Entry` — o mesmo padrão 2-4x já
documentado no Error Matrix (fase 11).

Passo 2 (cobertura do intervalo de confiança empírico, checada em `val`): cobertura observada de
84,95% para o intervalo nominal de 80% e 94,03% para o intervalo nominal de 90% — ambas **acima** do
alvo nominal (intervalo levemente conservador, não otimista). Este passo não tomou nenhuma decisão de
modelo/feature/hipótese.

## Ver também
- Relatório de fase: `reports/phase_reports/13_final_refit_and_val_check.md`
