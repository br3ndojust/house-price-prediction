# Fase 12 (iteração 2) — hypothesis_revert_augmentation_iter2 (Notebook: notebooks/12_hypothesis_revert_augmentation_iter2.ipynb)

## Objetivo
Testar se reverter a decisão de augmentation da fase 05 (Técnica B, perturbação controlada) — rodando o
pipeline **completo** (segmentação + features + modelo, não só o diagnóstico simplificado da fase 05) —
reduz o MAE em `test` (zipcode não visto).

## Metodologia / por quê
A fase 05 adotou a Técnica B por melhorar o `GroupKFold` MAE dentro de `train` em 20,5%. A fase 11
(Error Matrix) mostrou que essa mesma decisão piorou o MAE em `test` em 8,8% — um efeito sistemático
(vários zipcodes pioraram, não só um outlier), não capturado pela métrica de proxy usada na fase 05
(CV em `train`). Diferente da fase 05, aqui `test` é **deliberadamente a métrica decisiva**: a fase 11
mostrou que CV em `train` não é proxy confiável para generalização geográfica neste caso específico, e o
próprio propósito desta hipótese é checar generalização — não escolher entre candidatos como na fase 05.

## Decisões-chave
**Critério de aceite, definido ANTES do experimento (P4):** reverter para o `train` sem augmentation se
isso reduzir o MAE global em `test`. Uma única iteração — critério de parada do roadmap (seção 10).

## Entradas
- `data/processed/house_clean.parquet`
- `data/processed/split_assignment.parquet`
- `reports/model_selection_report.json`

## Saídas
- `reports/fase12_experiment.json`
- `reports/figures/12_revert_augmentation_iter2.png`

## Resultado
| Variante | MAE global TEST |
|---|---|
| Sem augmentation (pipeline completo) | **\$104.221** |
| Técnica B adotada (fase 05) | \$109.083 |

Reverter para o `train` original reduz o MAE em `test` em ~4,46%. Quebra por banda mostra que o efeito
não é uniforme: `Luxury` até melhora ligeiramente com a Técnica B, mas `Premium`, `Standard` e `Entry`
pioram consistentemente com augmentation — o MAE global, que é a métrica decisiva do critério
pré-registrado, piora com a Técnica B.

**Decisão: `REVERT_TO_NO_AUGMENTATION`.** `data/trusted/train_for_pipeline.parquet` reflete o `train`
original (revertido historicamente como consequência desta decisão — ver `reports/AUDIT_LOG.md`, fase
12 iteração 2); as fases 06-11 já foram reexecutadas sobre o `train` revertido. Lição registrada (P4):
o critério "nunca usar `test` para escolher entre candidatos" é correto como prática geral, mas quando a
métrica de proxy (CV em `train`) diverge da métrica real de interesse (generalização geográfica), a
decisão inicial pode ser errada e só é corrigível num ciclo de hipótese posterior — exatamente o
propósito desta fase. Ambas as decisões (fase 05 e fase 12) permanecem registradas, sem apagar a
original.

## Ver também
- Relatório: `reports/fase12_experiment.json`
- `roadmap/project_roadmap.md` (seção 10 — critério de parada do loop de hipóteses)
