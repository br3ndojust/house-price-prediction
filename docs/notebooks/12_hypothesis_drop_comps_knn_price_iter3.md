# Fase 12 (iteração 3) — hypothesis_drop_comps_knn_price_iter3 (Notebook: notebooks/12_hypothesis_drop_comps_knn_price_iter3.ipynb)

## Objetivo
Testar se remover `comps_knn_price` do conjunto oficial de 20 features melhora a generalização do
modelo final.

## Metodologia / por quê
`comps_knn_price` domina 40,5% do gain total (gain-based, XGBoost) — mais que o dobro da 2ª feature
(`sqft_living`, 19,1%) e quase tanto quanto as outras 19 features somadas (~59,5%). Essa discrepância
motiva a pergunta: será que o modelo generaliza melhor sem ela (menos dependência de uma única feature
target-derived, fase 06)? Avaliado com o mesmo rigor metodológico das fases 09/10/12: `GroupKFold(3)`
em `train` (mesmos hiperparâmetros oficiais do modelo final, XGBoost `n_estimators=400, max_depth=3,
learning_rate=0.05`) e um fit único em `train` avaliado em `test` nunca visto (mesmo método fase 11/12).

## Decisões-chave
Critério pré-registrado (P4 — definido ANTES de rodar, nunca ajustado depois do resultado):
- **TRAIN, `GroupKFold(3)`**: MAE do candidato (19 features) não pode piorar em relação ao baseline (20
  features).
- **TEST**: métrica decisiva para generalização (mesmo precedente da fase 12 iteração 2). MAE precisa
  cair ≥5% para adotar.
- **Adoção exige as duas condições** — TRAIN não piora E TEST melhora ≥5%. Resultado misto é reportado
  como inconclusivo, não adotado sem uma nova rodada de análise.
- **Quebra por `price_band` (P3)** — nenhuma banda pode piorar mais que 10%, mesmo que o número global
  passe no critério (mesma disciplina do critério de substituição de `docs/08_continuous_learning.md`,
  seção 4).

Se ADOTADA, a hipótese só viraria modelo oficial depois de refazer fase 10 (seleção) + fase 13 (refit
final + `val`) + fase 14 (contrato) com o conjunto de 19 features, via
`scripts/promote_hypothesis_drop_comps_knn_price.py` — não é uma fase que decide sozinha.

## Entradas
- `data/trusted/features_contextual.parquet`
- `artifacts/model_candidate.pkl`

## Saídas
- `reports/hypothesis_drop_comps_knn_price.json`
- `reports/figures/12_drop_comps_knn_price_iter3.png`

## Resultado
| Conjunto de features | MAE TRAIN GroupKFold(3) | MAE TEST (holdout) |
|---|---|---|
| Baseline (20 features, com `comps_knn_price`) | **\$76.906** | **\$104.221** |
| Sem `comps_knn_price` (19 features) | \$103.577 (+34,68%, piora) | \$99.015 (-4,99%) |

Por `price_band` em TEST: Luxury -9,61%, Premium -31,94%, Standard +5,02%, **Entry +34,07%**.

Remover `comps_knn_price` até melhora o MAE global de TEST (-4,99%) e melhora bastante `Luxury` e
`Premium` — mas piora fortemente o `TRAIN GroupKFold(3)` (+34,68%, viola a condição de não piorar) e
piora a banda `Entry` em +34,07% (viola o limite de 10% por banda). O ganho de TEST também é um "near
miss" que não atinge o corte decisivo de -5% (-4,99%).

**Decisão: `REJECT_KEEP_COMPS_KNN_PRICE`.** Nenhuma das três condições do critério pré-registrado foi
satisfeita; as duas primeiras já bastam para rejeitar (adoção exige todas). O ganho aparente na métrica
global de TEST esconderia uma piora severa e concentrada em `Entry` — exatamente o cenário que a
disciplina de quebra por `price_band` (P3) existe para capturar antes de uma decisão de promoção.
`comps_knn_price` permanece no conjunto oficial de 20 features; nenhuma reexecução de fase 10/13/14 é
necessária.

## Ver também
- Relatório: `reports/hypothesis_drop_comps_knn_price.json`
- `roadmap/project_roadmap.md` (seção 10 — critério de parada do loop de hipóteses)
