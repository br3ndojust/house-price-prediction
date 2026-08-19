# Fase 12 (iteração 1) — hypothesis_k_sweep_iter1 (Notebook: notebooks/12_hypothesis_k_sweep_iter1.ipynb)

## Objetivo
Testar se um k do índice espacial diferente do baseline de produção (k=30) reduz o erro de `Luxury`
(visto no Error Matrix da fase 09/11 como 2-4x pior que outras bandas) sem piorar o MAE global.

## Metodologia / por quê
`comps_knn_price`/`local_grade_percentile` usam k=30 vizinhos espaciais (`configs/comparable.yaml`,
baseline real de produção — os notebooks/relatórios das fases 06/07/09 tinham um erro de digitação
narrando "k=15", mas o código sempre leu o config corretamente; só o texto estava errado, corrigido
nesta fase, ver `reports/AUDIT_LOG.md`). Imóveis de luxo/waterfront são geograficamente esparsos (fase
03: só 163 imóveis no cluster físico de luxo em todo o dataset) — um k maior pode estabilizar mais a
estimativa nesses casos esparsos, melhorando o segmento que mais precisa dela. Testados k=15 (mais
local) e k=50 (mais suavizado) contra o baseline de produção (k=30), refazendo o fit do modelo XGBoost
vencedor (mesmos hiperparâmetros da fase 10) para cada variante de features espaciais.

## Decisões-chave
**Critério de aceite, definido ANTES do experimento (P4):** promover um k diferente do baseline só se o
MAE de `Luxury` em TEST cair ≥5% **e** o MAE global não piorar mais que 1%. Uma única iteração — não um
loop aberto perseguindo o segmento de luxo (critério de parada, `roadmap/project_roadmap.md` seção 10).

## Entradas
- `data/trusted/house_segments.parquet`
- `data/trusted/feature_metadata.json`
- `reports/model_selection_report.json`
- `reports/experiments/ledger.csv`

## Saídas
- `reports/fase10_experiment.json`
- `reports/experiments/ledger.csv` (3 novas linhas: `fase10_k15`, `fase10_k30`, `fase10_k50`)
- `reports/figures/12_k_sweep_iter1.png`

## Resultado
| k | MAE global (TEST) | MAE Luxury (TEST) |
|---|---|---|
| 15 | \$111.335 (+6,83%) | \$211.114 (-1,52%, piora) |
| **30 (baseline)** | **\$104.221** | **\$207.942** |
| 50 | \$100.476 (-3,59%) | \$198.453 (-4,56%) |

`k=50` melhora tanto o MAE global (-3,59%) quanto o Luxury (-4,56%) — na direção certa — mas não atinge
o corte pré-registrado de -5% em Luxury (near miss). `k=15` piora nos dois eixos.

**Decisão: `REJECT_KEEP_K30`.** Mantido `k=30` em produção (`configs/comparable.yaml` inalterado).
Disciplina P4: o critério não foi relaxado após ver o resultado, mesmo com `k=50` sendo uma melhora
direcionalmente correta por margem pequena. `k=50` fica registrado no ledger como pista para uma
eventual iteração futura, não descartado permanentemente. Fase encerrada após 1 iteração (critério de
parada do roadmap).

## Ver também
- Relatório: `reports/fase10_experiment.json`
- `roadmap/project_roadmap.md` (seção 10 — critério de parada do loop de hipóteses)
