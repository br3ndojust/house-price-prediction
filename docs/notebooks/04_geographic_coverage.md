# Fase 04 — geographic_coverage (Notebook: notebooks/04_geographic_coverage.ipynb)

## Objetivo
O split resultante tem zipcodes/regiões sub-representados em `train`? Isso correlaciona com erro?

## Metodologia / por quê
A fase 02 mostrou que `98039` é simultaneamente o zipcode mais caro e um dos mais escassos em volume
(50 imóveis). O notebook mede, só entre `train` e `test` (nunca `val` — P1: `val` só é tocado uma vez,
na fase 13): (1) comparação de distribuição de features-chave entre as partições; (2) distância
geográfica do zipcode de cada imóvel de `test` ao zipcode de `train` mais próximo; (3) cobertura no
espaço de features (distância ao vizinho mais próximo em `train`, fit só em `train`); (4) baldes de
escassez por zipcode em `train` (`LOW`/`MEDIUM`/`HIGH` por contagem); e treina um baseline rápido
(XGBoost, features físicas/demográficas cruas, sem engenharia ainda) para correlacionar essas métricas
de distância/escassez com o erro absoluto em `test`. Uma árvore de decisão pré-definida com o usuário
(antes da fase 05 rodar) converte essas correlações em um veredito STRONG/WEAK para a hipótese de
augmentation.

## Decisões-chave
- `val` fica fora da análise mesmo sendo só medição de distância/cobertura sem alvo (reforço de P1).
- Veredito é uma árvore de decisão fixa (definida antes de ver o resultado): STRONG exige train com
  zipcodes sub-representados **e** correlação significativa (`r>0.2`, `p<0.05`) entre erro e pelo menos
  um dos dois sinais de distância (geográfico ou de espaço de features).
- O veredito apenas orienta a expectativa da fase 05 — as 2 técnicas de augmentation são testadas de
  qualquer forma, por requisito explícito do usuário, mesmo que o veredito fosse WEAK.

## Entradas
- `data/processed/house_clean.parquet`
- `data/processed/split_assignment.parquet`

## Saídas
- `reports/geographic_coverage.json`
- `reports/figures/04_geographic_coverage.png`

## Resultado
- **3 zipcodes `LOW`** em `train` (n<100: 98024, 98039, 98148); 40 `MEDIUM` (100-500); 6 `HIGH` (>500).
- **Distância geográfica** (zip de `test` → zip de `train` mais próximo): 0,017° a 0,125° — sem
  isolamento geográfico extremo. Correlação com erro: **r=-0,027 (p=0,17), não significativa.**
- **Distância no espaço de features** (baseline XGBoost em features cruas): correlação com erro
  absoluto em `test`: **r=0,236 (p≈9×10⁻³⁵), altamente significativa.**
- **Veredito: hipótese de augmentation = STRONG** (`train_has_underrepresented_regions=true`,
  `error_correlates_with_scarcity_signals=true`) — o sinal real é imóvel fisicamente atípico frente ao
  que `train` cobre, não zipcode geograficamente isolado.

## Ver também
- Relatório de fase: `reports/phase_reports/04_geographic_coverage.md`
