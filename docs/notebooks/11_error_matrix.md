# Fase 11 — Error Matrix (Notebook: notebooks/11_error_matrix.ipynb)

## Objetivo
Onde o modelo funciona bem, onde falha (visto por `test`, repetível)?

## Metodologia / por quê
Nunca reportar só a métrica agregada (P3, P4) — o erro precisa ser decomposto por segmento (banda de
preço, cluster de propriedade, zipcode, características) para expor onde o modelo é confiável e onde
não é. Roda só sobre `test`, que é repetível entre iterações do loop de hipótese (fase 12); `val`
nunca é tocado aqui (P1) — só na fase 13, uma única vez, depois que augmentation/features/modelo/
hipóteses estiverem travados. A `error_matrix_by` decompõe as métricas globais (MAE, RMSE, MAPE,
mediana, p90, bias) por grupo; `characteristic_cuts` faz o mesmo para cortes booleanos de interesse
(waterfront, grade alta, view, renovado).

## Decisões-chave
- Avaliação roda exclusivamente sobre `test`, nunca sobre `val` (P1: `val` sagrado, tocado uma única
  vez na fase 13).
- Métricas nunca reportadas só de forma agregada — sempre com breakdown por banda de preço, cluster,
  zipcode e característica (P3, P4).
- `r2_log_score` é a mesma função usada na CV da fase 10 e na checagem final da fase 13, para manter
  R² comparável entre as três partições (fonte única, P5).
- `bias` é reportado com sinal explícito (positivo = modelo subestima o preço em média) para não
  esconder viés sistemático atrás do erro absoluto.

## Entradas
- `artifacts/model_candidate.pkl`
- `data/trusted/features_contextual.parquet`

## Saídas
- `reports/error_matrix.json`
- `reports/figures/11_error_matrix.png`

## Resultado
**Global (TEST, n=2.644):** MAE **\$104.220,80**, RMSE \$169.529,41, R²(log)=0,7656 — de volta ao
nível da execução original (pré-augmentation), confirmando que a reversão da fase 12 corrigiu a
piora identificada na primeira passagem por esta fase.

**Por banda de preço:** `Luxury` (n=482) MAE \$207.942,47 — cerca de 3,7x o MAE de `Entry` (n=771,
\$56.521,62).

**Por característica/cluster:** `waterfront=1` (n=38) é o pior corte isolado, MAE \$326.143,87 (mais
de 3x o MAE global) — mesmo grupo do `property_cluster=2` (também n=38, mesmo MAE). `grade>=10`
(n=194, MAE \$265.444,80) e `view>0` (n=310, MAE \$193.044,53) também erram bem acima da média.

**Por zipcode:** `98006` é o mais difícil (MAE \$185.180,78, n=498, 18,8% do `test`) — zipcode caro e
de alta variância caiu inteiro em `test` pela composição do split geográfico (só 10 dos 70 zipcodes
do dataset caem inteiramente em `test`).

## Ver também
- Config: `configs/model.yaml`
- Relatório de fase: `reports/phase_reports/11_error_matrix.md`
