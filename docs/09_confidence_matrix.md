# 09 — Matriz / Score de Confiança

Extensão pós-plano sobre a API (`docs/02_execution_plan_api.md`): não estava no plano original de
implementação da API — foi pedida depois, como uma camada de confiança por predição, sobre os splits
`TEST`/`VAL` já existentes. Segue a mesma disciplina metodológica do resto do projeto (P1/P4/P5):
**TEST calibra, VAL verifica uma única vez, nada é reajustado depois de ver o resultado.**

```
        TEST (model_candidate, nunca viu TEST)
          │
          ├── mede erro real por segmento (price_band × property_cluster × cobertura no espaço de
          │    features × waterfront/grade≥10) — nunca um MAE único (P3)
          ├── Confidence Score (0-100) = rank empírico do erro esperado do segmento vs. toda a
          │    distribuição de TEST
          ├── curva score → erro calibrada por regressão isotônica (monótona, sem peso escolhido à
          │    mão) e cortes de categoria = onde essa curva cruza os quartis do próprio erro
          │    calibrado (P25/P50/P75) — não um número "que parece bom"
          └── tudo fica congelado em `artifacts/confidence_calibration.json`
                  │
                  ▼
                 VAL (model_final, tocado nesta checagem)
                  │
                  └── aplica a calibração congelada sem recalcular nada: a confiança prometida em
                       TEST realmente se sustenta em dado nunca usado para calibrar?
```

Implementação: `src/evaluation/confidence.py` (lógica reutilizável, P5) +
`scripts/build_confidence_matrix.py` (orquestra TEST→calibração, VAL→verificação, escreve os
artefatos) + `app/infrastructure/confidence/confidence_scorer.py` (adapter que reusa as MESMAS funções
de `src/` para pontuar uma predição nova em produção). Não é uma fase de decisão de modelo/feature —
não muda `model_final.pkl` nem o conjunto de 20 features (P4).

## 1. Sinais usados por predição

| Sinal | De onde vem | Reuso |
|---|---|---|
| Banda de preço (`price_band`) | já prevista em `/predictions` | — |
| Cluster físico (`property_cluster`) | já previsto em `/predictions` | — |
| `waterfront` / `grade≥10` | atributos de entrada | mesmo corte de `src/evaluation/error_matrix.py::characteristic_cuts` (erro 2-4x maior, fase 11/13) |
| Cobertura no espaço de features | distância (z-score) ao vizinho mais próximo em TRAIN | reusa `src/evaluation/geographic_coverage.py::feature_space_coverage` (fase 04), bucketizada em HIGH/MEDIUM/LOW por tercis de TEST |
| Tamanho de amostra do segmento | `n` de TEST no segmento (com *backoff* se pequeno) | mesmo limiar `LOW_SAMPLE_THRESHOLD=20` de `src/evaluation/error_matrix.py` |

**Erro absoluto esperado** e **erro percentual esperado** (os dois primeiros itens pedidos) são a
saída da tabela de segmento, não um sinal de entrada — ver seção 2.

## 2. Tabela de erro esperado por segmento, com backoff (P4)

Cada predição cai numa chave `price_band|property_cluster|coverage_bucket|hard_flag` (37 combinações
observadas em TEST, chamada `level0`). Segmentos com `n < 20` (mesmo limiar de `error_matrix.py`) não
têm amostra suficiente para estimar erro com confiança — o *lookup* faz *backoff* em cascata:

1. `level0` (banda × cluster × cobertura × hard-flag) — 37 segmentos em TEST.
2. `level1` (banda × cobertura × hard-flag, sem cluster) — 22 segmentos.
3. `level2` (cobertura × hard-flag, sem banda) — 6 segmentos.
4. `global` — sempre com amostra suficiente (n=2644).

Cada linha da API carrega `segment_level_used` e `low_sample` explícitos — nunca reporta um erro
esperado de uma amostra estatisticamente frágil sem avisar.

## 3. Confidence Score (0-100)

```
score = 100 × (1 − ECDF_TEST(erro_esperado_do_segmento))
```

`ECDF_TEST` é a distribuição empírica (ponderada por volume) do erro esperado de TODAS as linhas de
TEST — um segmento cujo erro esperado é menor que 90% do restante da população de TEST recebe score
≈90. Isso é puramente relativo à distribuição observada, sem peso escolhido à mão.

## 4. Cortes de categoria — calibrados, não escolhidos "porque parecem bons"

Erro individual bruto (por linha) tem cauda mais larga do que qualquer segmento consegue prometer — a
primeira tentativa desta calibração usou os quartis do erro bruto individual como corte e a categoria
"Alta confiança" ficou praticamente vazia (nenhum segmento chegava tão baixo quanto o melhor caso
individual isolado). Correção: os cortes usam os quartis do **erro já calibrado** pela regressão
isotônica (`isotonic.predict(score)` para cada linha de TEST), garantindo que os 4 cortes são sempre
atingíveis pela própria curva.

**Resultado desta execução** (`reports/confidence_matrix_report.json`, `n_test=2644`):

| Categoria | Score mínimo | APE calibrado (teto, TEST) | n em TEST | MAE mediano em TEST |
|---|---|---|---|---|
| 🟢 Alta confiança | **88** | 16,3% (P25) | 222 | US$ 62.483 |
| 🟡 Confiança moderada | **55** | 20,9% (P50) | 960 | US$ 40.653 |
| 🟠 Baixa confiança | **17** | 23,1% (P75) | 887 | US$ 78.478 |
| 🔴 Muito baixa / revisão | < 17 | — | 575 | US$ 126.801 |

Note que os cortes **não** ficaram em 80/60/40 (o exemplo ilustrativo do pedido original) — ficaram em
88/55/17, porque é onde a curva calibrada de erro realmente muda de regime nesta execução. Isso é o
ponto: os cortes vêm do dado, não de uma convenção redonda.

## 5. Matriz 2×2 (versão didática)

Eixos: **cobertura** (distância ao TRAIN, corte na mediana de TEST) × **erro esperado** (corte na
mediana do erro esperado do segmento, também de TEST) — os dois eixos usam só sinais conhecidos
**antes** de saber o resultado real (diferente de uma quebra retrospectiva por erro real, que só serve
para diagnóstico, não para confiança prospectiva — bug corrigido durante a implementação, ver commit).

|  | Erro esperado baixo | Erro esperado alto |
|---|---|---|
| **Cobertura alta** | 🟢 ALTA — MAE mediano US$ 43.038 (TEST) / US$ 41.023 (VAL) | 🟡 MÉDIA — US$ 61.869 (TEST) / US$ 39.641 (VAL) |
| **Cobertura baixa** | 🟡 MÉDIA — US$ 67.254 (TEST) / US$ 62.589 (VAL) | 🔴 BAIXA — US$ 127.078 (TEST) / US$ 67.700 (VAL) |

## 6. Verificação em VAL (`n_val=3868`) — a confiança prometida se sustenta?

| Categoria | n em VAL | APE observado (mediana) | Teto prometido (TEST) | Dentro do teto? |
|---|---|---|---|---|
| 🟢 Alta confiança | 628 | 12,3% | 16,3% | ✅ sim |
| 🟡 Confiança moderada | 1.540 | 10,3% | 20,9% | ✅ sim |
| 🟠 Baixa confiança | 1.051 | 10,8% | 23,1% | ✅ sim |
| 🔴 Muito baixa / revisão | 649 | 13,2% | — | — |

Todas as categorias com teto definido ficaram **dentro** do erro prometido em TEST — nenhuma promessa
foi quebrada no sentido "prometeu menos erro do que entregou".

### Limitação observada (honestidade técnica, P4)

A **ordem** das categorias não se sustentou perfeitamente em VAL: "Alta confiança" (12,3% de APE
mediano) teve erro observado **maior** que "Confiança moderada" (10,3%) e "Baixa confiança" (10,8%) —
`is_monotonic_by_category: false` no relatório. A matriz 2×2 (seção 5), mais simples/robusta, se
comporta melhor em VAL (ALTA=10,1% claramente o melhor).

Leitura honesta: a categorização fina de 4 níveis é sensível ao ruído de segmentos com poucas
combinações distintas (~37 células), e ao menos parte do efeito é o padrão já documentado na seção
11.3 do `PROJETO_VISAO_GERAL.md` — a banda `Entry` tem MAPE historicamente **mais alto** que
`Standard`/`Premium` (viés de superestimação), não um MAPE baixo só porque o imóvel é barato. Como
parte dos segmentos "Alta confiança" desta calibração vem de bandas mais baratas, esse viés específico
de banda puxa o desempenho observado em VAL para baixo dentro dessa categoria, mesmo com o teto de
erro (seção anterior) tendo sido respeitado.

**Decisão:** mantido como está, documentado abertamente (mesma disciplina da seção 9 do
`PROJETO_VISAO_GERAL.md` sobre o erro 2-4x maior em Luxury — aceito e monitorado, não escondido, não
"reajustado até ficar bonito"). Uso recomendado: confiar no **teto de erro por categoria** (sempre
respeitado em VAL) e na **matriz 2×2** para leitura rápida; tratar a ordem fina de 4 categorias como
indicativa, não como um ranking perfeitamente monotônico — texto explícito nas respostas da API/portal.

## 7. Onde isso vive no código

| Arquivo | Papel |
|---|---|
| `src/evaluation/confidence.py` | lógica reutilizável — segmentos, backoff, score, isotônica, cortes, matriz 2×2 |
| `scripts/build_confidence_matrix.py` | orquestra TEST (calibra) → VAL (verifica), escreve os artefatos |
| `artifacts/confidence_calibration.json` | artefato congelado consumido pela API (lookup + ECDF + cortes) |
| `reports/confidence_matrix_report.json` | relatório humano (números desta seção) |
| `tests/test_confidence.py` | testes unitários de `src/evaluation/confidence.py` |
| `app/domain/entities/prediction_confidence.py` | entidade `PredictionConfidence` |
| `app/domain/ports/confidence_service.py` | port `ConfidenceService` |
| `app/infrastructure/confidence/confidence_scorer.py` | adapter — carrega o artefato, reusa `src/evaluation/confidence.py` |
| `app/api/v1/routers/confidence.py` | `POST /predictions/confidence`, `GET /model/confidence-calibration` |
| `portal/pages/1_Predições.py` | mostra o Confidence Score de cada predição testada no portal |

Sem página dedicada de metodologia/cortes no portal desde a redução a 3 páginas
(`docs/PROJETO_VISAO_GERAL.md`, seção 12.9) — os cortes e a metodologia completa ficam só neste
documento.

## 8. Como regerar a calibração

```bash
python scripts/build_confidence_matrix.py
```

Precisa de `artifacts/model_candidate.pkl` (TRAIN) e `artifacts/model_final.pkl` (TRAIN+TEST) já
treinados (fases 10/13 do pipeline principal). Reescreve `artifacts/confidence_calibration.json` e
`reports/confidence_matrix_report.json`. A API relê o artefato no próximo restart/reload — não há
`reload()` a quente dedicado para este artefato (diferente do modelo principal, P6); reiniciar o
processo/container é suficiente dado que a calibração muda com pouca frequência (só quando o
pipeline principal é retreinado).
