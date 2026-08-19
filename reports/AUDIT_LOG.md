# Audit Log — Previsão de Preços de Casas (King County / Seattle)

Documento consolidado de auditoria. Uma entrada por fase/bloco executado, em ordem cronológica. Cada
entrada aponta para o relatório detalhado da fase (`reports/phase_reports/`) e para os artefatos reais
gerados. Ver plano de execução completo, princípios (`P0`-`P7`) e roteiro em `roadmap/`.

Regras de leitura: este arquivo deve dar o quadro completo do projeto sem precisar abrir mais nada.

---

## Sumário executivo

**O que foi entregue:** os 4 entregáveis oficiais do desafio (entendimento dos dados, variáveis
importantes/modelo/generalização, estratégia de deploy, aprendizado contínuo) + comunicação com
stakeholders, através de 15 fases numeradas (00-14), cada uma com pergunta → hipótese →
análise/experimento → métricas → decisão → artefato (`P7`).

**Resultado final:** XGBoost (`n_estimators=400, max_depth=3, learning_rate=0.05`), 20 features,
refit em `train`+`test` (fase 13). MAE em `val` (checagem única, sagrada) = **\$72.517** (MAPE 14,1%).
Erro 2-4x maior em Luxury/waterfront/grade alto — limitação conhecida, aceita conscientemente, não
escondida (ver seção 5 de `docs/07_deploy_strategy.md` para o SLA diferenciado proposto).

**Decisões-chave e por que elas importam para quem for auditar:**
- **`zipcode` nunca é feature** — só chave de merge/split/diagnóstico (`P1`). Split geográfico
  (`GroupShuffleSplit`) garante que a avaliação reflete zipcode nunca visto, não holdout aleatório.
- **EDA pré-split, sobre 100% dos dados** (fase 02) — reordenação pedida em revisão do usuário, formaliza
  a distinção entre leitura descritiva (permitida em 100% dos dados) e transformação que aprende
  (`.fit`, sempre pós-split, só em `train`).
- **`val` tocado uma única vez** (fase 13), depois de augmentation/features/modelo/hipóteses estarem
  todos travados — outra correção da mesma revisão (antes disso, `val` era tocado cedo demais).
- **Decisão de augmentation revertida na fase 12** — a fase 05 adotou uma técnica que melhorava a
  métrica de treino (CV -20,5%) mas piorava a generalização real (`test` +8,8%); só descoberto porque
  o Error Matrix (fase 11) quebrou o resultado por segmento em vez de aceitar o número agregado. A
  correção foi propagada por todas as fases downstream (06-14) antes de prosseguir. Esta é, sozinha, a
  melhor evidência prática de por que o projeto nunca decide por métrica agregada isolada (`P4`).
- **Empates técnicos sempre reportados**, nunca escondidos atrás de um argmax (fase 10: XGBoost vs.
  Ridge; fase 12 iteração 1: k=30 vs. k=50 no índice espacial).

**Dívidas técnicas conscientes, não escondidas:**
- Erro em Luxury/waterfront/grade alto segue 2-4x o mercado de massa — não resolvido, encaminhado como
  requisito de SLA/monitoramento por segmento (`docs/07`), não perseguido indefinidamente (critério de
  parada, `roadmap/project_roadmap.md` seção 10).
- Sem recalibração por early stopping além do refit simples da fase 13 (simplificação consciente do
  escopo enxuto desta execução).
- Composição de zipcodes de `test`/`val` (10/11 de 70 totais) tem variância alta — um único zipcode
  difícil (`98006`) pode dominar a métrica agregada de uma partição; mitigado reportando sempre por
  corte (Error Matrix), nunca só agregado.

---

---

## ⚠ Reestruturação do projeto (2026-08-16)

Após o Bloco 5 (fases 00-11 originais, todas commitadas), revisão do usuário identificou 2 problemas
metodológicos reais e pediu reestruturação antes de prosseguir. Plano completo em
[`docs/01_execution_plan.md`](../docs/01_execution_plan.md). Resumo:

1. **EDA rodava só em `train`, depois do split** — corrigido: EDA descritivo completo passa a rodar sobre
   **100% dos dados**, antes do split (nova fase 02), incluindo descoberta de representação geográfica.
2. **`val` era tocado cedo demais** (dentro da antiga fase 09, antes do loop de hipóteses da fase 10) —
   corrigido: `val` só é tocado **uma única vez**, no fim de tudo, via um refit final `train`+`test` (nova
   fase 13).

Além disso, 2 fases novas foram inseridas: **04 `geographic_coverage`** (cobertura geográfica/escassez por
zipcode, motivada pela observação do usuário de que a maioria dos zipcodes concentra em `train`) e
**05 `augmentation_experiment`** (compara baseline sem augmentation vs. 2 técnicas — SMOGN e perturbação
controlada —, só sobre `train`). E o roadmap ganhou overlay explícito de **CRISP-DM** (seção 5.1 de
`roadmap/project_roadmap.md`).

**As entradas do Bloco 0-5 abaixo permanecem como registro histórico real das decisões já tomadas** — não
foram apagadas, só a numeração/ordem de execução das fases mudou. Mapeamento de números antigos → novos:
`00`→`00`, `01`→`01`, `04`(geographic_structure)→`02`(eda_completo), `02`(canonical_split)→`03`,
`03`(market_property_segmentation)→`06`, `05`(raw_derived)→`07`, `06`(contextual)→`08`,
`07`(feature_validation)→`09`, `08`(model_selection)→`10`, `09`(error_matrix)→`11` (sem a checagem de
`val`, que migra pra nova fase `13`), `10`(hipótese k-sweep)→`12`, `11`(promotion_contract)→`14`. A fase
`13` (`final_refit_and_val_check`) é inteiramente nova.

As entradas de fase abaixo a partir daqui (pós-reestruturação) seguem a numeração nova.

---

## Fase 02 (nova) — EDA completo (100% dos dados, pré-split)

**Data:** 2026-08-16 · **Relatório:** [`phase_reports/02_eda_completo.md`](phase_reports/02_eda_completo.md)

**Decisão:** substitui a antiga fase 04 — mesma análise (correlação/outlier/padrão geográfico), agora
sobre 100% dos dados (21.612 linhas), pré-split. Top correlações com `price_log`: `grade` (0,704),
`sqft_living` (0,695), `hous_val_amt` (0,631) — praticamente idênticas às da antiga fase 04 (que rodava
só em `train`), confirmando que a mudança de escopo não altera a conclusão de fundo.

**Achado não previsto (motiva a fase 04 nova):** zipcode `98039` (Medina) é simultaneamente o mais caro
do dataset e um dos 3 zipcodes com menor volume de dado (50 imóveis, mínimo entre os 70). Descoberta de
representação geográfica (`zip_representation_summary`) feita nesta fase, antes do split.

**Desvio do roadmap:** nenhum de escopo — é a correção pedida pelo usuário (EDA pré-split).

**Artefatos:** `scripts/run_eda.py`, `src/evaluation/geographic_coverage.py`,
`reports/eda_summary.json`, `reports/zip_representation_summary.csv`, 3 figuras,
`notebooks/02_eda_completo.ipynb`.

---

## Fase 03 (era fase 02) — Split canônico

**Data:** 2026-08-16 · **Relatório:** [`phase_reports/03_canonical_split.md`](phase_reports/03_canonical_split.md)

**Decisão:** mesma lógica/números de antes (`GroupShuffleSplit` por zipcode: train 69,87%/49 zipcodes,
test 12,23%/10, val 17,90%/11, zero overlap) — só reordenada para rodar depois da fase 02 (EDA). Papel
de `train`/`test`/`val` formalizado: `test` relatado durante desenvolvimento mas nunca decide; `val`
tocado uma única vez, na fase 13.

**Desvio do roadmap:** nenhum — renumeração pura, lógica idêntica à antiga fase 02.

**Artefatos:** inalterados (mesmos de antes), notebook/relatório renomeados para fase 03.

---

## Fase 04 (nova) — Geographic Coverage & Data Sufficiency

**Data:** 2026-08-16 · **Relatório:** [`phase_reports/04_geographic_coverage.md`](phase_reports/04_geographic_coverage.md)

**Decisão:** veredito **augmentation_hypothesis = STRONG**. 3 zipcodes `LOW` (<100 imóveis) em `train`.
Distância geográfica pura NÃO correlaciona com erro (r=-0,027, ns). Distância no espaço de features
correlaciona fortemente (r=0,236, p≈9×10⁻³⁵) — imóvel fisicamente atípico frente ao que `train` cobre é
sistematicamente mais difícil de prever, independente de geografia.

**Desvio do roadmap:** nenhum — fase inteiramente nova, pedida pelo usuário.

**Achado não previsto:** o mecanismo de dificuldade não é geográfico (distância a zipcodes vizinhos),
é de cobertura no espaço de features — isso muda a estratégia de augmentation da fase 05 (focar em
reforçar cobertura de features/raridade de alvo, não tentar "cobrir mais zipcode").

**Riscos/dívidas abertas:** amostra de correlação é sobre 2.644 linhas de `test` (n razoável, mas os 3
zipcodes `LOW` específicos têm poucochíssimos imóveis cada — qualquer conclusão zip-a-zip precisa de
cautela estatística, P4).

**Artefatos:** `scripts/analyze_geographic_coverage.py`, `reports/geographic_coverage.json`,
`reports/figures/04_geographic_coverage.png`, `notebooks/04_geographic_coverage.ipynb`.

---

## Fase 05 (nova) — Data Augmentation Experiment

**Data:** 2026-08-16 · **Relatório:** [`phase_reports/05_augmentation_experiment.md`](phase_reports/05_augmentation_experiment.md)

**Decisão:** **Técnica B (perturbação controlada) adotada** — MAE de `GroupKFold(3)` dentro de `train`
cai 11,4% (107.515→95.276), acima do critério pré-registrado (≥2%). Técnica A (SMOGN) rejeitada (piora
4,5%). `data/trusted/train_for_pipeline.parquet` (22.650 linhas) segue para as fases 06+.

**Desvio do roadmap:** nenhum — fase inteiramente nova, pedida pelo usuário. Ajuste do usuário aplicado:
baseline sem augmentation avaliado com o mesmo rigor completo das técnicas (não um "modelo rápido" só
de triagem).

**Achado honesto (não escondido):** o ganho de Técnica B quase não aparece em `test` (geral: 97.416→
97.397; subconjunto atípico da fase 04: 116.334→116.144) — sugere que o mecanismo real é tornar `train`
mais denso perto de pontos já existentes (reduz variância de CV), não necessariamente estender
cobertura para o imóvel atípico que motivou a hipótese da fase 04. Critério pré-registrado foi
respeitado mesmo assim (P4: não mover a régua após ver o resultado), com a ressalva documentada.

**Riscos/dívidas abertas:** o mecanismo de ganho de Técnica B não é o hipotetizado — se o erro em
segmentos atípicos/luxo persistir nas fases seguintes (Error Matrix, fase 11), este achado é o primeiro
lugar a revisitar.

**Artefatos:** `src/augmentation/smogn.py`, `src/augmentation/perturbation.py`,
`scripts/run_augmentation_experiment.py`, `tests/test_augmentation.py`,
`reports/augmentation_experiment.json`, `data/trusted/train_for_pipeline.parquet`,
`reports/figures/05_augmentation_comparison.png`, `notebooks/05_augmentation_experiment.ipynb`.

---

## Fase 06 (era fase 03) — Segmentação de mercado e perfil físico

**Data:** 2026-08-16 · **Relatório:** [`phase_reports/06_market_property_segmentation.md`](phase_reports/06_market_property_segmentation.md)

**Decisão:** mesma metodologia da execução anterior (quartil de preço + cluster físico k=3). Rodou
primeiro sobre o `train` aumentado da fase 05 (cluster de luxo físico 163→222); **corrigido pós fase 12**
para rodar sobre `train` sem augmentation — números batem exatamente com a execução original
(cluster de luxo físico de volta a 163).

**Desvio do roadmap:** nenhum de escopo — renumeração + fonte de `train` atualizada (fase 05, revertida
na fase 12).

**Riscos/dívidas abertas:** nenhuma nova.

**Artefatos:** `scripts/run_segmentation.py`, `data/trusted/house_segments.parquet`,
`data/trusted/price_quartiles.json`, `artifacts/property_cluster_model.pkl`,
`reports/segmentation_report.json`, `reports/figures/06_segmentation.png`,
`notebooks/06_market_property_segmentation.ipynb`.

---

## Fase 07 (era fase 05) — Feature engineering RAW/DERIVED

**Data:** 2026-08-16 · **Relatório:** [`phase_reports/07_feature_engineering_raw_derived.md`](phase_reports/07_feature_engineering_raw_derived.md)

**Decisão:** 6 features geradas. **Corrigido pós fase 12** para `train` sem augmentation — correlação
bruta idêntica à execução original (`grade_condition_interaction` 0,530 top). Decisão de promoção
adiada pra fase 09.

**Desvio do roadmap:** nenhum — renumeração + fonte de `train` atualizada.

**Artefatos:** `data/trusted/features_raw_derived.parquet`, `reports/figures/07_raw_derived.png`,
`notebooks/07_feature_engineering_raw_derived.ipynb`.

---

## Fase 08 (era fase 06) — Feature engineering CONTEXTUAL

**Data:** 2026-08-16 · **Relatório:** [`phase_reports/08_feature_engineering_contextual.md`](phase_reports/08_feature_engineering_contextual.md)

**Decisão:** índice espacial (k=30). **Corrigido pós fase 12** para `train` sem augmentation —
`comps_knn_price` r TRAIN=0,833/TEST=0,601 (gap -0,232), idêntico à execução original, não leakage.

**Desvio do roadmap:** nenhum — renumeração + fonte de `train` atualizada.

**Artefatos:** `data/trusted/features_contextual.parquet`, `artifacts/spatial_index.pkl`,
`reports/figures/08_contextual.png`, `notebooks/08_feature_engineering_contextual.ipynb`.

---

## Fase 09 (era fase 07) — Feature validation (leakage + ablation)

**Data:** 2026-08-16 · **Relatório:** [`phase_reports/09_feature_validation.md`](phase_reports/09_feature_validation.md)

**Decisão:** rodou primeiro sobre `train` aumentado (MAE 95.276→75.741, -20,5%). **Corrigido pós fase
12** para `train` sem augmentation — conjunto final idêntico (contextual + `property_age`), MAE
107.515→77.198 (-28,2%), números batendo exatamente com a execução original. `raw_derived` rejeitado
nos dois casos (piora quando combinado com contextual).

**Desvio do roadmap:** nenhum — renumeração + recomputado sobre `train` aumentado.

**Riscos/dívidas abertas:** nenhuma nova.

**Artefatos:** `reports/ablation_fase09.json`, `data/trusted/feature_metadata.json`,
`data/processing/feature_registry.csv`, `reports/figures/09_ablation.png`,
`notebooks/09_feature_validation.ipynb`.

---

## Fase 10 (era fase 08) — Model selection

**Data:** 2026-08-16 · **Relatório:** [`phase_reports/10_model_selection.md`](phase_reports/10_model_selection.md)

**Decisão:** rodou primeiro sobre `train` aumentado (MAE CV 75.531). **Corrigido pós fase 12** para
`train` sem augmentation — vencedor XGBoost (`n_estimators=400, max_depth=3, learning_rate=0.05`), MAE
CV 76.906±6.649, R²(log)=0.887, idêntico à execução original. Empate técnico reportado (P4).

**Desvio do roadmap:** nenhum — renumeração + recomputado sobre `train` aumentado.

**Artefatos:** `artifacts/model_candidate.pkl`, `reports/model_selection_report.json`,
`reports/figures/10_model_selection.png`, `notebooks/10_model_selection.ipynb`.

---

## Fase 11 (era fase 09) — Error Matrix

**Data:** 2026-08-16 · **Relatório:** [`phase_reports/11_error_matrix.md`](phase_reports/11_error_matrix.md)

**Decisão:** primeira execução (com augmentation) deu MAE global TEST \$113.385 — pior que os
\$104.221 anteriores, motivando a fase 12. **Corrigido pós fase 12** (`train` sem augmentation): MAE
global TEST de volta a **\$104.221**, idêntico à execução original.

**Desvio do roadmap:** nenhum de escopo — mas achado relevante quanto à decisão da fase 05, resolvido
na fase 12.

**Achado crítico (não escondido, motivou a fase 12):** augmentation melhorou CV dentro de `train`
(-20,5%) mas piorou `test`/zipcode-não-visto (+8,8%) — efeito sistemático (vários zipcodes pioraram,
não só um outlier). A métrica de decisão da fase 05 (CV em `train`) não foi proxy confiável para
generalização geográfica aqui, mesmo seguindo a regra pré-registrada corretamente. Resolvido revertendo
a decisão na fase 12.

**Riscos/dívidas abertas:** nenhuma — resolvido na fase 12.

**Artefatos:** `reports/error_matrix.json`, `reports/figures/11_error_matrix.png`,
`notebooks/11_error_matrix.ipynb`.

---

## Fase 12, iteração 1 — Hipótese: k do índice espacial

**Data:** 2026-08-16 · **Relatório:** [`phase_reports/12_hypothesis_k_sweep_iter1.md`](phase_reports/12_hypothesis_k_sweep_iter1.md)

**Decisão:** conteúdo idêntico ao executado originalmente como "fase 10" antes da reestruturação (ver
entrada histórica mais abaixo, "Fase 10 — Iteração 1 (delimitada): k do índice espacial") — a hipótese
(k=15/30/50 no índice espacial de comparáveis) não dependia de EDA/split/augmentation, então os números
não mudam. Arquivos renomeados de `10_hypothesis_k_sweep.*`/`10_k_sweep.png` para
`12_hypothesis_k_sweep_iter1.*`/`12_k_sweep_iter1.png` para refletir a numeração atual (fase 12,
primeira de duas iterações do loop de hipótese). Resultado: `REJECT_KEEP_K30` (k=50 quase atinge o
critério de -5% em Luxury, fica em -4,56%).

**Desvio do roadmap:** nenhum — renumeração pura de arquivo, sem mudança de conteúdo/resultado.

**Artefatos:** `notebooks/12_hypothesis_k_sweep_iter1.ipynb`, `reports/figures/12_k_sweep_iter1.png`
(conteúdo original de `scripts/run_experiment.py` antes de ser reescrito para a iteração 2).

---

## Fase 12, iteração 2 — Hipótese: reverter augmentation?

**Data:** 2026-08-16 · **Relatório:** [`phase_reports/12_hypothesis_revert_augmentation_iter2.md`](phase_reports/12_hypothesis_revert_augmentation_iter2.md)

**Decisão:** **`REVERT_TO_NO_AUGMENTATION`.** Pipeline completo (não só o diagnóstico da fase 05) sem
augmentation: MAE TEST 106.254; com Técnica B: 113.384. Reverter reduz ~6,3%.
`data/trusted/train_for_pipeline.parquet` volta a ser o `train` original (15.100 linhas). **A decisão
da fase 05 é revisada** (ver nota adicionada na entrada da fase 05 acima) — não apagada, corrigida para
frente (P4).

**Desvio do roadmap:** nenhum — exercita o loop de hipóteses (fase 12+) exatamente como previsto na
seção 10, inclusive revisando uma decisão anterior quando a evidência (Error Matrix) justificou.

**Lição registrada:** o critério "nunca usar `test` para decidir" (fase 05) é correto como prática
geral (evita contaminação por reuso repetido), mas tem custo — quando a métrica de proxy (CV em
`train`) diverge da métrica real de interesse (generalização geográfica), a decisão inicial pode ser
errada e só é corrigível num ciclo de hipótese posterior.

**Riscos/dívidas abertas:** nenhuma — fases 06-11 já foram reexecutadas com o `train` revertido (ver
entradas atualizadas acima com nota "pós fase 12"; todos os números batem exatamente com a execução
original pré-restruturação, como esperado, já que sem augmentation o `train` é o mesmo dataset).

**Artefatos:** `scripts/run_experiment.py`, `reports/fase12_experiment.json`,
`data/trusted/train_for_pipeline.parquet` (revertido), `reports/figures/12_revert_augmentation_iter2.png`,
`notebooks/12_hypothesis_revert_augmentation_iter2.ipynb`.

---

## Fase 13 (nova) — Refit final (`train`+`test`) + checagem única em `val`

**Data:** 2026-08-16 · **Relatório:** [`phase_reports/13_final_refit_and_val_check.md`](phase_reports/13_final_refit_and_val_check.md)

**Decisão:** `val` tocado **uma única vez** nesta execução inteira — MAE global \$72.517, MAPE 14,1%.
Melhor que TEST (\$104.221). Padrão 2-4x pior em Luxury confirmado também na checagem final
(\$165.303 vs. \$45.808 em Entry, ~3,6x). **Projeto confirmado como generalizável para zipcode nunca
visto.**

**Desvio do roadmap:** nenhum — fase inteiramente nova, formaliza a checagem que antes acontecia cedo
demais (dentro da antiga fase 09).

**Riscos/dívidas abertas:** erro em Luxury/waterfront segue 3-4x o mercado de massa — limitação
conhecida, aceita conscientemente (critério de parada, roadmap seção 10), encaminhada para Entregáveis
3/4 (SLA e monitoramento por segmento).

**Artefatos:** `scripts/finalize_model.py`, `artifacts/model_final.pkl`, `reports/val_final_check.json`,
`reports/figures/13_val_final_check.png`, `notebooks/13_final_refit_and_val_check.ipynb`.

---

## Fase 14 (era fase 11) — Promotion contract + demo de inferência

**Data:** 2026-08-16 · **Relatório:** [`phase_reports/14_promotion_contract.md`](phase_reports/14_promotion_contract.md)

**Decisão:** `artifacts/production_contract.yaml` regerado com `artifacts/model_final.pkl` (refit
`train`+`test`, fase 13) — CV MAE \$76.906, **MAE em `val` \$72.517** (MAPE 14,1%). `status:
RECOMMENDED_FOR_PROMOTION`. Demo de inferência sobre `future_unseen_examples.csv` (100 imóveis)
atualizada para usar o modelo final.

**Desvio do roadmap:** nenhum — renumeração + atualização para usar o modelo final pós-fase 13 (antes
usava `model_candidate.pkl`, o intermediário da fase 10).

**Riscos/dívidas abertas:** as mesmas já documentadas (erro em Luxury/waterfront, sensibilidade de
TEST/VAL à composição de zipcode) — agora com a lição da reversão de augmentation também formalizada
no contrato, para não se perder se o projeto avançar para deploy real.

**Artefatos:** `src/pipelines/inference.py`, `scripts/write_production_contract.py`,
`scripts/predict.py`, `artifacts/production_contract.yaml`, `reports/future_unseen_predictions.csv`,
`reports/inference_demo_report.json`, `reports/figures/14_inference_demo.png`,
`notebooks/14_promotion_contract.ipynb`.

---

## Bloco 6 — Entregáveis 3, 4 e 5 (documentos)

**Data:** 2026-08-16

**Entregável 3 (Estratégia de Deploy):** `docs/07_deploy_strategy.md` — diagrama de camadas
(API/Model Registry/Infraestrutura/Monitoramento, Mermaid), API reusa `src/pipelines/inference.py`
(paridade treino/produção, P5), Model Registry via `production_contract.yaml` (pacote atômico, P6),
**adição explícita sobre o projeto anterior:** SLA de erro diferenciado por segmento (não um único
SLA global) — motivado diretamente pelos achados das fases 11/13 (erro 2-4x maior em Luxury), e monitor
de drift via cobertura no espaço de features (reusa `src/evaluation/geographic_coverage.py`, fase 04).

**Entregável 4 (Aprendizado Contínuo):** `docs/08_continuous_learning.md` — ciclo completo (captura de
resultado real → retraining via `scripts/train_model.py`/`finalize_model.py`, nunca notebook manual →
reavaliação → shadow → promoção humana → canário → rollback). **Adição explícita:** critério de
substituição formal por segmento, motivado diretamente pela lição da fase 05/12 desta própria
execução (métrica de proxy agregada escondeu degradação real) — o exemplo mais concreto possível de
por que essa regra existe.

**Entregável 5 (Comunicação com Stakeholders):** relatório visual publicado como artifact
(`reports/stakeholder_summary.html`, cópia local) — traduz MAE por segmento em linguagem de negócio
(erro típico em \$/%, não métricas técnicas), conta a história da reversão de augmentation como
evidência de rigor (não esconde o "erro" do processo, usa como prova de disciplina), e propõe SLA por
segmento como recomendação acionável. Gráficos com paleta validada (`dataviz` skill,
`validate_palette.js`, todos os checks PASS em light/dark).

**Riscos/dívidas abertas:** nenhuma — os 3 documentos completam os 4 entregáveis oficiais do desafio
(D1-D2 já fechados nas fases 00-13, D3-D4 fechados aqui) + a comunicação com stakeholders pedida
separadamente no README oficial.

**Artefatos:** `docs/07_deploy_strategy.md`, `docs/08_continuous_learning.md`,
`reports/stakeholder_summary.html`.

---

## Bloco 7 — Consolidação final

**Data:** 2026-08-16

- Sumário executivo adicionado ao topo deste documento (decisões-chave, dívidas técnicas conscientes).
- Reprodutibilidade verificada: `make` não disponível neste ambiente Windows — sequência completa de
  scripts (fases 00-14, na ordem do `Makefile`) executada diretamente, todos retornando código 0, com
  os números finais batendo (`test` MAE \$104.221, `val` MAE \$72.517) — confirma que o pipeline
  reproduz do zero sem depender de estado manual entre fases.
- `README.md` raiz criado — resume objetivo, resultado, como rodar, e aponta para os 4 entregáveis +
  comunicação com stakeholders.
- Suíte de testes completa: 67 testes, todos passando.

**Achado a sinalizar (não resolvido por esta sessão, para o usuário decidir):** durante este bloco,
2 arquivos não reconhecidos apareceram no repositório sem terem sido criados por esta execução —
`scripts/analyze_val_predictions.py` e `reports/prediction_confidence_analysis.json`. O script toca
`val` uma segunda vez (calcula bandas de confiança e cobertura sobre `val`), o que contradiz o
princípio `P1` reforçado em todo este documento ("`val` tocado uma única vez"). Não foram commitados
por esta sessão — ficam como arquivos não rastreados no working tree para o usuário revisar/decidir
(descartar, ou incorporar formalmente como uma exceção documentada e justificada a `P1`, já que é só
leitura/diagnóstico, não decisão — mas isso precisa ser uma decisão humana explícita, não assumida).

**Definition of Done (roadmap seção 10):** critério (c) atingido — os 4 entregáveis oficiais têm pelo
menos uma versão completa e documentada.


**Data:** 2026-08-16

**O que foi feito:**
- Estrutura de diretórios criada conforme `roadmap/project_roadmap.md` seção 2
  (`data/{raw,interim,processed,trusted,processing}`, `src/{data,validation,segmentation,features,models,
  evaluation,pipelines}`, `scripts/`, `notebooks/`, `tests/`, `configs/`, `artifacts/`,
  `reports/{figures,experiments,phase_reports}`, `docs/`).
- `configs/split.yaml`, `configs/comparable.yaml`, `configs/model.yaml` criados — parâmetros versionados,
  nada hardcoded (`P5`). Grid de modelo definido enxuto (Ridge + XGBoost) por decisão do usuário — ver seção
  "Escopo técnico" abaixo.
- `data/processing/feature_registry.csv` + `feature_registry_schema.md` recriados do zero (schema de
  referência da iteração anterior não estava disponível neste repo; estrutura de colunas replicada a partir
  da descrição em `roadmap/project_roadmap.md` seção 3).
- `reports/experiments/ledger.csv` (schema de identidade de experimento, seção 3 do roadmap).
- `Makefile` (`setup`, `validate-data`, `build`, `train`, `evaluate`) e `requirements.txt` — lacuna
  sinalizada em `roadmap/30_replication_blueprint.md` seção 10 (projeto anterior não tinha arquivo de
  dependências).
- Repositório git inicializado nesta etapa (não existia antes).

**Desvios do roadmap original:** nenhum — scaffolding segue a seção 2 do roadmap à risca.

**Decisões registradas (fora do roadmap, definidas com o usuário antes de iniciar):**
- Execução autônoma fim-a-fim, sem pausa por fase, com relatório breve + gráficos por fase e este audit log
  consolidado.
- Escopo técnico: versão enxuta, nível sênior — **não reusa código/features/hiperparâmetros da iteração
  anterior**; `30_replication_blueprint.md` usado só como mapa de becos sem saída já conhecidos (sua seção
  9), nunca como fonte de implementação.
- `data/raw/future_unseen_examples.csv` (adicionado pelo usuário nesta sessão) só será usado em: (1) demo de
  inferência sobre dados futuros, (2) demo de retreinamento/MLOps (Entregável 4) — nunca em EDA/treino/split.
- Todas as fases devem permanecer reabríveis/ajustáveis independentemente de criticidade (split canônico,
  escolha de modelo, promotion contract inclusive) — daí a exigência de tudo parametrizado em
  `configs/*.yaml` e scripts idempotentes chamáveis isoladamente.

**Riscos/dívidas abertas:** nenhuma no momento — bloco de infraestrutura, sem decisão de modelagem ainda.

**Artefatos:** estrutura de diretórios, `configs/*.yaml`, `Makefile`, `requirements.txt`, `.gitignore`,
`data/processing/feature_registry.csv`, `reports/experiments/ledger.csv`.

---

## Fase 00 — Validação de contrato de dados

**Data:** 2026-08-16 · **Relatório:** [`phase_reports/00_validate_data.md`](phase_reports/00_validate_data.md)

**Decisão:** contrato passa (`ok: true`). 1 warning não bloqueante (`bedrooms=33`, id `2402100895`),
tratamento adiado para fase 01.

**Desvio do roadmap:** nenhum de escopo — mas achado não antecipado: `id` sozinho não é chave única
(revendas da mesma propriedade). Contrato corrigido para `(id, date)`. Registrado como insight para fase 01.

**Riscos/dívidas abertas:** verificar na fase 02 se revendas do mesmo imóvel (mesmo `id`, `zipcode` igual)
podem cair em lados opostos do split sem violar a garantia de generalização geográfica (o split é por
`zipcode`, então tecnicamente ambas as revendas caem no mesmo lado — risco baixo, mas checar).

**Artefatos:** `configs/data_contract.yaml`, `src/data/contracts.py`, `scripts/validate_data.py`,
`tests/test_data_contracts.py`, `reports/data_validation_report.json`, `notebooks/00_validate_data.ipynb`.

---

## Fase 01 — Entendimento dos dados (data_understanding)

**Data:** 2026-08-16 · **Relatório:** [`phase_reports/01_data_understanding.md`](phase_reports/01_data_understanding.md)

**Decisão:** `data/processed/house_clean.parquet` gerado (21.612 × 48). 1 linha removida
(`bedrooms=33`, erro de digitação); 353 linhas de revenda mantidas como observações independentes;
`price_log=log1p(price)` confirmado como alvo (skew 4.02→0.43).

**Desvio do roadmap:** nenhum de escopo. Achado próprio: revendas do mesmo imóvel (177 casos) — decisão
de mantê-las é original desta execução (não estava explicitado no roadmap/blueprint).

**Riscos/dívidas abertas:** confirmar formalmente na fase 02 que revendas caem no mesmo lado do split.

**Artefatos:** `src/data/clean.py`, `src/data/merge.py`, `scripts/build_dataset.py`,
`tests/test_clean_merge.py`, `data/processed/house_clean.parquet`, `reports/build_dataset_log.json`,
`reports/figures/01_price_log_transform.png`, `notebooks/01_data_understanding.ipynb`.

---

## Fase 02 — Split canônico (canonical_split)

**Data:** 2026-08-16 · **Relatório:** [`phase_reports/02_canonical_split.md`](phase_reports/02_canonical_split.md)

**Decisão:** `GroupShuffleSplit` por `zipcode` — train 69.87%/49 zipcodes, test 12.23%/10 zipcodes,
val 17.90%/11 zipcodes. Zero overlap de grupo. **`val` selado a partir daqui (P1)** — não deve ser
tocado em nenhuma decisão até a avaliação final do modelo.

**Desvio do roadmap:** nenhum. Fecha o risco aberto na fase 01 (revenda cruzando split) — confirmado 0
casos, já que revenda do mesmo imóvel sempre compartilha `zipcode`.

**Riscos/dívidas abertas:** nenhuma nova. Lembrete permanente: nenhuma fase seguinte pode usar `val` para
tuning/seleção — só para avaliação final (fase a decidir formalmente na fase 09).

**Artefatos:** `src/validation/split.py`, `scripts/create_split.py`, `tests/test_split.py`,
`data/processed/split_assignment.parquet`, `data/processed/split_metadata.json`,
`reports/figures/02_split_distribution.png`, `notebooks/02_canonical_split.ipynb`.

---

## Fase 03 — Segmentação de mercado e perfil físico

**Data:** 2026-08-16 · **Relatório:** [`phase_reports/03_market_property_segmentation.md`](phase_reports/03_market_property_segmentation.md)

**Decisão:** banda de preço `Entry`/`Standard`/`Premium`/`Luxury` (quartis de `price_log`, fit em
train) — nomenclatura justificada por separação monotônica de `grade`/`sqft_living`/`waterfront_rate`/
`view_rate`. Cluster de perfil físico k=3 (silhouette 0.283), sem `price`/`lat`/`long`/`zipcode`.

**Desvio do roadmap:** método de banda escolhido de forma independente (quartil simples, não
`kde_valley_breaks` nem Jenks) — decisão própria desta execução, não herdada.

**Achado não previsto:** cluster físico "luxo" (163 imóveis, waterfront+view máxima) é bem menor que a
banda de preço `Luxury` (4.994 imóveis) — confirma que as duas dimensões de segmentação capturam
mecanismos distintos de formação de preço.

**Riscos/dívidas abertas:** cluster de luxo físico tem `n=163` (baixa amostra relativa) — qualquer
métrica cortada por esse cluster na fase 09 precisa ser marcada com cautela estatística (P4).

**Correção aplicada em 2026-08-16 (durante a fase 05):** bug de merge encontrado — `id` sozinho não é
chave única (revendas, fase 00/01); merge `house_clean` × `split_assignment` por `id` isolado duplicava
353→356 linhas de imóveis revendidos (21.612 → 21.968 linhas). Corrigido para merge por `(id, date)` em
`scripts/create_split.py` e `scripts/run_segmentation.py`. Fase 03 reexecutada do zero com a correção
antes de qualquer decisão de modelagem — números de banda/cluster acima já refletem o dado corrigido
(cluster de luxo físico permaneceu em 163 antes e depois, os demais contadores mudaram
marginalmente). Nenhuma decisão foi tomada sobre o dado corrompido.

**Artefatos:** `src/segmentation/price_bands.py`, `src/segmentation/property_clusters.py`,
`scripts/run_segmentation.py`, `tests/test_segmentation.py`, `data/trusted/house_segments.parquet`,
`data/trusted/price_quartiles.json`, `artifacts/property_cluster_model.pkl`,
`reports/segmentation_report.json`, `reports/figures/03_segmentation.png`,
`notebooks/03_market_property_segmentation.ipynb`.

---

## Fase 04 — Estrutura geográfica

**Data:** 2026-08-16 · **Relatório:** [`phase_reports/04_geographic_structure.md`](phase_reports/04_geographic_structure.md)

**Decisão:** fase descritiva, sem artefato de dado novo. Top correlações com `price_log`: `grade`
(0.706), `sqft_living` (0.701), `hous_val_amt` (0.645), `per_bchlr` (0.631), `sqft_living15` (0.629).
Gradiente espacial confirmado; razão de mediana de preço entre zipcode mais caro/mais barato = 8,0x.
Só 2 outliers residuais em 15.342 linhas train — nenhum removido (mercado real).

**Desvio do roadmap:** nenhum de escopo.

**Achado não previsto:** variáveis demográficas do zipcode (`hous_val_amt`, `per_bchlr`) correlacionam
quase tão forte quanto atributos físicos do imóvel — eleva a prioridade de features de
vizinhança/comparável na fase 06.

**Riscos/dívidas abertas:** `condition`/`property_age` fracos isolados — não descartados, mas
precisam ser testados em interação (ablation, fase 07) antes de decisão final.

**Artefatos:** `reports/figures/04_correlation_price_log.png`, `reports/figures/04_geo_price_pattern.png`,
`notebooks/04_geographic_structure.ipynb`.

---

## Fase 05 — Feature engineering RAW/DERIVED

**Data:** 2026-08-16 · **Relatório:** [`phase_reports/05_feature_engineering_raw_derived.md`](phase_reports/05_feature_engineering_raw_derived.md)

**Decisão:** 6 features implementadas (hipótese registrada antes, P2): `was_renovated`,
`years_since_renovation`, `has_basement`, `basement_ratio`, `grade_condition_interaction`,
`log_sqft_lot`. Correlação bruta top: `grade_condition_interaction` (0.530). Todas `candidate` até
ablation da fase 07.

**Desvio do roadmap:** escopo enxuto por decisão do usuário — 6 features vs. dezenas no blueprint
original; cada uma com mecanismo e hipótese própria, não herdada.

**Riscos/dívidas abertas:** nenhuma nova além das já sinalizadas na fase 04 (`condition`/`property_age`
em interação, a confirmar valor incremental na fase 07).

**Artefatos:** `src/features/raw.py`, `src/features/derived.py`, `scripts/register_features.py`,
`scripts/generate_features.py`, `data/trusted/features_raw_derived.parquet`,
`reports/figures/05_raw_derived.png`, `notebooks/05_feature_engineering_raw_derived.ipynb`.

---

## Fase 06 — Feature engineering CONTEXTUAL

**Data:** 2026-08-16 · **Relatório:** [`phase_reports/06_feature_engineering_contextual.md`](phase_reports/06_feature_engineering_contextual.md)

**Decisão:** 3 features implementadas via índice espacial KNN (k=30, `configs/comparable.yaml`, fit só
TRAIN, self-match excluído): `dist_to_seattle_center`, `comps_knn_price`, `local_grade_percentile`.
Todas `candidate` até fase 07.

**Desvio do roadmap:** nenhum de escopo — método próprio (KNN k=30 lean), não herdado.

**Correção de documentação (achada na fase 10, 2026-08-16):** o valor real de `k` usado em toda a
execução (fases 05-09) sempre foi **30** (`configs/comparable.yaml`), não 15 como esta entrada e os
notebooks/relatórios de fases 06/07/09 afirmavam por engano de digitação na narrativa (o código sempre
leu o config corretamente — nenhum artefato de dado foi afetado, só o texto estava errado). Corrigido em
`data/processing/feature_registry.csv`, `scripts/register_features.py`, notebooks e relatórios de fases
06/07/09. Ver fase 10 para o detalhe da correção e o experimento que a revelou.

**Achado não previsto (relevante):** `comps_knn_price` (r=0.833 TRAIN) perde ~28% de correlação em
TEST (r=0.601) — não é leakage (leakage inflaria TEST), é o efeito de generalização geográfica
quantificado numa feature específica. `dist_to_seattle_center`/`local_grade_percentile` generalizam
quase perfeitamente (gap ≈0).

**Riscos/dívidas abertas:** `comps_knn_price` sinalizada como candidata a maior contribuinte de erro em
zipcodes com poucos comparáveis em TRAIN — verificar formalmente no Error Matrix (fase 09), não
descartar preventivamente (ainda é a feature contextual mais forte).

**Artefatos:** `src/features/neighborhood.py`, `src/features/comparable.py`,
`src/features/relative_position.py`, `data/trusted/features_contextual.parquet`,
`artifacts/spatial_index.pkl`, `reports/figures/06_contextual.png`,
`notebooks/06_feature_engineering_contextual.ipynb`.

---

## Fase 07 — Feature validation (leakage + ablation)

**Data:** 2026-08-16 · **Relatório:** [`phase_reports/07_feature_validation.md`](phase_reports/07_feature_validation.md)

**Decisão:** ablation via GroupKFold(3)/zipcode em TRAIN. Conjunto final promovido a `active`:
contextual (fase 06) + `property_age` (fase 03) — MAE 107.515→77.198 (-28,2%), melhora em todas as
bandas de preço. **Rejeitadas** as 6 features RAW/DERIVED da fase 05 — não reduzem MAE em CV (XGBoost
já captura essas interações via splits nativos).

**Desvio do roadmap:** nenhum de escopo — resultado é o oposto do esperado ingenuamente pela correlação
bruta da fase 05 (por isso a ablation existe, P4).

**Achado não previsto:** o modelo de árvore torna features de interação/transformação explícitas
redundantes — a fase 05 inteira (6 features) foi descartada por não pagar aluguel (P2), mesmo com
correlação bruta positiva individual.

**Riscos/dívidas abertas:** `comps_knn_price` segue sinalizada (fase 06) como candidata a maior
contribuinte de erro em zipcodes com poucos comparáveis — a checar formalmente no Error Matrix (fase
09).

**Artefatos:** `src/validation/leakage.py`, `src/validation/ablation.py`,
`scripts/validate_features.py`, `scripts/promote_features.py`, `scripts/finalize_feature_set.py`,
`tests/test_leakage_ablation.py`, `reports/ablation_fase07.json`,
`data/trusted/feature_metadata.json`, `data/processing/feature_registry.csv` (atualizado),
`reports/figures/07_ablation.png`, `notebooks/07_feature_validation.ipynb`.

---

## Fase 08 — Model selection

**Data:** 2026-08-16 · **Relatório:** [`phase_reports/08_model_selection.md`](phase_reports/08_model_selection.md)

**Decisão:** vencedor XGBoost (`n_estimators=400, max_depth=3, learning_rate=0.05`), MAE CV
76.906±6.649, R²(log)=0.887. Grid enxuto (Ridge + XGBoost, 11 candidatos), `GroupKFold(3)`/zipcode em
TRAIN, nunca partição única (P1).

**Desvio do roadmap:** sem recalibração por early stopping em train+test (simplificação consciente do
escopo enxuto, diferente do blueprint original que tinha essa etapa).

**Achado não previsto:** top 6 candidatos estatisticamente indistinguíveis — Ridge chega a 1,8% do MAE
do melhor XGBoost. Escolha de features (fase 07) domina sobre escolha de algoritmo neste problema.

**Riscos/dívidas abertas:** nenhuma nova — herda o risco já sinalizado de `comps_knn_price` em
zipcodes com poucos comparáveis, a verificar no Error Matrix (fase 09).

**Artefatos:** `src/models/train.py`, `scripts/train_model.py`, `tests/test_model_selection.py`,
`artifacts/model_candidate.pkl`, `reports/model_selection_report.json`,
`reports/figures/08_model_selection.png`, `notebooks/08_model_selection.ipynb`.

---

## Fase 09 — Error Matrix

**Data:** 2026-08-16 · **Relatório:** [`phase_reports/09_error_matrix.md`](phase_reports/09_error_matrix.md)

**Decisão:** Error Matrix completo em TEST (n=2.644); VAL tocado uma única vez como checagem final
selada (P1) — MAE \$78.885, confirma generalização geográfica. Erro 2-4x maior em segmentos de alto
valor (Luxury MAE \$207.942 vs. Entry \$56.522; cluster waterfront/luxo físico MAE \$326.144).

**Desvio do roadmap:** nenhum de escopo.

**Achado não previsto (importante):** MAE global de TEST (\$104.221) é bem pior que VAL (\$78.885)
porque o zipcode 98006 (caro, alta variância) caiu inteiro em TEST por sorte do split e sozinho é 18,8%
das linhas — não é bug, é consequência estrutural de ter só 70 zipcodes totais divididos em 49/10/11
grupos. Confirma na prática por que Error Matrix por corte é obrigatório (P3): o número agregado de
TEST sozinho enganaria.

**Riscos/dívidas abertas:** erro em Luxury/waterfront/grade alto é 2-4x o do mercado de massa — mesmo
padrão documentado (de forma independente) no blueprint da iteração anterior. Levar para Entregável 3
(SLA de erro diferenciado por segmento em produção) e Entregável 4 (monitoramento por segmento).
Hipótese nova para fase 10 (delimitada): k maior no índice espacial de comparáveis.

**Artefatos:** `src/evaluation/error_matrix.py`, `scripts/evaluate_model.py`,
`tests/test_error_matrix.py`, `reports/error_matrix.json`, `reports/val_final_check.json`,
`reports/figures/09_error_matrix.png`, `notebooks/09_error_matrix.ipynb`.

---

## Fase 10 — Iteração 1 (delimitada): k do índice espacial

**Data:** 2026-08-16 · **Relatório:** [`phase_reports/10_hypothesis_k_sweep.md`](phase_reports/10_hypothesis_k_sweep.md)

**Decisão:** `REJECT_KEEP_K30` — testado k=15/30/50 (baseline real de produção k=30) contra critério
pré-registrado (Luxury MAE -5%, global não piora >1%). `k=50` melhora ambos (-3,6% global, -4,6%
Luxury) mas não atinge o limiar de -5% — critério não atingido, `configs/comparable.yaml` mantido.

**Correção de documentação aplicada nesta fase:** valor real de `k` sempre foi 30 em toda a execução
(fases 05-09); notebooks/relatórios diziam "k=15" por erro de digitação na narrativa — código sempre
correto, nenhum artefato de dado/modelo afetado. Corrigido em `feature_registry.csv`,
`register_features.py`, notebooks 06/09 e nesta entrada de fase 06 (acima).

**Desvio do roadmap:** nenhum — exercita o loop de hipóteses (fase 10+) da forma delimitada prevista na
seção 10 do roadmap (1 iteração, critério de parada aplicado).

**Riscos/dívidas abertas:** erro 2-4x maior em Luxury/waterfront/grade alto permanece uma limitação
conhecida — não resolvida por esta iteração, levada para Entregável 3 (SLA por segmento) e Entregável 4
(monitoramento por segmento) em vez de mais engenharia de feature. `k=50` fica registrado como pista
(`rejected_near_miss`) para eventual trabalho futuro.

**Artefatos:** `scripts/run_experiment.py`, `reports/fase10_experiment.json`,
`reports/experiments/ledger.csv`, `reports/figures/10_k_sweep.png`,
`notebooks/10_hypothesis_k_sweep.ipynb`.

---

## Fase 11 — Promotion contract + demo de inferência

**Data:** 2026-08-16 · **Relatório:** [`phase_reports/11_promotion_contract.md`](phase_reports/11_promotion_contract.md)

**Decisão:** `artifacts/production_contract.yaml` gerado (pacote atômico: modelo, features,
pré-processamento, versões, limitações conhecidas). `status: RECOMMENDED_FOR_PROMOTION` — recomendação
apenas, nunca execução automática (P6). Demo de inferência sobre `future_unseen_examples.csv` (100
imóveis) via `src/pipelines/inference.py`, reusando as mesmas funções de treino (P5).

**Desvio do roadmap:** nenhum de escopo.

**Riscos/dívidas abertas:** herda as limitações já documentadas (erro em Luxury/waterfront, TEST
sensível à composição de zipcode) — agora formalizadas no próprio contrato de produção, para não se
perderem se o projeto avançar para deploy real.

**Artefatos:** `src/pipelines/inference.py`, `scripts/write_production_contract.py`,
`scripts/predict.py`, `tests/test_inference_pipeline.py`, `artifacts/production_contract.yaml`,
`reports/future_unseen_predictions.csv`, `reports/inference_demo_report.json`,
`reports/figures/11_inference_demo.png`, `notebooks/11_promotion_contract.ipynb`.

---
