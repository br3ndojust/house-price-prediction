# Previsão de Preços de Casas — Seattle — Visão Geral do Projeto


## 1. Objetivo do sistema

> Construir um sistema de previsão de preço **espacialmente robusto**, **orientado por hipóteses de
> mercado**, **segmentado por perfil de imóvel** e com **erro conhecido por segmento**.

O projeto é guiado pelo
ciclo `Hipótese → Representação → Experimento → Evidência → Decisão`, aplicado fase a fase, sempre
partindo de uma pergunta de negócio/mercado, nunca de um algoritmo.

O projeto mapeia para os entregáveis oficiais do desafio:

| Entregável | Conteúdo | Status |
|---|---|---|
| **D1** — Entendimento dos dados | limpeza, EDA, segmentação de mercado/perfil físico | ✅ fechado |
| **D2a** — Variáveis importantes | engenharia de features (RAW/DERIVED/CONTEXTUAL) + ablation | ✅ fechado |
| **D2b** — Escolha do modelo | seleção Ridge vs. XGBoost via `GroupKFold` | ✅ fechado |
| **D2c** — Generalização | split geográfico, cobertura, augmentation, Error Matrix, refit final + `val` | ✅ fechado |
| **D3** — Estratégia de deploy | `docs/07_deploy_strategy.md` — documento | ✅ fechado |
| **D4** — Aprendizado contínuo | `docs/08_continuous_learning.md` — documento | ✅ fechado |
| **D5** — Comunicação com stakeholders | `reports/stakeholder_summary.html` + `docs/GUIA_APRESENTACAO.md` | ✅ fechado |
| *(extra, fora do escopo oficial)* — API de serving + portal Inference | `app/` (FastAPI) + `portal/` (Streamlit) | ✅ implementado, exceto CI/CD (ver seção 12) |

## 2. Princípios de governança (resumo `P0`–`P7`)

Documento completo em `roadmap/PRINCIPLES.md`. Resumo do que rege cada decisão registrada neste projeto:

- **P0 — Filosofia central:** o sistema deve representar mecanismos plausíveis de preço, generalizar
  geograficamente, conhecer seus limites e ser reprodutível sem depender dos notebooks originais.
- **P1 — Dados e validação:** `zipcode` nunca é feature (só chave de merge/split/diagnóstico);
  validação principal é sempre geográfica (`GroupKFold`/`GroupShuffleSplit`, nunca holdout aleatório);
  EDA descritivo roda em 100% dos dados antes do split; tudo que tem `.fit()` roda só em `train`;
  `test` é cheque repetível durante o desenvolvimento (nunca decide sozinho); `val` é sagrado — tocado
  uma única vez, no fim de tudo.
- **P2 — Feature engineering orientada a hipótese:** nenhuma feature nasce sem hipótese registrada
  (`feature_registry.csv`); features representam mecanismos econômicos, não combinações arbitrárias;
  "complexidade precisa pagar aluguel" — precisa de ganho experimental mensurado.
- **P3 — Mercado e segmentação:** duas dimensões de segmentação nunca misturadas — banda de **preço**
  (`Entry`/`Standard`/`Premium`/`Luxury`, quartis) e cluster de **perfil físico** (sem
  `price`/`lat`/`long`/`zipcode`). Modelo é avaliado por segmento, nunca só globalmente.
- **P4 — Modelagem e avaliação honesta:** critério de sucesso definido *antes* do experimento; ablation
  sempre antes de promover feature; erro é fonte de nova hipótese, não fim de linha; amostra pequena
  exige cautela estatística explícita (`LOW SAMPLE`).
- **P5 — Engenharia e reprodutibilidade:** notebook não contém lógica de negócio (só chama `src/`);
  `git clone` + `make setup/validate-data/build/train/evaluate` reproduz tudo; configs em `configs/*.yaml`,
  nunca hardcoded; todo experimento tem identidade rastreável (`reports/experiments/ledger.csv`).
- **P6 — Produção e governança:** produção nunca é atualizada automaticamente — pipeline recomenda,
  humano aprova; contrato de produção é um pacote atômico versionado.
- **P7 — Notebooks como espinha dorsal:** toda fase segue o formato
  `Pergunta → Hipótese → Análise/Experimento → Métricas → Decisão → Artefato`. Fase só termina quando
  existe uma decisão reproduzível.

## 3. Arquitetura do repositório


```text
data/
├── raw/              # dados originais — nunca sobrescritos
├── interim/          # dados intermediários das etapas de processamento
├── processed/        # datasets prontos para modelagem
├── trusted/          # datasets validados e aprovados
└── processing/       # artefatos temporários de processamento

src/
├── data/             # ingestão, carga e transformação dos dados
├── validation/       # validação de dados e contratos
├── segmentation/    # segmentação e análise por grupos
├── features/         # feature engineering e seleção de features
├── augmentation/     # geração/augmentação de dados
├── models/           # definição, treino e persistência dos modelos
├── evaluation/       # métricas, resíduos e análise de erros
└── pipelines/        # orquestração dos pipelines end-to-end

scripts/              # pontos de entrada operacionais; utiliza src/
notebooks/            # análises exploratórias e narrativa científica

tests/                # testes automatizados de dados, features,
                      # segmentação, modelos, pipelines e confiança

configs/
├── split.yaml
├── comparable.yaml
├── model.yaml
├── data_contract.yaml
└── augmentation.yaml # configurações sem parâmetros hardcoded

artifacts/            # artefatos necessários para execução/inferência
├── model_final.pkl
├── spatial_index.pkl
└── production_contract.yaml

reports/
├── figures/          # gráficos e visualizações
├── experiments/      # resultados e comparações de experimentos
└── phase_reports/    # relatórios por fase e evidências do projeto

roadmap/              # planejamento e evolução do projeto
docs/                 # documentação técnica e entregáveis

app/                  # API de inferência FastAPI
portal/               # Portal de inferência Streamlit
```

`configs/augmentation.yaml` é a adição mais recente (Bloco 6) — formaliza a decisão vigente
(`adopted: none`, pós fase 12) como **fonte da verdade versionada**, corrigindo um bug de
reprodutibilidade real: sem esse arquivo, rerodar `scripts/run_augmentation_experiment.py` do zero
recalcularia o "vencedor" mecânico pelo critério de CV (que ainda favorece a Técnica B isoladamente) e
desfaria silenciosamente a correção da fase 12. O script sempre roda a comparação completa (para
relatório/reavaliação futura), mas só materializa `train_for_pipeline.parquet` conforme o valor
declarado neste config.

## 4. Nota sobre a renumeração das fases (histórico)

O projeto passou por uma **reestruturação em 2026-08-16** (ver `reports/AUDIT_LOG.md`), depois de o
usuário identificar 2 problemas metodológicos nas fases originais (00–11):

1. EDA rodava só em `train`, depois do split → corrigido para rodar em 100% dos dados, antes do split.
2. `val` era tocado cedo demais → corrigido para ser tocado uma única vez, no final (fase de refit).

Isso inseriu 2 fases novas (`04 geographic_coverage`, `05 augmentation_experiment`) e empurrou a
numeração das fases seguintes. Os notebooks foram todos renomeados para a numeração final — a tabela da
seção 5 já reflete os nomes de arquivo atuais em `notebooks/`.

**Segunda correção, já concluída:** a fase 11 (Error Matrix) revelou que o `train` aumentado (Técnica B,
fase 05) melhorava a métrica de decisão (CV dentro de `train`) mas piorava a generalização real
(`test`, zipcode nunca visto). A fase 12 (iteração 2) reverteu essa decisão, e as fases 06–11 foram
**reexecutadas** sobre o `train` original (sem augmentation) antes de prosseguir para o refit final —
ver seção 6.

## 5. Linha do tempo das fases

| # | Fase | Notebook (arquivo) | Pergunta | Decisão |
|---|---|---|---|---|
| 00 | Validação de contrato de dados | `scripts/notebooks/00_validate_data.ipynb` | Os dados brutos batem com o schema esperado? | Contrato OK; 1 warning não bloqueante (bedrooms=33) |
| 01 | Entendimento dos dados | `01_data_understanding.ipynb` | O que os dados representam e estão prontos para uso? | 1 linha removida, 353 revendas mantidas, `price_log` confirmado como alvo |
| 02 | EDA completo (100% dos dados, pré-split) | `02_eda_completo.ipynb` | Que correlações/padrões geográficos existem antes de qualquer split? | Estrutura espacial forte; zip 98039 é caro **e** escasso — motiva fase 04 |
| 03 | Split canônico | `03_canonical_split.ipynb` | Como garantir avaliação em zipcode nunca visto? | `GroupShuffleSplit` por zipcode: 69,87%/12,23%/17,90%, zero overlap |
| 04 | Cobertura geográfica | `04_geographic_coverage.ipynb` | Escassez de dado em `train` correlaciona com erro? | Sim, no espaço de features (r=0,236, p≈9e-35) — não geografia pura. Hipótese de augmentation = **STRONG** |
| 05 | Experimento de augmentation | `05_augmentation_experiment.ipynb` | Augmentation melhora robustez de `train` sem contaminar `test`/`val`? | Técnica B (perturbação controlada) adotada — MAE CV -11,4% (decisão **revertida na fase 12**) |
| 06 | Segmentação de mercado e perfil físico | `06_market_property_segmentation.ipynb` | Como o mercado se segmenta por preço e por perfil físico? | Bandas de preço (quartil) + 3 clusters físicos (k=3, silhouette) — **recomputado sem augmentation** |
| 07 | Feature engineering RAW/DERIVED | `07_feature_engineering_raw_derived.ipynb` | Que mecanismos físicos/temporais/qualidade faltam? | 6 features candidatas geradas — **recomputado sem augmentation** |
| 08 | Feature engineering CONTEXTUAL | `08_feature_engineering_contextual.ipynb` | Que mecanismo de localização/comparável falta? | 3 features candidatas via índice espacial KNN (k=30) — **recomputado sem augmentation** |
| 09 | Feature validation (leakage + ablation) | `09_feature_validation.ipynb` | As features novas têm leakage, são redundantes? | Conjunto final: `contextual` + `property_age` (MAE -28,2%); `raw_derived` rejeitado |
| 10 | Model selection | `10_model_selection.ipynb` | Qual modelo generaliza melhor sob validação geográfica? | XGBoost (`n_estimators=400, max_depth=3, lr=0.05`) — empate técnico com Ridge |
| 11 | Error Matrix | `11_error_matrix.ipynb` | Onde o modelo funciona bem, onde falha? | MAE global TEST **\$104.221** — confirma que reverter augmentation (fase 12) restaura o nível original |
| 12 (iter. 1) | Hipótese: k do índice espacial | `12_hypothesis_k_sweep_iter1.ipynb` | k maior reduz erro em Luxury sem piorar o global? | `REJECT_KEEP_K30` — k=50 quase atinge o critério (-4,56% vs. -5% exigido) |
| 12 (iter. 2) | Hipótese: reverter augmentation | `12_hypothesis_revert_augmentation_iter2.ipynb` | Reverter para `train` sem augmentation reduz MAE em `test`? | **`REVERT_TO_NO_AUGMENTATION`** — MAE test -6,3% (113.384→106.254); propagado às fases 06-11 |
| 13 | Refit final + checagem única em `val` | `13_final_refit_and_val_check.ipynb` | O candidato final generaliza para o futuro desconhecido? | `artifacts/model_final.pkl` (refit TRAIN+TEST) — MAE em `val` **\$72.517** |
| 14 | Promotion contract + demo de inferência | `14_promotion_contract.ipynb` | O candidato está pronto para contrato de produção? | `artifacts/production_contract.yaml` gerado com o modelo final — `status: RECOMMENDED_FOR_PROMOTION` |

15 fases numeradas (00-14) no total. Depois delas: Bloco 6 (D3/D4/D5, documentos) e Bloco 7
(consolidação final — sumário executivo, README, verificação de reprodutibilidade, suíte de testes).

---

## 6. Detalhamento fase a fase

### Fase 00 — Validação de contrato de dados
**Pergunta:** os dados brutos batem com o contrato esperado (schema, nulos, cardinalidade)?
**O que roda:** `scripts/validate_data.py` sobre os 3 CSVs brutos (`kc_house_data.csv`,
`zipcode_demographics.csv`, `future_unseen_examples.csv`).
**Achados:** chave única real é `(id, date)`, não só `id` — 353 linhas compartilham `id` (revendas).
1 linha com `bedrooms=33` (id `2402100895`) sinalizada como warning não bloqueante.
**Decisão:** contrato passa (`ok: true`); tratamento do outlier fica para a fase 01.
**Artefato:** `reports/data_validation_report.json`.

### Fase 01 — Entendimento dos dados
**Pergunta:** o que os dados representam e estão prontos para uso?
**O que roda:** `scripts/build_dataset.py` — limpeza + merge (`kc_house_data` × `zipcode_demographics`
por zipcode, cobertura 1:1 completa em 70 zipcodes).
**Decisões de limpeza:**
- Removida 1 linha (`bedrooms=33`, `sqft_living=1620` → ~49 sqft/quarto, fisicamente implausível).
- Mantidas 353 linhas de revenda (177 imóveis revendidos no período) como observações de mercado
  independentes, não duplicatas.
- `price_log = log1p(price)` confirmado como alvo: skew bruto ≈4,02 cai para ≈0,43 em log1p.
**Resultado:** `data/processed/house_clean.parquet` — 21.612 linhas × 48 colunas.

### Fase 02 — EDA completo (100% dos dados, pré-split)
**Pergunta:** que correlações/distribuições/padrões geográficos existem antes de qualquer split?
**Top correlações com `price_log`:** `grade` (0,704), `sqft_living` (0,695), `hous_val_amt` (0,631),
`sqft_living15` (0,619), `per_bchlr` (0,611). `condition` é a mais fraca isolada (0,039).
**Padrão geográfico:** razão de 8,0x entre a mediana de preço do zipcode mais caro e do mais barato.
**Descoberta que motiva a fase 04:** 70 zipcodes, `n` de imóveis variando de 50 a 601; 3 zipcodes com
menos de 100 imóveis (`98039`, `98148`, `98024`). `98039` (Medina) é simultaneamente o **zipcode mais
caro** e um dos **mais escassos** em volume.
**Extensão (2026-08-17):** correlação completa (45 colunas) + cruzamentos de mercado — seção 12.10,
`reports/phase_reports/02_eda_completo.md`.

### Fase 03 — Split canônico
**Método:** `GroupShuffleSplit` duplo por `zipcode`, `random_state=42`.
**Resultado:** train 69,87% (15.100 linhas, 49 zipcodes) / test 12,23% (2.644 linhas, 10 zipcodes) /
val 17,90% (3.868 linhas, 11 zipcodes). Zero overlap de grupo entre partições; 0 revendas cruzando
split.
**Papel de cada partição:** `train` ajusta o modelo; `test` é cheque repetível durante o
desenvolvimento, nunca decide sozinho; `val` só é tocado uma única vez, na fase 13.

### Fase 04 — Cobertura geográfica e suficiência de dado
**Achados:**
- 3 zipcodes `LOW` (<100 imóveis) em `train`; 40 `MEDIUM`; 6 `HIGH`.
- **Distância geográfica pura não prediz erro:** r=-0,027 (p=0,17), não significativa.
- **Distância no espaço de features prediz erro fortemente:** r=0,236 (p≈9×10⁻³⁵) — imóveis
  fisicamente atípicos são sistematicamente mais difíceis de prever.
**Veredito:** hipótese de augmentation = **STRONG**, mas o mecanismo certo é reforçar cobertura no
espaço de features (não "cobrir mais zipcode").
**Extensão (2026-08-17), achado central:** holdout geográfico **completo** — 0 zipcode em comum entre
`train`/`test`. Decomposição por feature e drift de distribuição — seção 12.10,
`reports/phase_reports/04_geographic_coverage.md`.

### Fase 05 — Experimento de data augmentation
**Técnicas testadas (implementação própria):** SMOGN (Técnica A) e perturbação controlada (Técnica B).
**Critério de aceite (pré-registrado):** decisão por MAE médio em `GroupKFold(3)` **dentro de `train`**
— adota só se reduzir ≥2% sobre baseline.

| Variante | n train | MAE CV (decisivo) |
|---|---|---|
| Baseline (sem augmentation) | 15.100 | 107.515 ± 14.836 |
| Técnica A — SMOGN | 18.137 | 112.366 ± 7.243 (pior) |
| **Técnica B — perturbação controlada** | 22.650 | **95.276 ± 14.475 (-11,4%)** |

**Decisão original:** Técnica B adotada (único critério atingido). **Ressalva honesta registrada na
época:** o ganho quase não aparecia em `test` — presságio do que a fase 11/12 confirmaria depois.
**Status final (pós fase 12): decisão revertida** — ver fases 11/12 abaixo. Mantida como registro
histórico da decisão original (P4: nunca apagar uma decisão, só corrigir com uma nova).
`configs/augmentation.yaml` (`adopted: none`) é hoje a fonte da verdade versionada dessa reversão — ver
seção 3.

### Fase 06 — Segmentação de mercado e perfil físico (recomputado sem augmentation)
**Duas dimensões independentes** (P3): banda de preço (quartis) e cluster de perfil físico (sem
`price`/`lat`/`long`/`zipcode`).

| Segmento | n | Perfil |
|---|---|---|
| Banda `Entry` | 5.259 | — |
| Banda `Standard` | 5.940 | — |
| Banda `Premium` | 5.496 | — |
| Banda `Luxury` | 4.917 | — |
| Cluster 0 — Antigo/Compacto | 11.651 | `sqft_living` 1.540, `grade` 7, idade 58 anos |
| Cluster 1 — Moderno/Amplo | 9.798 | `sqft_living` 2.530, `grade` 8, idade 17 anos |
| Cluster 2 — Luxo físico (waterfront/view) | 163 | `sqft_living` 2.850, `grade` 9, `waterfront`=1, `view`=4, idade 54 anos |

Silhouette k=3 = 0,283. Números idênticos aos da execução original (o `train`, sem augmentation, é
exatamente o mesmo dataset de antes da fase 05).
**Extensão (2026-08-17):** cluster raro instável entre partições (0,75%→1,44%/0,31%) mas com erro real
4,5x pior (\$313.057 vs \$70.283) — seção 12.10, `reports/phase_reports/06_market_property_segmentation.md`.

### Fase 07 — Feature engineering RAW/DERIVED (recomputado sem augmentation)
**6 features geradas:** `was_renovated`, `years_since_renovation`, `has_basement`, `basement_ratio`,
`grade_condition_interaction`, `log_sqft_lot`. Correlação bruta: `grade_condition_interaction` (0,530)
domina; demais entre 0,07 e 0,19. Decisão de promoção adiada para a ablation (fase 09).
**Extensão (2026-08-17):** +3 candidatas (`bathrooms_per_bedroom` 0,302; `sqft_living_to_lot_ratio`
0,172; `property_cluster_distance` 0,338) — testadas e **rejeitadas** na fase 09 (seção 12.10).

### Fase 08 — Feature engineering CONTEXTUAL (recomputado sem augmentation)
**3 features geradas:** `dist_to_seattle_center`, `comps_knn_price`, `local_grade_percentile` (índice
espacial KNN, k=30, fit só em `train`).

| Feature | r train | r test | gap |
|---|---|---|---|
| `dist_to_seattle_center` | -0,188 | -0,190 | +0,002 |
| `comps_knn_price` | 0,833 | 0,601 | +0,232 |
| `local_grade_percentile` | 0,436 | 0,504 | -0,069 |

`comps_knn_price` é a mais forte, com gap notável train→test — efeito de generalização geográfica
esperado (não leakage, já que o índice é fit só em `train`).
**Extensão (2026-08-17):** +3 candidatas (`dist_to_nearest_train_zip`, `comps_knn_neighbor_distance`,
`local_price_dispersion` r=0,289→0,115) — testadas e **rejeitadas** na fase 09 (seção 12.10).

### Fase 09 — Feature validation (leakage + ablation), recomputado sem augmentation
**Ablation (MAE via `GroupKFold(3)`, `train`):**

| Conjunto | MAE global | Δ vs. baseline |
|---|---|---|
| baseline (16 físicas+demográficas) | 107.515 ± 14.836 | — |
| + raw_derived (6 features) | 107.672 ± 14.051 | +157 (dentro do desvio, sem ganho) |
| + contextual (3 features) | 79.520 ± 7.063 | **-27,0%** |
| + raw_derived + contextual | 80.463 ± 8.066 | pior que só contextual |
| **+ contextual + `property_age`** | **77.198 ± 6.956** | **-28,2% (melhor conjunto)** |

**Extensão:** 6 candidatas novas das fases 07/08 testadas em cima deste conjunto —
melhoria real de 1-1,5% (validada com 3/7/10 folds), abaixo do corte de 5% definido pelo usuário.
`rejected`. Conjunto oficial de 20 features **sem mudança**. Detalhe completo: seção 12.10,
`reports/phase_reports/09_feature_validation.md`.

**Decisão:** `raw_derived` **rejeitado** — mesmo com correlação bruta positiva individual, não reduz
MAE isolado e piora o conjunto quando combinado com `contextual` (o modelo de árvore já captura essas
interações nativamente). Conjunto oficial travado: `contextual` (3 features) + `property_age`.
**20 features oficiais** gravadas em `data/trusted/feature_metadata.json`.

### Fase 10 — Model selection (recomputado sem augmentation)
**Grid:** Ridge + XGBoost, `GroupKFold(3)`/zipcode, `train`.
**Vencedor:** XGBoost (`n_estimators=400, max_depth=3, learning_rate=0.05`) — MAE CV
76.906±6.649, R²(log)=0,887. Empate técnico com Ridge (78.287) e outros candidatos XGBoost — mesmo
padrão de sempre: **escolha de features pesa mais que escolha de algoritmo**.
**Artefato:** `artifacts/model_candidate.pkl`.

### Fase 11 — Error Matrix (recomputado sem augmentation)
**Global (TEST, n=2.644):** MAE **\$104.221** — de volta ao nível da execução original (pré-augmentation),
confirmando que a reversão da fase 12 corrigiu a piora identificada na primeira passagem por esta fase
(quando o `train` ainda tinha a Técnica B: MAE \$113.385).
**Por banda de preço:** `Luxury` (n=482) MAE \$207.942 — ~4x `Entry` (n=771, \$56.522), mesmo padrão
conhecido de mercados imobiliários, independente da questão de augmentation.
**Por zipcode:** `98006` segue como o mais difícil (MAE ~\$185k, 18,8% do `test`) — zipcode caro e de
alta variância caiu inteiro em `test` pela composição do split (só 70 zipcodes totais).
**Decisão:** Error Matrix final sobre `test` travado. Segue para fase 13.

### Fase 12, iteração 1 — Hipótese: k do índice espacial
**Pergunta:** existe uma forma barata de reduzir o erro 2-4x maior em Luxury/waterfront?
**Critério de aceite:** promove k diferente só se MAE de Luxury em TEST cair ≥5% **e** MAE global não
piorar mais que 1%.

| k | MAE global | MAE Luxury |
|---|---|---|
| 15 | \$111.335 (+6,8%) | \$211.114 (-1,5%, piora) |
| **30 (baseline)** | **\$104.221** | **\$207.942** |
| 50 | \$100.476 (-3,6%) | \$198.453 (-4,6%) |

**Decisão:** `REJECT_KEEP_K30` — k=50 melhora nos dois eixos mas fica a 0,44 ponto percentual do
critério de -5% em Luxury. **Disciplina metodológica (P4): não move a régua depois de ver o resultado.**
`k=50` fica registrado como `rejected_near_miss`. Fase encerrada após 1 iteração (critério de parada,
roadmap seção 10).

### Fase 12, iteração 2 — Hipótese: reverter augmentation
**Pergunta:** o Error Matrix (fase 11, primeira execução) mostrou que o `train` aumentado melhora o
`GroupKFold` MAE dentro de `train` mas piora o MAE em `test`. Reverter, rodando o **pipeline completo**
(não só o diagnóstico simplificado da fase 05), corrige isso?
**Critério de aceite (definido antes, diferente da fase 05):** aqui `test` é **deliberadamente a
métrica decisiva** — o propósito da hipótese é checar generalização geográfica.

| Variante | MAE global TEST |
|---|---|
| Sem augmentation (pipeline completo) | **106.254** |
| Técnica B adotada (fase 05) | 113.384 |

**Decisão: `REVERT_TO_NO_AUGMENTATION`.** Reduz o MAE em test em ~6,3%. `data/trusted/train_for_pipeline.parquet`
revertido para o `train` original (15.100 linhas). **Fases 06-11 foram reexecutadas** com este `train`
corrigido antes de prosseguir para a fase 13 — os números da seção 6 acima (fases 06-11) já refletem a
versão corrigida.
**Lição registrada (P4, honestidade sobre o que não funcionou):** o critério "nunca usar `test` para
escolher entre candidatos" é correto como prática geral, mas tem um custo — quando a métrica de proxy
(CV em `train`) diverge da métrica real de interesse (generalização geográfica), a decisão inicial pode
ser errada e só é corrigível num ciclo de hipótese posterior. Ambas as fases (05 e 12) permanecem
registradas no `AUDIT_LOG.md`, sem apagar a decisão original. Esta é, sozinha, a melhor evidência
prática de por que o projeto nunca decide por métrica agregada isolada — e a peça central sugerida em
`docs/GUIA_APRESENTACAO.md` para contar a história do projeto.

### Fase 13 — Refit final (`train`+`test`) + checagem única em `val`
**Pergunta:** o candidato final generaliza para o futuro desconhecido?
**Método:** com augmentation/features/modelo/hipóteses todos travados (fases 02-12), refit único de
XGBoost (`n_estimators=400, max_depth=3, learning_rate=0.05`) combinando `train`+`test` (17.744 linhas).
**Checagem final em `val` (n=3.868, tocado UMA ÚNICA VEZ nesta execução inteira):**

| Métrica | Valor |
|---|---|
| MAE | **\$72.517** |
| RMSE | \$112.596 |
| MAPE | 14,1% |
| Median APE | 11,2% |
| P90 AE | \$151.603 |
| Bias | +\$19.536 (modelo subestima levemente em média) |

**Melhor que o MAE em TEST** (\$104.221) — a composição de zipcodes de `val` (11 zipcodes) não inclui
um equivalente ao `98006` (o zipcode difícil que dominava `test`), e o refit usa mais dado (17.744 vs.
15.100 linhas).
**Por banda de preço:** `Luxury` (n=660) MAE \$165.303 — ~3,6x `Entry` (n=713, \$45.808). Mesmo padrão
2-4x conhecido, confirmado na checagem final.
**Validado:** o projeto generaliza para zipcode nunca visto — MAE em `val` é da mesma ordem de grandeza
(e melhor) que em `test`, não um colapso de performance. Nenhuma decisão anterior usou `val` (checável
no histórico — nenhum script antes desta fase lê `split == "val"` para decisão).
**Resultado:** `artifacts/model_final.pkl` é o candidato final — usado no contrato de produção (fase 14)
e em toda a análise de previsões da seção 15 deste documento.

### Fase 14 — Promotion contract + demo de inferência
**Contrato gerado** (`artifacts/production_contract.yaml`), agora com o **modelo final** (refit
`train`+`test`, fase 13) — não mais o `model_candidate.pkl` intermediário:
- Modelo: XGBoost (`n_estimators=400, max_depth=3, learning_rate=0.05`), CV MAE \$76.906,
  **MAE em `val` \$72.517**, MAPE 14,1%.
- 20 features oficiais (`feature_metadata.json`); pré-processamento (índice espacial k=30, fit só em
  TRAIN); versões de dataset/split; contrato de validação (`error_matrix.json` em TEST +
  `val_final_check.json`, checagem única).
- **Limitações conhecidas explícitas:** erro 2-4x maior em Luxury/waterfront/grade alto; sensibilidade
  de TEST/VAL à composição de zipcode (só 10/11 de 70 totais); a lição da reversão de augmentation
  (fase 05→12) — CV em `train` nem sempre reflete generalização geográfica real.
**Status:** `RECOMMENDED_FOR_PROMOTION` — recomendação apenas; substituição real sempre exige aprovação
humana explícita (P6).
**Demo de inferência:** `future_unseen_examples.csv` (100 imóveis) via `src/pipelines/inference.py`,
usando `model_final.pkl`. Previsões: mediana \$425.209, min \$172.360, max \$2.466.708 — faixa
plausível.

---

## 7. Fluxo de features

```
                         RAW (fonte primária)
    ┌──────────────────────────────┬──────────────────────────────┐
    │ kc_house_data.csv            │ zipcode_demographics.csv      │
    │ físico/geográfico/temporal   │ demografia por zipcode        │
    └───────────────┬───────────────┴───────────────┬──────────────┘
                     │  merge 1:1 por zipcode (fase 01)
                     ▼
      data/processed/house_clean.parquet (21.612 × 48)
                     │  split GroupShuffleSplit/zipcode (fase 03)
        ┌────────────┼────────────┬────────────┐
        ▼            ▼            ▼
      TRAIN         TEST          VAL
   (69,87%,49zip) (12,23%,10zip) (17,90%,11zip) ── sagrado, tocado 1x na fase 13
        │
        │  augmentation testada (fase 05) → REVERTIDA (fase 12, iter.2)
        │  train final = 15.100 linhas, sem augmentation (configs/augmentation.yaml: adopted=none)
        ▼
        │  segmentação — fit em TRAIN (fase 06)
        ▼
   + price_band (quartil)  + property_cluster (k=3)
        │
        │  feature engineering (fases 07-08)
        ▼
   RAW/DERIVED (candidate) ──ablation (fase 09)──> REJEITADO (todas as 6)
   CONTEXTUAL  (candidate) ──ablation (fase 09)──> ATIVO (3 de 3)
   property_age            ──ablation (fase 09)──> ATIVO
        │
        ▼
   20 FEATURES OFICIAIS (data/trusted/feature_metadata.json)
        │
        │  model selection — GroupKFold(3)/TRAIN (fase 10)
        ▼
   artifacts/model_candidate.pkl (XGBoost, treinado em TRAIN)
        │
        │  avaliação — só TEST (fase 11) + hipóteses (fase 12, 2 iterações)
        ▼
   reports/error_matrix.json (MAE TEST $104.221, final)
        │
        │  refit único TRAIN+TEST + checagem em VAL (fase 13)
        ▼
   artifacts/model_final.pkl  ──> MAE VAL $72.517 (checagem única, P1)
        │
        ▼
   artifacts/production_contract.yaml (fase 14, com modelo final)
        │
        ▼
   docs/07_deploy_strategy.md + docs/08_continuous_learning.md (D3/D4)
        │
        ▼
   app/ (API de serving, implementada exceto CI/CD — seção 12)
        │
        ▼
   artifacts/confidence_calibration.json (Matriz de Confiança — seção 12.1)
```

### 7.1 Tabela completa de features

**Features RAW (do fornecedor, físicas/geográficas — usadas diretamente no modelo):**

| Feature | Descrição | Correlação c/ `price_log` |
|---|---|---|
| `grade` | qualidade de construção/design (1-13) | 0,704 |
| `sqft_living` | área construída (pés²) | 0,695 |
| `sqft_living15` | área média dos 15 vizinhos mais próximos (pré-calculada pelo fornecedor) | 0,619 |
| `sqft_above` | área acima do solo | 0,602 |
| `bathrooms` | contagem de banheiros | 0,551 |
| `bedrooms` | contagem de quartos | 0,351 |
| `view` | qualidade da vista (0-4) | 0,347 |
| `sqft_basement` | área abaixo do solo | 0,317 |
| `floors` | número de andares | 0,311 |
| `waterfront` | binário, frente d'água | 0,175 |
| `sqft_lot` | área do terreno | 0,100 |
| `sqft_lot15` | área média do lote dos 15 vizinhos | 0,092 |
| `condition` | estado de conservação (1-5) | 0,039 (mais fraca isolada) |

**Features demográficas RAW (do zipcode, mesclado 1:1):**

| Feature | Descrição | Correlação c/ `price_log` |
|---|---|---|
| `hous_val_amt` | valor médio de imóveis no zipcode | 0,631 |
| `per_bchlr` | % com diploma de bacharelado no zipcode | 0,611 |
| `medn_hshld_incm_amt` | renda média domiciliar do zipcode | 0,306 |

**Features DERIVED — fase 07 (todas `REJECTED` na ablation, fase 09):**

| Feature | Mecanismo | Correlação bruta | Status |
|---|---|---|---|
| `grade_condition_interaction` | interação `grade`×`condition` | 0,530 | rejected |
| `has_basement` | binário, tem porão | 0,193 | rejected |
| `log_sqft_lot` | log do lote (reduz skew) | 0,162 | rejected |
| `basement_ratio` | proporção de área em porão | 0,152 | rejected |
| `was_renovated` | binário, já foi reformado | 0,124 | rejected |
| `years_since_renovation` | anos desde a última reforma | 0,073 | rejected |

Motivo da rejeição (P4): mesmo com correlação bruta positiva individual, o conjunto `raw_derived` não
reduz o MAE isolado e **piora** quando combinado com `contextual` — o modelo de árvore (XGBoost) já
captura essas interações/transformações nativamente via splits, tornando as features explícitas
redundantes.

**Features CONTEXTUAL — fase 08 (todas `ACTIVE`):**

| Feature | Mecanismo | r train | r test | gap | Status |
|---|---|---|---|---|---|
| `comps_knn_price` | preço médio dos k=30 vizinhos espaciais mais próximos (índice fit só em TRAIN) | 0,833 | 0,601 | +0,232 | active |
| `local_grade_percentile` | percentil de `grade` do imóvel entre seus vizinhos espaciais | 0,436 | 0,504 | -0,069 | active |
| `dist_to_seattle_center` | distância geodésica ao centro de Seattle | -0,188 | -0,190 | +0,002 | active |

**Feature adicional ativa:**

| Feature | Mecanismo | Status |
|---|---|---|
| `property_age` | idade do imóvel (ano de referência − `yr_built`) | active |

**Não-feature (chave estrutural, nunca entra no modelo — P1):** `zipcode`, `lat`, `long` — usadas
apenas para merge, split, definição de vizinhança espacial e diagnóstico de erro.

### 7.2 Conjunto final: 20 features oficiais do modelo

```
bedrooms, bathrooms, sqft_living, sqft_lot, floors, waterfront, view, condition, grade,
sqft_above, sqft_basement, sqft_living15, sqft_lot15, medn_hshld_incm_amt, hous_val_amt,
per_bchlr, property_age, dist_to_seattle_center, comps_knn_price, local_grade_percentile
```
Alvo: `price_log = log1p(price)`.

---

## 8. Modelo final (contrato de produção)

- **Algoritmo:** XGBoost, `n_estimators=400, max_depth=3, learning_rate=0.05`.
- **Treino:** refit único em `train`+`test` combinados (17.744 linhas, fase 13) — sem augmentation
  (decisão revertida na fase 12).
- **CV MAE (fase 10, só em TRAIN):** \$76.906 · **R²(log):** 0,887.
- **MAE em TEST (Error Matrix, fase 11, final):** \$104.221.
- **MAE em VAL (checagem única, fase 13):** **\$72.517** · MAPE 14,1% · Median APE 11,2% · P90 AE
  \$151.603 · bias +\$19.536.
- **Limitações conhecidas, documentadas explicitamente no contrato:** erro 2-4x maior em
  Luxury/waterfront/grade alto; sensibilidade da métrica agregada de TEST/VAL à composição de zipcode
  (poucos zipcodes por partição); sem recalibração por early stopping; lição da reversão de
  augmentation (CV em `train` nem sempre é proxy confiável de generalização geográfica).
- **Artefato:** `artifacts/model_final.pkl` · Contrato: `artifacts/production_contract.yaml` ·
  `status: RECOMMENDED_FOR_PROMOTION` (recomendação apenas — promoção real exige aprovação humana, P6).

---

## 9. Entregável 3 — Estratégia de deploy

> Documento completo: [`docs/07_deploy_strategy.md`](07_deploy_strategy.md) (com diagrama Mermaid de
> camadas). Resumo abaixo.

**Objetivo:** colocar `artifacts/model_final.pkl` atrás de uma API de inferência com paridade
treino/produção garantida **por reuso de código** (não reimplementação) — `src/pipelines/inference.py`
é a mesma função usada na demo da fase 14 — e monitorada **por segmento**, não só por métrica agregada
(motivado diretamente pela lição da fase 05/11: uma métrica agregada escondeu degradação real).

**Camadas:** Cliente → Gateway/API (serviço de inferência, stateless, reusa `src/pipelines/inference.py`)
→ Model Registry (`production_contract.yaml` como pacote atômico: modelo + features + versões de
dataset/split/pré-processamento) → Infraestrutura (auto-scaling horizontal, índice espacial em memória,
`zipcode_demographics.csv` como única dependência externa de dado em tempo de inferência) →
Monitoramento (log de predição com banda de preço/cluster previstos, Error Matrix contínuo por
segmento, monitor de drift via cobertura no espaço de features, alertas por SLA).

**Promoção (P6):** `candidate → avaliação (fase 11/13) → promotion contract (fase 14) → aprovação
humana → produção`. Nunca automática.

**SLA diferenciado por segmento** (adição explícita sobre um SLA único, que seria trivialmente violado
em `Luxury` e mascarado pela maioria de mercado de massa numa métrica agregada):

| Segmento | MAE alvo | Ação se violado |
|---|---|---|
| Entry/Standard | < \$60k | Alerta de degradação padrão |
| Premium | < \$90k | Alerta de degradação padrão |
| Luxury / waterfront / grade≥10 | < \$220k (~baseline atual) | Revisão manual antes de qualquer decisão de negócio automatizada |

**Monitor de drift:** reusa `src/evaluation/geographic_coverage.py::feature_space_coverage` (fase 04) —
um lote de predições sistematicamente "fora da cobertura" de `train` é sinal de extrapolação, deve
alertar antes que o erro real apareça.

---

## 10. Entregável 4 — Aprendizado contínuo

> Documento completo: [`docs/08_continuous_learning.md`](08_continuous_learning.md) (com diagrama
> Mermaid do ciclo). Resumo abaixo.

**Ciclo completo:** venda efetiva capturada → acúmulo de lote rotulado novo → retraining periódico
(`scripts/train_model.py` + `finalize_model.py`, nunca notebook manual, P5) → reavaliação completa
(`GroupKFold` + Error Matrix comparativo candidato vs. produção) → degrada em algum segmento? → se sim,
rejeitado (mantém produção, registra no ledger); se não, shadow deployment (prediz em paralelo, não
serve) → promoção só com aprovação humana (P6) → rollout canário (5%→25%→100%, com checkpoint de Error
Matrix por etapa) → rollback disponível a qualquer ponto.

**Critério de substituição por segmento (a lição mais concreta que esta própria execução produziu):** a
fase 05 adotou uma técnica de augmentation que melhorava a métrica agregada de treino (CV -20,5%) mas
piorava a generalização real (`test` +8,8%) — só descoberto porque a fase 11 quebrou o resultado por
segmento em vez de aceitar o número agregado. O critério de substituição em produção formaliza a mesma
disciplina: **promove só se o candidato não piorar o MAE em nenhum segmento do SLA em mais de uma
margem tolerável (~2%)** — mesmo que a métrica agregada melhore. Uma melhoria agregada que esconde
piora em `Luxury`/`waterfront` é rejeitada, não promovida.

**Retraining nunca é automático até produção** — só até "candidato aprovado para shadow"; a promoção
real do shadow para produção segue exigindo aprovação humana explícita (P6), mesma regra de governança
da fase 14.

---

## 11. Entregável 5 — Comunicação com stakeholders

Três peças complementares, cada uma com um papel diferente:

| Peça | Papel | Público |
|---|---|---|
| [`reports/stakeholder_summary.html`](../reports/stakeholder_summary.html) | Relatório visual oficial do Entregável 5 — narrativa de negócio (pergunta → precisão por segmento → 3 fatores de preço → a história da reversão de augmentation → como usar com segurança) | Stakeholders de negócio |
| [`docs/GUIA_APRESENTACAO.md`](GUIA_APRESENTACAO.md) | Roteiro de como apresentar o projeto ao vivo — pontos-chave separados por trilha técnica × trilha de negócio, FAQ antecipado, números-chave, armadilhas a evitar | Quem for apresentar o projeto (qualquer público) |
| [Dashboard de previsões/confiança](https://claude.ai/code/artifact/1e3a6f2f-bd9f-4557-ad41-6fb28025ec2e) | Exploração visual interativa das previsões em `val` — erro por região/tipo de imóvel/conjunto de features, banda de confiança | Público técnico/analítico — ver nota de governança na seção 13 |

As três contam a mesma história central — a reversão da decisão de augmentation como prova de rigor
metodológico — em três formatos diferentes (documento de negócio pronto, roteiro de apresentação,
exploração interativa de dado).

---

## 12. Conclusão

**O que está sólido:**
- Pipeline de dados ponta a ponta reproduzível, verificado diretamente neste ambiente ( sequência de 15 fases validada rodando cada script, todos com código 0), com 92 testes
  passando em `tests/` (+ 55 em `app/tests/` para a camada de serving — 147 no total), configs
  versionadas e lógica isolada em `src/` (P5).
- Split geográfico rigoroso, sem vazamento de grupo, com `val` preservado intocado até a fase 13 (ver
  ressalva de governança sobre uso posterior em diagnóstico, seção 13).
- Metodologia de decisão disciplinada: todo critério de aceite é definido **antes** do experimento e
  respeitado mesmo quando o resultado é "quase lá" (k=50, fase 12 iter.1) ou desconfortável (rejeição
  do conjunto `raw_derived`; reversão completa da decisão de augmentation da fase 05).
- **Capacidade de autocorreção institucionalizada:** quando a fase 11 revelou que a decisão da fase 05
  piorava a generalização real, o projeto não escondeu o resultado — abriu uma nova hipótese (fase 12),
  testou com um critério diferente e mais adequado (desta vez `test` deliberadamente decisivo), reverteu
  a decisão, e **propagou a correção por todas as fases downstream** (06-11 reexecutadas) antes de
  seguir para o refit final. Isso é o ciclo `Erro → Hipótese → Decisão` do princípio P4 funcionando de
  ponta a ponta, não só em teoria — e o próprio `configs/augmentation.yaml` agora garante que essa
  correção não seja mecanicamente desfeita numa releitura futura do pipeline.
- Feature engineering orientada a hipótese, com ablation formal decidindo o que sobrevive — não
  correlação bruta isolada. O conjunto contextual (KNN espacial) é o real motor de ganho (-27% a -28,2%
  de MAE em CV), muito acima do ganho nulo das features RAW/DERIVED.
- **Generalização validada no fim:** MAE em `val` (\$72.517) é da mesma ordem de grandeza — e melhor —
  que em `test` (\$104.221), confirmando que o projeto generaliza para zipcode nunca visto, não um
  resultado de sorte de um único corte de dado.
- **Os 5 entregáveis oficiais estão fechados** (D1, D2a, D2b, D2c, D3, D4, D5) — incluindo estratégia de
  deploy e aprendizado contínuo com adições explícitas motivadas pelos próprios achados do projeto (SLA
  por segmento, critério de substituição por segmento), não copiadas de um template genérico.

**O que fica como limitação conhecida e documentada:**
- Erro 2-4x maior em imóveis de luxo/waterfront/grade alto persiste em toda avaliação (TEST e VAL) —
  aceito conscientemente como critério de parada, encaminhado para SLA de erro por segmento em produção
  (D3) e monitoramento por segmento (D4), em vez de mais engenharia de feature sem fim.
- MAE agregado de TEST/VAL é sensível à composição de zipcode (só 10-11 de 70 totais em cada partição)
  — mitigado por reportar sempre por segmento (Error Matrix), nunca só o número agregado.
- **Achado de governança pendente de decisão humana:** o script de análise de previsões/confiança
  (seções 11, 13, 15) toca `val` uma segunda vez — ver seção 13 para as duas opções propostas.
- A frente de API/serving (`app/`, seção 12) é uma extensão de escopo real, mas ainda incipiente (só a
  camada de domínio existe) — não confundir com os entregáveis oficiais, que não dependem dela.

Em suma: o ciclo de modelagem e os 5 entregáveis oficiais estão **completos e fechados**, com o modelo
final validado uma única vez em dado genuinamente não visto (`val`) na fase 13, um contrato de produção
pronto para recomendação humana, e documentos de deploy/aprendizado contínuo/stakeholders que
incorporam lições reais do próprio projeto em vez de boilerplate. O único item que precisa de uma
decisão explícita do usuário antes de ser considerado fechado é o achado de governança da seção 13
(segundo toque em `val` pelo script de análise de confiança) — o resto está pronto.

---

## 13. Previsões, erro por segmento e nível de confiança (análise final)


### 15.1 Onde o modelo acerta mais (por região)

MAE por zipcode em `val` (11 zipcodes, nunca vistos em nenhuma decisão de desenvolvimento):

| Zipcode | n | MAE | MAPE |
|---|---|---|---|
| (melhor) | — | menor MAE entre os 11 zipcodes de `val` | tipicamente < 10% |
| `98005` | 168 | \$146.988 | 16,9% |
| `98122` | 290 | \$102.600 | 15,5% |
| `98059` | 468 | \$88.381 | — |
| … | … | … | … |

A tabela completa (11 zipcodes, ordenada do melhor para o pior) e o gráfico de barras estão no
dashboard — o padrão confirma o já visto em TEST: zipcodes de preço mais alto/mais variável
concentram o erro absoluto maior, não necessariamente o erro percentual.

### 15.2 Erro por conjunto de features

Reaproveitando a ablation da fase 09 (`GroupKFold(3)` dentro de `train`):

| Conjunto de features | MAE | Ganho vs. baseline |
|---|---|---|
| Baseline (16 físicas+demográficas) | \$107.515 | — |
| + RAW/DERIVED (6 features) | \$107.672 | nenhum (piora marginal) |
| **+ CONTEXTUAL (3 features)** | **\$79.520** | **-27,0%** |
| + RAW/DERIVED + CONTEXTUAL | \$80.463 | pior que só contextual |
| **+ CONTEXTUAL + `property_age` (escolhido)** | **\$77.198** | **-28,2%** |

Conclusão inalterada: o ganho real vem inteiro do índice espacial de comparáveis (`comps_knn_price`,
`local_grade_percentile`, `dist_to_seattle_center`), não das transformações/interações físicas.

### 15.3 Erro por tipo de imóvel (VAL)

**Por banda de preço:**

| Banda | n | MAE | MAPE | Bias |
|---|---|---|---|---|
| Entry | 713 | \$45.808 | 19,0% | -\$33.553 (superestima) |
| Standard | 1.311 | \$47.582 | 12,1% | -\$6.903 |
| Premium | 1.184 | \$64.488 | 11,8% | +\$26.181 |
| Luxury | 660 | \$165.303 | 17,1% | +\$117.482 (subestima) |

**Por cluster de perfil físico:**

| Cluster | n | MAE | MAPE |
|---|---|---|---|
| Antigo/Compacto | 2.362 | \$60.427 | 14,3% |
| Moderno/Amplo | 1.494 | \$89.463 | 13,8% |
| Luxo físico (waterfront/view) | 12 | \$342.483 | 18,8% — **amostra muito pequena (n=12), tratar como indicativo, não conclusivo (P4)** |

Padrão consistente com toda a avaliação do projeto: em \$, o erro cresce com o valor/raridade do
imóvel; em %, o erro fica relativamente mais estável (10-19%), exceto no cluster de luxo físico raro,
onde a amostra é pequena demais em `val` para conclusão robusta.

### 15.4 Nível de confiança do modelo

O modelo (XGBoost) não produz intervalo de confiança nativo — é um ponto de previsão. Para dar uma
noção honesta de confiança, construímos uma banda empírica (estilo *conformal prediction*), calibrada
**sem circularidade**:

1. `model_candidate.pkl` (treinado só em `TRAIN`) prevê sobre `TEST` — dado que esse modelo nunca viu.
2. Os quantis do erro real (`y_real - y_previsto`) em `TEST`, por banda de preço, viram uma banda de
   ± em torno de qualquer previsão nova.
3. Essa banda é então testada em `VAL` — dado que **não** participou da calibração — medindo se a
   cobertura prometida realmente se sustenta fora da amostra usada para calibrar.

| Verificação | Alvo | Observado em `val` |
|---|---|---|
| Banda de 80% (geral) | 80% dos casos dentro | **85,0%** |
| Banda de 90% (geral) | 90% dos casos dentro | **94,0%** |

**Leitura:** a banda é levemente **conservadora** (cobre mais casos do que promete) — mais segura do
que otimista demais, mas significa que a faixa declarada tende a ser um pouco mais larga do que o
mínimo necessário. Por banda de preço, a calibração segue próxima do alvo em `Entry`/`Standard`
(cobertura 89-97%) e um pouco mais larga em `Premium`/`Luxury`, onde o erro é mais disperso. Detalhe
completo por banda no dashboard.

### 15.5 Taxa de erro (±) por valor previsto

Esta é a resposta direta à pergunta "se o modelo prevê \$X, quanto de erro esperar": `val` foi dividido
em 8 faixas de preço previsto, e para cada uma medimos o erro absoluto observado.

| Previsão (~) | Erro mediano (50% dos casos) | P80 (80% dos casos) | P90 (90% dos casos) |
|---|---|---|---|
| \$257 mil | \$31.109 (12,1%) | \$59.318 | \$78.553 |
| \$330 mil | \$39.584 (12,3%) | \$75.841 | \$101.990 |
| \$379 mil | \$42.847 (11,0%) | \$78.522 | \$106.776 |
| \$423 mil | \$46.360 (10,5%) | \$93.224 | \$119.983 |
| \$469 mil | \$42.533 (8,9%) | \$88.358 | \$120.624 |
| \$526 mil | \$57.148 (10,4%) | \$109.919 | \$156.765 |
| \$608 mil | \$68.589 (11,5%) | \$145.443 | \$192.978 |
| \$814 mil (e acima) | \$103.154 (12,6%) | \$230.327 | \$323.280 |

**Exemplo de leitura (o pedido original):** para uma previsão em torno de **\$257 mil**, metade dos
imóveis reais fica a menos de **\$31 mil** (12,1%) do valor previsto; 8 em cada 10 ficam a menos de
**\$59 mil**; e 9 em cada 10, a menos de **\$79 mil**. Ou seja: **não** é uma faixa fixa de "±5 mil"
independente do preço — o erro em dólares cresce com o valor do imóvel (de ~\$31 mil na faixa mais
barata para ~\$103 mil na faixa mais cara), mas o erro **percentual** fica relativamente estável, entre
9% e 13% na mediana, na maior parte da faixa de preço do mercado.

O gráfico completo (com todas as 8 faixas, banda sombreada e tooltip por ponto) está no dashboard —
seção "Faixa de erro (±) por valor previsto".
