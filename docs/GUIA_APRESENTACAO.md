# Guia de Apresentação do Projeto — Previsão de Preços de Casas (Seattle)

> Roteiro de apoio para apresentar o projeto a dois públicos diferentes: avaliação/plateia técnica e
> stakeholders de negócio. Baseado no estado final do projeto — 5 entregáveis oficiais fechados (D1-D2c,
> D3, D4, D5) + API de serving/portal Inference + Matriz de Confiança implementadas (fora do escopo oficial).
> Ver `docs/PROJETO_VISAO_GERAL.md` para os números completos,
>  e `docs/09_confidence_matrix.md` para a metodologia completa da Matriz de Confiança.

---

## 1. O arco narrativo (use em qualquer público, ajustando a profundidade)

Não abra com modelo/algoritmo. Abra com a pergunta de negócio e conte o projeto como uma investigação,
não como uma lista de tarefas concluídas:

1. **Problema:** prever preço de imóvel de forma que generalize para regiões nunca vistas, não só
   "acertar a média".
2. **Abordagem:** o projeto é orientado por hipótese, não por algoritmo — cada fase responde uma
   pergunta específica antes de decidir a próxima.
3. **Descoberta central:** o mercado tem estrutura espacial forte, mas o que prediz erro não é
   "distância geográfica" — é "quão atípico o imóvel é frente ao que o modelo já viu". Isso muda toda a
   estratégia de features/augmentation depois.
4. **Momento de rigor #1 (o ponto mais forte da apresentação):** uma decisão foi tomada corretamente
   segundo a regra, mas o resultado real (checado depois) mostrou que a métrica de proxy enganou. O
   projeto reverteu a decisão, propagou a correção por todo o pipeline, e documentou os dois lados —
   sem esconder o erro. **Esse é o parágrafo que mais diferencia o projeto — não pule.**
5. **Resultado final:** modelo validado uma única vez em dado nunca tocado (`val`), com erro conhecido
   por segmento, não só uma média.
6. **Momento de rigor #2:** cada previsão agora vem com um Confidence Score (0-100) — e a construção
   dessa camada revelou (e corrigiu) dois bugs reais de metodologia antes de ir para produção, além de
   documentar uma limitação que sobrou mesmo depois da correção.
7. **Fechamento com produto real:** o modelo não fica só em notebook — tem API, portal de MLOps e
   documento de deploy/aprendizado contínuo prontos, com o que falta (CI/CD) declarado, não escondido.
8. **Limitações honestas + próximos passos:** o que ainda erra mais (imóveis de luxo), por quê, e o
   que falta para produção real.

Essa mesma espinha dorsal serve para os dois públicos — muda só o nível de detalhe de cada passo.

---

## 2. Trilha TÉCNICA — o que enfatizar

Público: avaliadores técnicos, outros cientistas de dados, engenheiros. Aqui o rigor metodológico *é*
o produto — não economize nos porquês.

### 2.1 Pontos que demonstram maturidade metodológica

- **Validação geográfica, não aleatória.** `GroupShuffleSplit`/`GroupKFold` por `zipcode` — explique
  por que um split aleatório por linha superestimaria a generalização (zipcode vazando entre
  treino/teste). `zipcode` nunca é feature (P1) — só chave de agrupamento/split/diagnóstico.
- **EDA roda em 100% dos dados, antes do split.** Descreva a diferença entre "leitura" (não tem
  `.fit()`, pode rodar em tudo) e "transformação que aprende" (só em `train`, sempre pós-split). Esse é
  um erro comum em projetos de ML que o seu evita explicitamente.
- **`val` é sagrado — tocado uma única vez para decisão de modelo.** Consegue provar isso: nenhum
  script de decisão de pipeline (feature/modelo/hiperparâmetro) lê `split == "val"` fora da fase 13.
  Duas exceções conhecidas, ambas puramente leitura/diagnóstico pós-decisão (nunca decidem nada) e
  documentadas abertamente: o dashboard de previsões e a verificação da Matriz de Confiança (seção 2.3)
  — se um avaliador perguntar sobre isso, é sinal de que ele leu com atenção; a resposta certa é
  reconhecer a exceção, explicar por que ela não contamina nenhuma decisão, e apontar que o próprio
  projeto sinalizou isso sozinho (ver `docs/PROJETO_VISAO_GERAL.md`, seção 13).
- **Ablation, não correlação bruta, decide feature.** Conte o caso `raw_derived`: 6 features com
  correlação bruta individual positiva (`grade_condition_interaction` = 0,53), mas rejeitadas porque
  pioram o conjunto combinado com `contextual` — o modelo de árvore já captura essas interações
  nativamente. **Correlação alta não é motivo de promoção; ganho incremental medido é.**
- **Critério de aceite sempre definido antes do experimento (P4).** Dois exemplos concretos e fortes:
  - Fase 10 (k-sweep): k=50 melhorava tudo, mas ficou a 0,44 ponto percentual do critério de -5% em
    Luxury — e foi **rejeitado mesmo assim**. Isso é disciplina, não teimosia: "não movemos a régua
    depois de ver o resultado".
  - Fase 05 → 12: a decisão de augmentation seguiu a regra (decidir por CV dentro de `train`, nunca por
    `test`) e mesmo assim a decisão inicial estava errada — corrigida só quando o Error Matrix (fase 11)
    expôs a divergência entre proxy e realidade.
- **Empate técnico é reportado, não escondido atrás do "vencedor".** Ridge chegou a ~1,8% do MAE do
  XGBoost — mensagem: **escolha de features pesou mais que escolha de algoritmo** neste problema. Isso é
  uma afirmação forte e defensável, não um placeholder de slide.
- **Avaliação sempre segmentada (Error Matrix), nunca só a métrica agregada.** Global × banda de preço ×
  cluster físico × zipcode. Prepare-se para explicar por que isso importa: uma melhora global pode
  esconder piora grave num segmento.

### 2.2 A história da reversão do augmentation (a "prova de rigor #1" — conte com calma)

Roteiro sugerido, em 4 frases:
1. "Testamos duas técnicas de augmentation em `train`, com um critério pré-registrado: promove só se o
   MAE de validação cruzada dentro de `train` melhorar ≥2%. A técnica B melhorou 11,4% — adotamos."
2. "Mais tarde, o Error Matrix mostrou que essa mesma decisão piorava o erro em `test` (zipcode nunca
   visto) em 8,8% — apesar de termos seguido a regra corretamente."
3. "Abrimos uma nova hipótese, com um critério diferente e mais adequado a essa pergunta específica
   (`test` deliberadamente decisivo, porque o propósito ali era medir generalização) — e revertemos."
4. "Recomputamos as 6 fases seguintes (segmentação, features, ablation, seleção de modelo, avaliação)
   com a correção — não só trocamos um número, propagamos a mudança pelo pipeline inteiro. E fixamos a
   decisão em `configs/augmentation.yaml`, versionado, pra ela nunca ser desfeita silenciosamente numa
   releitura futura do pipeline."

Se perguntarem "isso não é uma falha do projeto?" — resposta pronta: **não, é o ciclo de hipótese
funcionando.** A falha seria não checar depois, ou esconder o resultado ruim. As duas decisões (a
original e a reversão) continuam registradas no audit log — nada foi apagado.

### 2.3 A Matriz de Confiança (a "prova de rigor #2" — dois bugs reais achados e corrigidos)

Contexto rápido: cada previsão da API agora carrega um **Confidence Score (0-100)** e uma categoria
(Alta / Moderada / Baixa / Muito baixa confiança), calibrados em `TEST` e verificados uma única vez em
`VAL` — mesma disciplina (`TEST` calibra, `VAL` verifica, nada se reajusta depois de ver o resultado).

**A parte que mais vale contar — dois bugs reais, achados e corrigidos durante a implementação, não
depois que alguém reclamou:**

1. **Cortes de categoria inatingíveis.** Primeira versão usava os quartis do **erro bruto individual**
   como corte de categoria — e a categoria "Alta confiança" ficava praticamente vazia, porque nenhum
   segmento (que é sempre uma média, mais suave) consegue chegar tão baixo quanto o melhor caso
   individual isolado. Correção: cortar nos quartis do **erro já calibrado** pela regressão isotônica —
   garante que os 4 cortes são sempre atingíveis pela própria curva.
2. **Matriz 2×2 usando erro real em vez de esperado.** A primeira versão da matriz didática (cobertura ×
   erro) cortava o eixo "erro" pelo erro **real observado** — o que é diagnóstico retrospectivo, não
   confiança prospectiva (você não sabe o erro real antes de prever). Corrigido para usar o erro
   **esperado do segmento**, o único que existe no momento da predição.

**Resultado desta execução** (não são números redondos de propósito — vêm do dado):

| Categoria | Score mínimo | Teto de erro (TEST) | MAE mediano (TEST) |
|---|---|---|---|
| 🟢 Alta confiança | **88** | 16,3% | US$ 62.483 |
| 🟡 Confiança moderada | **55** | 20,9% | US$ 40.653 |
| 🟠 Baixa confiança | **17** | 23,1% | US$ 78.478 |
| 🔴 Muito baixa / revisão | < 17 | — | US$ 126.801 |

**Terceira honestidade, documentada sem maquiagem:** verificado em `VAL`, cada categoria respeitou seu
teto de erro prometido — mas a **ordem fina** das 4 categorias não ficou perfeitamente monotônica
("Alta confiança" teve erro observado um pouco maior que "Confiança moderada" e "Baixa confiança").
Causa identificada: parte da categoria "Alta confiança" vem de bandas de preço mais baratas (`Entry`),
que já tem um viés de erro percentual mais alto documentado desde a fase 11/13 — não um problema novo,
é o mesmo padrão de sempre aparecendo numa lente diferente. **Decisão:** mantido como está, documentado
abertamente; uso recomendado é confiar no teto de erro por categoria (sempre respeitado) e na matriz
2×2 (mais simples, mais robusta em `VAL`) — tratar a ordem fina como indicativa, não perfeita.

Se perguntarem "por que não escondeu isso e chamou de pronto?" — resposta pronta: **a mesma cultura do
projeto inteiro.** Documentar o que não funcionou perfeitamente é mais crível do que prometer perfeição.

### 2.4 API + Portal Inference (implementado, fora do escopo oficial)

Não fazia parte dos entregáveis do desafio — foi construído como extensão para mostrar o modelo
funcionando de ponta a ponta, não só documentado. Pontos a citar se perguntado:

- **Clean Architecture** (`domain` → `application` → `infrastructure`/`api`) — reusa `src/`/`scripts/`
  do pipeline de ML, não duplica lógica de inferência/treino.
- Endpoints cobrem inferência, confiança por predição, feedback (fecha o ciclo de aprendizado
  contínuo), model registry/promoção (P6, nunca automática), retraining, drift (PSI), explicabilidade
  (SHAP global + local).
- Portal Streamlit externo (8 páginas) — cliente HTTP puro da API, para o engenheiro de MLOps acompanhar
  e agir (aprovar promoção, disparar retraining) sem tocar em código.
- **O que falta, declarado sem rodeio:** CI/CD (`.github/workflows/`) não foi implementado — decisão
  explícita de escopo, não esquecimento.
- 161 testes no total (81 do pipeline de ML + 80 da API/serving).

### 2.5 Perguntas técnicas prováveis (com resposta pronta)

| Pergunta | Resposta curta |
|---|---|
| Por que não usar `lat`/`long` direto no modelo? | Entram só via features derivadas leakage-safe (distância, KNN espacial, percentil local) — coordenada/zipcode cru memorizaria identidade administrativa, não aprenderia mercado (P1). |
| Como vocês sabem que `comps_knn_price` não é vazamento? | Índice espacial é fit só em `train`; o gap de correlação train→test (0,83→0,60) é o esperado de generalização geográfica — leakage inflaria `test`, não o contrário. |
| Por que XGBoost e não um modelo mais complexo (rede neural, ensemble)? | Grid enxuto por decisão consciente de escopo; resultado mostrou empate técnico entre XGBoost e Ridge — sinal de que mais complexidade de modelo não era o gargalo, features eram. |
| Qual é o intervalo de confiança da previsão? | Duas camadas: banda empírica por faixa de preço (dashboard, calibrada em TEST/verificada em VAL) e o Confidence Score por predição (seção 2.3), mesma disciplina de calibração. |
| Por que o Confidence Score não é perfeitamente monotônico em VAL? | Documentado abertamente — parte do efeito é o viés já conhecido de `Entry` ter MAPE mais alto. Teto de erro por categoria continua respeitado; a matriz 2×2 é a leitura mais robusta. |
| `val` foi mesmo tocado só uma vez? | Para decisão de pipeline, sim (checável — nenhum script de decisão lê `split=="val"` fora da fase 13). Dois scripts de leitura/diagnóstico pós-decisão (dashboard, verificação da Matriz de Confiança) tocam `val` de novo só para reportar — não influenciam nenhuma escolha, e isso está documentado como um achado de governança em aberto, não escondido. |
| Como isso escala/se mantém em produção? | Contrato de produção versionado (schema de features, versão de modelo/dataset/pré-processamento) — nunca promoção automática, sempre aprovação humana (P6). Existe API/portal implementados para operacionalizar isso. |
| O que vocês fariam com mais tempo/dados? | Ir atrás do segmento de luxo com mais dado real (não synthetic) em vez de mais engenharia de feature; formalizar CI/CD; resolver a decisão de governança pendente sobre os scripts que tocam `val` uma segunda vez. |

### 2.6 Recursos para ter abertos/prontos na tela

- `roadmap/PRINCIPLES.md` — se pedirem para justificar uma decisão de governança.
- `reports/AUDIT_LOG.md` — histórico cronológico completo, inclusive a reversão.
- `docs/PROJETO_VISAO_GERAL.md` — visão fase-a-fase com números, seção 13 tem o achado de governança.
- `docs/09_confidence_matrix.md` — metodologia completa da Matriz de Confiança.
- `app/README.md` / `portal/README.md` — arquitetura e decisões de escopo da API/portal.
- [Dashboard de previsões/confiança](https://claude.ai/code/artifact/1e3a6f2f-bd9f-4557-ad41-6fb28025ec2e) —
  scatter previsto×real, erro por região/feature-set/tipo de imóvel, banda de confiança.
- `http://localhost:8000/docs` (Swagger) e portal em `http://localhost:8501`, se for demonstrar ao vivo
  (`docker compose up --build` — ver `docs/DOCKER_GUIA.md`).

---

## 3. Trilha STAKEHOLDER / negócio — o que enfatizar

Público: quem decide se o modelo vai para produção, patrocinador, área de negócio. Aqui o vocabulário
técnico atrapalha — traduza tudo para "o que isso significa pra decisão de preço/risco".

### 3.1 Regra de ouro: traduza toda métrica para dinheiro e para "quando confiar"

Não diga "MAE de \$72.517" sem contexto. Diga:

> "Em média, a previsão erra cerca de 14% do valor do imóvel. Para uma casa de \$300 mil, isso é uma
> faixa de erro de aproximadamente ±\$30-40 mil — suficiente para apoiar decisão, não para substituir
> avaliação formal em transações de alto valor."

Tabela pronta para essa tradução (fonte: seção 15.5 de `docs/PROJETO_VISAO_GERAL.md`):

| Se o modelo previu... | Espere um erro típico de... | Em 9 de 10 casos, o erro fica até... |
|---|---|---|
| ~\$257 mil | ~\$31 mil (12%) | ~\$79 mil |
| ~\$423 mil | ~\$46 mil (11%) | ~\$120 mil |
| ~\$608 mil | ~\$69 mil (12%) | ~\$193 mil |
| ~\$814 mil ou mais | ~\$103 mil (13%) | ~\$323 mil |

**Mensagem central:** o erro em dólares cresce com o preço do imóvel, mas o erro percentual fica
relativamente estável (9-13% na maior parte do mercado) — **exceto** em imóveis de luxo, onde é maior
(ver 3.2).

### 3.2 Onde confiar mais, onde confiar menos

Fale isso sem rodeio — é a informação mais acionável para quem vai usar o modelo:

- **Confiança mais alta:** imóveis de padrão `Entry`/`Standard`/`Premium` (a maioria do mercado) —
  erro percentual entre ~10% e ~19%.
- **Confiança mais baixa:** imóveis `Luxury` e o cluster físico de waterfront/vista privilegiada — erro
  2 a 4 vezes maior em dólares. Não é um bug a ser escondido: é uma característica conhecida do mercado
  de imóveis de alto padrão (menos transações comparáveis, mais variação idiossincrática de preço) —
  **e o time investigou ativamente uma correção, testou, e documentou por que não adotou** (ver 2.1,
  caso k=50) em vez de simplesmente ignorar o problema.
- **Recomendação de uso:** tratar a previsão do modelo como um ponto de partida/triagem, com revisão
  humana obrigatória em segmentos de alto valor — não como preço final automático nesses casos.

### 3.3 Novidade: cada previsão agora vem com um "selo de confiança"

Explique isso em termos simples, sem jargão de calibração:

> "Além do preço previsto, o sistema agora entrega uma nota de confiança de 0 a 100 e um selo — Alta,
> Moderada, Baixa ou Muito Baixa (essa última, recomendamos revisão manual). Isso foi construído e
> testado com o mesmo rigor do resto do projeto: durante a construção, encontramos e corrigimos dois
> erros reais de cálculo antes de considerar pronto — e documentamos abertamente uma limitação que
> sobrou (a ordem das 4 categorias não é perfeita, mas cada uma cumpre o teto de erro prometido)."

Isso é uma ótima resposta para "como sei quando confiar no número?" — em vez de um preço solto, o
usuário de negócio recebe preço + um selo de quanto confiar nele.

### 3.4 Por que confiar no processo (mesmo sem entender os detalhes técnicos)

Quatro frases que resumem por que o projeto é confiável sem precisar do jargão:

1. "Toda decisão importante teve uma regra definida **antes** de ver o resultado — inclusive quando o
   resultado não veio como esperado, a regra foi respeitada, não ajustada depois."
2. "Quando encontramos uma decisão que parecia certa mas não funcionava na prática, nós mesmos
   detectamos, corrigimos e documentamos — sem esperar alguém de fora encontrar depois. Isso aconteceu
   duas vezes: na decisão de aumentar os dados de treino, e na construção da nota de confiança."
3. "O modelo final só foi testado contra dado 100% novo (nunca usado em nenhuma decisão) uma única vez,
   no final — para garantir que o número que estamos mostrando aqui é honesto, não inflado por reuso
   repetido dos mesmos dados de teste."
4. "O modelo não fica só em relatório — já existe uma API funcionando e um portal para o time técnico
   acompanhar e aprovar mudanças, com aprovação humana obrigatória antes de qualquer substituição em
   produção."

### 3.5 O que ainda falta (não esconda isso — controla a expectativa)

- **Não há promoção automática para produção.** O pipeline recomenda (`RECOMMENDED_FOR_PROMOTION`); a
  substituição do modelo em produção sempre exige aprovação humana explícita.
- **CI/CD (integração/entrega contínua automatizada) ainda não foi implementado** — a API e o portal
  funcionam e têm testes automatizados, mas o pipeline de build/deploy automático na nuvem é o próximo
  passo, não algo já em produção.
- **O erro em imóveis de luxo é uma limitação conhecida**, não resolvida — decisão consciente de não
  perseguir esse segmento indefinidamente sem um resultado que realmente comprovasse ganho (ver 2.1).
- **A ordem fina das 4 categorias de confiança não é perfeita** (ver 3.3) — use o selo como guia, não
  como garantia absoluta de ranking entre "Alta" e "Moderada".

### 3.6 Perguntas de negócio prováveis (com resposta pronta)

| Pergunta | Resposta curta |
|---|---|
| O modelo está pronto para usar hoje? | Está recomendado para promoção, mas a decisão final de colocar em produção é humana — e o ideal é começar com revisão humana obrigatória em imóveis de alto valor. |
| Quanto o modelo erra, na prática? | Em média ~14% do valor do imóvel; para a maior parte do mercado, entre 9% e 13%; em imóveis de luxo, pode passar de 17-19%. |
| Como sei em qual previsão confiar mais? | Cada previsão vem com uma nota de confiança de 0-100 e um selo (Alta/Moderada/Baixa/Muito Baixa) — abaixo de "Muito Baixa", recomenda-se revisão manual antes de qualquer decisão. |
| Por que não é mais preciso? | O mercado de imóveis de alto padrão tem poucas transações comparáveis e mais variação — o time testou uma correção específica para isso e não adotou porque não passou no critério definido antes do teste (rigor, não desistência). |
| Isso substitui um avaliador humano? | Não — é uma ferramenta de apoio/triagem com faixa de erro conhecida, não um substituto de avaliação formal, especialmente em imóveis caros. |
| O que muda se tivermos mais dados no futuro? | Há um processo de aprendizado contínuo desenhado e parcialmente implementado (API já tem endpoint de retraining/feedback) — falta só a automação de deploy (CI/CD) e a operação real em produção. |

---

## 4. Roteiro sugerido de apresentação (única, com profundidade ajustável)

Estrutura de ~18-22 minutos que funciona para plateia mista (ajuste o tempo de cada bloco conforme o
público predominante):

1. **Abertura (1-2 min):** o problema de negócio e o objetivo do sistema (seção 1 deste guia).
2. **Como o projeto foi construído (3-4 min):** ciclo hipótese→decisão, validação geográfica, avaliação
   sempre por segmento. Nível técnico ajustável — para stakeholders, fale em "regras definidas antes de
   ver o resultado"; para técnicos, cite `GroupKFold`/P1-P7 diretamente.
3. **A descoberta central + a reversão do augmentation (4-5 min, o primeiro clímax):** conte a história
   de 2.2 — funciona nos dois públicos, é o momento que mais gera confiança.
4. **Resultado final + Matriz de Confiança (4-5 min, o segundo clímax):** números da seção 3.1 (tradução
   em dinheiro) + onde confiar mais/menos (3.2) + o selo de confiança por predição (3.3/2.3) — mostre o
   dashboard e, se possível, uma chamada real à API retornando `confidence_score`.
5. **API/portal + o que falta (2-3 min):** mostre que o modelo roda de verdade (Swagger ou portal), e
   diga com clareza o que ainda não foi feito (CI/CD, seção 3.5).
6. **Limitações e próximos passos (2-3 min):** seção 3.5, sem rodeio.
7. **Perguntas:** use as tabelas de FAQ (2.5 e 3.6) como preparo, não como script a ler.

---

## 5. Números-chave para decorar (cola rápida)

| Métrica | Valor |
|---|---|
| Dataset | 21.612 imóveis, 70 zipcodes, Seattle |
| Split | 70% treino / 12% teste / 18% validação, por zipcode (nunca por linha) |
| Features no modelo final | 20 (16 físicas/demográficas + `property_age` + 3 contextuais espaciais) |
| Modelo final | XGBoost (`n_estimators=400, max_depth=3, learning_rate=0.05`) |
| MAE em validação final (`val`, tocado 1x para decisão) | **\$72.517** |
| MAPE em validação final | **14,1%** (mediana: 11,2%) |
| Erro típico (mediana) | de ~\$31 mil (imóveis ~\$250 mil) a ~\$103 mil (imóveis ~\$800 mil+) |
| Cobertura da banda de confiança de 90% (dashboard) | 94% observado em `val` (levemente conservadora) |
| Confidence Score — cortes de categoria (calibrados, não redondos) | 88 / 55 / 17 (Alta / Moderada / Baixa) |
| Confidence Score — teto de erro por categoria (TEST) | Alta 16,3% · Moderada 20,9% · Baixa 23,1% |
| Segmento com maior erro relativo | `Luxury` / cluster waterfront-vista (2-4x o erro do mercado padrão) |
| Ganho de MAE por engenharia de features | -28,2% (baseline → conjunto final, via ablation) |
| Decisão mais importante do projeto | Reversão do augmentation (fase 12) após Error Matrix expor divergência entre proxy (CV em train) e generalização real (test) |
| Testes automatizados | 161 no total (81 pipeline de ML + 80 API/serving) |
| Entregáveis oficiais | 5 de 5 fechados (D1, D2a, D2b, D2c, D3, D4, D5) |
| Além do escopo oficial | API + portal Inference implementados; falta só CI/CD |

---

## 6. Armadilhas a evitar

- **Não venda o modelo como "preciso".** Venda-o como "com erro conhecido e documentado por segmento" —
  isso é a força real do projeto, não uma fraqueza a disfarçar.
- **Não esconda a reversão do augmentation atrás de "ajustamos o modelo".** Contada bem, é um dos dois
  pontos mais fortes da apresentação; escondida, parece que só está sendo omitida.
- **Não apresente o Confidence Score como perfeito.** A ordem fina das 4 categorias não é
  perfeitamente monotônica em `VAL` — isso já está documentado, e fingir que não existe é pior do que
  admitir e explicar o teto de erro (que sempre se sustentou).
- **Não prometa que o modelo resolve o segmento de luxo.** Ele não resolve — e o time testou e
  documentou por que a correção óbvia (k maior) não passou no critério.
- **Não diga que a API está "em produção" ou que tem CI/CD.** Está implementada e testada localmente
  (Docker incluso), mas CI/CD é uma lacuna declarada, não um detalhe a esconder.
- **Não confunda D3/D4 (documentos) com a implementação real em `app/`.** Deixe claro que os documentos
  desenharam a estratégia e a API é uma realização concreta dela — mas ainda sem pipeline de deploy
  automatizado.
- **Para o público técnico, não pule os "porquês" das convenções (P1-P7)** — são exatamente o que
  demonstra maturidade, não detalhe de implementação.
