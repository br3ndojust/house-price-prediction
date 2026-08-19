# Entregável 5 — Comunicação com Stakeholders

> Como apresentar os resultados deste modelo para um público de negócio — sem jargão técnico, com
> métricas traduzidas em impacto e uma história, não uma lista de números soltos.

## 1. Regra de ouro: nunca solte uma métrica sem contexto de negócio

Não diga "MAE de US$ 72.517". Diga:

> "Em média, a previsão erra cerca de 14% do valor do imóvel. Para uma casa de US$ 300 mil, isso é
> uma faixa de erro de aproximadamente ±US$ 30-40 mil — suficiente para apoiar uma decisão de
> precificação, não para substituir uma avaliação formal em transações de alto valor."

| Se o modelo previu... | Espere um erro típico de... | Em 9 de 10 casos, o erro fica até... |
|---|---|---|
| ~US$ 257 mil | ~US$ 31 mil (12%) | ~US$ 79 mil |
| ~US$ 423 mil | ~US$ 46 mil (11%) | ~US$ 120 mil |
| ~US$ 608 mil | ~US$ 69 mil (12%) | ~US$ 193 mil |
| ~US$ 814 mil ou mais | ~US$ 103 mil (13%) | ~US$ 323 mil |

**Mensagem central:** o erro em dólares cresce com o preço do imóvel, mas o erro percentual fica
relativamente estável (9-13%) na maior parte do mercado — **exceto** em imóveis de luxo, onde é
maior (seção 2).

## 2. Onde confiar mais, onde confiar menos

Diga isso sem rodeio — é a informação mais acionável para quem for usar o modelo no dia a dia:

- **Confiança mais alta:** imóveis de padrão de entrada até premium (a maior parte do mercado) —
  erro percentual entre ~10% e ~19%.
- **Confiança mais baixa:** imóveis de luxo e o grupo com frente d'água / vista privilegiada — erro
  2 a 4 vezes maior em dólares. Isso não é um defeito escondido: é uma característica conhecida
  desse tipo de imóvel (menos transações comparáveis no histórico, mais variação de preço por
  motivos que não são só características físicas). O time investigou uma correção específica para
  isso, testou, e documentou por que não foi adotada — a correção parecia ajudar mas não passou no
  critério de qualidade definido antes do teste.
- **Recomendação de uso:** tratar a previsão como ponto de partida / triagem, com revisão humana
  obrigatória em imóveis de alto valor — não como preço final automático nesses casos.

## 3. A história que mais demonstra rigor (vale contar com calma)

Roteiro em 4 frases, funciona para qualquer público:

1. "Testamos uma técnica para deixar o modelo mais robusto a imóveis atípicos, com um critério
   definido **antes** de ver o resultado: só adotar se melhorasse pelo menos 2% numa validação
   interna. A técnica melhorou bem mais que isso — adotamos."
2. "Mais tarde, ao quebrar o erro por segmento em vez de olhar só a média, percebemos que essa mesma
   decisão piorava o erro num teste com dado geograficamente novo — mesmo tendo seguido a regra
   corretamente."
3. "Abrimos uma nova verificação, com um critério diferente e mais adequado a essa pergunta
   específica (medir generalização geográfica de verdade) — e revertemos a decisão."
4. "Não só trocamos um número: recalculamos todas as etapas seguintes do processo com a correção, e
   deixamos a decisão registrada de forma que ela nunca seja desfeita silenciosamente numa execução
   futura."

Se perguntarem "isso não é uma falha do projeto?" — resposta pronta: **não, é o processo de
melhoria funcionando como deveria.** A falha seria não checar depois, ou esconder o resultado ruim.
As duas decisões (a original e a correção) continuam registradas, nada foi apagado.

## 4. O "selo de confiança" por previsão

Além do preço previsto, o sistema entrega uma **nota de confiança de 0 a 100** e uma categoria — Alta,
Moderada, Baixa ou Muito Baixa (essa última, recomenda-se revisão manual). Em vez de um preço solto,
quem usa o sistema recebe preço + uma indicação de o quanto confiar nele, calibrada a partir do erro
histórico real de imóveis parecidos.

## 5. Gráficos recomendados para a apresentação

Prefira sempre visual a tabela de números quando o público for de negócio:

- **Previsto × real (dispersão):** mostra visualmente o quão perto as previsões ficam do valor real,
  com uma linha de "acerto perfeito" — mais convincente que qualquer métrica isolada.
- **Erro por segmento (barras):** um gráfico simples com o erro percentual médio por banda de preço,
  deixando claro onde o modelo é mais forte e mais fraco, na mesma lógica da tabela da seção 2.
- **Mapa/barra por região:** erro por zipcode ou região — útil se o público tiver familiaridade
  geográfica com o mercado local.
- Este projeto já tem essas visualizações prontas como dashboard interativo (previsto×real, erro por
  região/tipo de imóvel, banda de confiança) — usar ao vivo na apresentação é mais forte que
  screenshots estáticos.

## 6. O que já existe operacional, e o que ainda falta (não esconda isso — controla a expectativa)

Além do modelo em si, existe hoje uma API (FastAPI) + um portal de operação (Streamlit) reais, não só
o pipeline de treino — o time que for operar isso no dia a dia tem, sem precisar mexer em código:

- **Registro de modelo com rollback** — qualquer versão treinada (inclusive o modelo original) pode
  ser reativada a qualquer momento, com aprovação humana explícita em cada troca.
- **Retraining sob demanda ou automático** — com dado novo (venda real registrada, ou upload de
  planilha), sem atalho manual: reexecuta o mesmo pipeline de seleção usado no treino original. Um
  gatilho automático dispara o retraining sozinho a cada 500 vendas novas acumuladas (configurável) —
  mas nunca promove sozinho: todo candidato é comparado contra o modelo em produção por segmento e só
  fica marcado como "recomendado" se não piorar em nenhum deles; a decisão de ativar continua sempre
  humana.
- **Monitor de drift** — acompanha se a distribuição dos imóveis recebidos recentemente ainda parece
  com a do treino, por modelo ativo, com histórico consultável.
- **Nota de confiança por previsão** (seção 4) e Error Matrix por segmento, sempre disponíveis, não
  só num relatório pontual.

O que ainda falta:

- **Não há promoção automática de modelo para produção.** O sistema recomenda; a substituição real
  sempre exige aprovação humana explícita (Entregável 4).
- **CI/CD (automação de build/deploy) ainda não foi implementado** — a API e o portal já funcionam,
  com testes automatizados e Docker, mas o pipeline de implantação automática na nuvem é o próximo
  passo.
- **Shadow deployment e rollout canário gradual** (Entregável 4, seção 10) continuam só desenhados —
  a promoção hoje troca a versão ativa de uma vez, sem etapa intermediária de tráfego parcial.
- **O erro em imóveis de luxo é uma limitação conhecida**, não resolvida — decisão consciente de não
  perseguir esse segmento indefinidamente sem uma correção que realmente comprovasse ganho.

## 7. Perguntas de negócio prováveis (com resposta pronta)

| Pergunta | Resposta curta |
|---|---|
| O modelo está pronto para usar hoje? | Está recomendado, mas a decisão final de produção é humana — o ideal é começar com revisão humana obrigatória em imóveis de alto valor. |
| Quanto o modelo erra, na prática? | Em média ~14% do valor do imóvel; para a maior parte do mercado, entre 9% e 13%; em imóveis de luxo, pode passar de 17-19%. |
| Como sei em qual previsão confiar mais? | Cada previsão vem com uma nota de confiança de 0-100 e uma categoria (Alta/Moderada/Baixa/Muito Baixa) — abaixo de "Muito Baixa", recomenda-se revisão manual. |
| Por que não é mais preciso em imóveis caros? | Menos transações comparáveis e mais variação de preço nesse segmento — o time testou uma correção específica e não adotou porque não passou no critério definido antes do teste. |
| Isso substitui um avaliador humano? | Não — é uma ferramenta de apoio/triagem com faixa de erro conhecida, não um substituto de avaliação formal, especialmente em imóveis caros. |
| O que muda se tivermos mais dados no futuro? | O processo de aprendizado contínuo (Entregável 4) já está desenhado para isso — capturar vendas novas, reavaliar por segmento, promover só com aprovação humana. |

## 8. Armadilhas a evitar na apresentação

- Não venda o modelo como "preciso" — venda como "com erro conhecido e documentado por segmento",
  que é a força real do projeto.
- Não esconda a história da correção (seção 3) atrás de "ajustamos o modelo" — contada bem, é o
  ponto mais forte da apresentação.
- Não prometa que o modelo resolve o segmento de luxo — ele não resolve, e isso foi testado e
  documentado, não ignorado.
- Não diga que a API está "em produção" — está implementada e testada localmente, mas falta CI/CD.
