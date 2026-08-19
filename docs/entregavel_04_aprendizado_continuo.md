# Entregável 4 — Aprendizado Contínuo

> Documento de estratégia — não é a implementação em si. O objetivo aqui é mostrar **como o modelo
> aprenderia com dados novos ao longo do tempo**, sem virar um processo manual nem um risco de piorar
> produção silenciosamente. A estratégia é o entregável; CI/CD e
> infraestrutura cloud não fazem parte da implementação.

## 1. Por que isso importa aqui especificamente

O preço de imóveis muda com o mercado (juros, oferta, novos bairros valorizando) — um modelo
treinado uma vez em 2026 degrada com o tempo se nunca vir dado novo. Ao mesmo tempo, o próprio
desenvolvimento deste projeto mostrou um risco real: uma decisão de modelagem melhorou a métrica
de treino usada para decidir, mas piorou o resultado em dado nunca visto — só foi percebido porque
a avaliação foi quebrada por segmento em vez de aceitar o número agregado (ver Entregável 5, seção
4, para a história completa). **O ciclo de aprendizado contínuo abaixo aplica a mesma disciplina em
produção**: nunca aceitar "melhorou em média" sem checar se algum segmento piorou.

## 2. Ciclo completo

```
Venda efetiva acontece (preço real de um imóvel já previsto)
        │
        ▼
Acúmulo de lote de dado novo rotulado (features + preço real)
        │  (gatilho: volume mínimo ex. 500 vendas novas, OU intervalo fixo ex. trimestral —
        │   o que vier primeiro)
        ▼
Retraining periódico (mesmo pipeline de treino usado no modelo atual — nunca um atalho manual)
        │
        ▼
Reavaliação completa — candidato novo vs. modelo em produção, por segmento (seção 4)
        │
        ├── degradou em algum segmento do SLA? ──sim──▶ rejeitado, mantém produção,
        │                                                registra o motivo, dado some p/ próximo ciclo
        │
        └── não degradou em nenhum segmento
                │
                ▼
        Shadow deployment (prevê em paralelo ao tráfego real, não serve ao cliente)
                │
                ▼
        Aprovação humana explícita (nunca automática)
                │
                ▼
        Rollout canário (5% → 25% → 100% do tráfego, com checagem por segmento em cada etapa)
                │
                ▼
        Rollback disponível a qualquer momento (versão anterior nunca é apagada do registry)
```

## 3. Captura do resultado real

Cada predição em produção já loga a banda de preço e o cluster físico previstos (Entregável 3,
seção 5). Quando a venda efetiva de um imóvel previsto se concretiza — via integração com o sistema
de transação ou registro público de venda — o par `(atributos do imóvel, preço real)` vira uma
linha de dado rotulado novo, no mesmo formato dos dados de treino originais. Não exige re-anotação
manual: o rótulo é o preço de venda que já existe naturalmente no processo de negócio.

## 4. Retraining — nunca um atalho manual

O candidato novo passa pelo **mesmo pipeline completo** que gerou o modelo atual (mesma limpeza,
mesma engenharia de features, mesmo processo de seleção de modelo) — nunca um retrain "rápido" via
notebook manual. Isso garante que o modelo novo é comparável ao atual em igualdade de condições, e
que nenhuma etapa de qualidade (ex: checagem de vazamento de dado, ablation de features) é pulada
só porque é um reforço, não um treino do zero.

## 5. Reavaliação — o critério que decide se o candidato é promovido

Esta é a parte mais importante do ciclo, e a que o desenvolvimento deste projeto mostrou na prática
ser necessária: **uma melhora agregada pode esconder uma piora séria num segmento específico.**

Critério de aceite:

- Validação cruzada por região geográfica (nunca split aleatório por linha — imóveis do mesmo
  bairro em treino e teste ao mesmo tempo superestimaria a capacidade real de generalização).
- Comparação de erro (Error Matrix) entre candidato e modelo em produção, quebrada por banda de
  preço, cluster físico, características especiais (waterfront, alto padrão) e região.
- **Promove só se o candidato não piorar o MAE em nenhum segmento do SLA (Entregável 3, seção 5) em
  mais de uma margem tolerável (ex: 2%)** — mesmo que a métrica agregada melhore. Uma melhoria
  agregada que esconde piora no segmento de luxo é **rejeitada**, não promovida.

## 6. Shadow deployment

Candidato aprovado na reavaliação roda em paralelo à produção, recebendo o mesmo tráfego real, mas
sem que a predição chegue ao cliente, por um período mínimo (ex: 2 semanas ou volume equivalente ao
usado na reavaliação offline). Isso confirma que o comportamento em dado de produção real bate com
a reavaliação feita offline antes de qualquer decisão de promoção — cobre o risco de o dado de
produção ter alguma característica que a reavaliação offline não capturou.

## 7. Promoção — sempre humana

O resultado do shadow deployment + a comparação por segmento é apresentado para aprovação humana
explícita. O sistema **recomenda** (o candidato sai marcado como pronto para promoção), mas nunca
substitui o modelo em produção sozinho. Isso é intencional: decisões que afetam preço de imóveis
para o negócio merecem um humano no loop, mesmo com todo o rigor estatístico anterior.

## 8. Rollout canário e rollback

Depois da aprovação: liberação gradual de tráfego (ex: 5% → 25% → 100%), com checagem de erro por
segmento em cada etapa antes de avançar para a próxima. Se algo piorar em qualquer etapa, o rollback
é imediato — a versão anterior do modelo nunca é sobrescrita no registry, então reverter é só trocar
qual versão está ativa, não reconstruir o modelo antigo do zero.

## 9. O que isso evita

- Modelo "congelado" no dia do treino original, cada vez mais desatualizado em relação ao mercado.
- Retraining virar um evento manual e arriscado, sem processo repetível.
- Uma melhora média mascarando uma piora concentrada num segmento de maior valor (exatamente o tipo
  de erro que este projeto encontrou e corrigiu durante o próprio desenvolvimento).
- Substituição de modelo em produção sem revisão humana.

## 10. Nota sobre a implementação real (API/portal)

Este documento descreve a **estratégia completa** — parte dela já está implementada de verdade em
`app/` (API) + `portal/` (Streamlit), parte continua só desenhada aqui. Honestidade sobre a diferença:

**Já implementado e operacional (API + portal):**
- Retraining sob demanda somando dado novo rotulado — predições com feedback registrado (seção 3
  deste documento) ou upload de CSV — reexecutando o mesmo pipeline de seleção de modelo (seção 4).
  Cada linha nova é sorteada aleatoriamente entre `train`/`test` (`test_size` configurável, padrão
  20% — nunca 100% em `train`), o mesmo tipo de partição usada no dataset original; `val` nunca
  recebe dado novo (P1, sagrado).
- **Gatilho automático por volume** (seção 2) — cada `POST /feedback` checa quanto feedback novo se
  acumulou desde o último disparo automático; ao cruzar um limite configurável (padrão 500), dispara
  sozinho um retraining com o dado novo acumulado. Configurável em `GET`/`PUT
  /training/auto-retrain-config`, ou na tela Registro de modelo do portal (seção "Configuração do
  gatilho automático"). Gatilho por **intervalo fixo** (ex.: trimestral) não está implementado — só o
  gatilho por volume.
- **Avaliação comparativa + critério de promoção objetivo** (seções 5 e 9 daqui) — todo candidato
  (automático ou manual) é reavaliado em `val` **e comparado contra o modelo ativo no mesmo `val`**,
  por segmento (`price_band`). `recommended_for_promotion` (visível em `GET /model/versions` e na
  tela Registro de modelo) é `True` só se o candidato não piorar o MAE em nenhum segmento além de uma
  margem tolerável (2% por padrão, `src/evaluation/promotion_criterion.py`) — a mesma disciplina da
  seção 5, agora automatizada. **A recomendação nunca decide sozinha**: candidatos não recomendados
  continuam visíveis e deployáveis manualmente, a palavra final é sempre humana (P6).
- Reavaliação por segmento em `val` a cada candidato (Error Matrix), nunca só a métrica agregada.
- Aprovação humana explícita antes de qualquer promoção (tela Registro de Modelo, botão "Fazer
  deployment") — inclusive rollback pra qualquer versão anterior, incluindo o modelo original.
- Monitor de drift (PSI por feature + drift de previsão), específico por versão de modelo ativa.
- Escolha manual de features e de métrica de seleção por job de retraining — mais granular do que o
  desenho original previa.
- Tela Registro de modelo tem um filtro pra ver só os modelos que o pipeline automático treinou
  (`triggered_by = "auto:volume_threshold"`), sem deixar de comparar/selecionar qualquer outra versão.

**Ainda só estratégia, não implementado:**
- Gatilho por **intervalo fixo** (ex.: trimestral) — só o gatilho por volume de dado novo existe de
  verdade.
- **Shadow deployment (seção 6) — decisão consciente de não implementar, não uma lacuna esquecida.**
  Shadow real exige infraestrutura de tráfego paralelo (duas versões do modelo recebendo a mesma
  requisição em produção, comparando resultado sem servir ao cliente) — este projeto é uma API/portal
  de instância única, local/Docker, sem volume de tráfego real nem infraestrutura de roteamento de
  tráfego. Fingir um "shadow" sem tráfego real seria simular honestidade, não ter ela. O que existe no
  lugar: avaliação offline comparativa em `val` (item acima) — mais fraca que shadow real (não
  confirma comportamento em dado de produção ao vivo), mas honesta sobre o que é.
- Rollout canário gradual (seção 8) — a promoção troca a versão ativa de uma vez, não em etapas de
  tráfego crescente.

Pergunta que costuma vir depois de mostrar isso: *"mas isso está realmente em produção?"* — resposta:
**a estratégia é o entregável.** Como extensão, foi implementada localmente uma versão real da
arquitetura em FastAPI + Streamlit + Docker, com o gatilho automático, a avaliação comparativa e o
critério de promoção funcionando de ponta a ponta. CI/CD e infraestrutura cloud não fazem parte da
implementação — isso é bem mais defensável do que tentar vender uma implementação local como produção.

Ver `app/README.md` e `portal/README.md` para o detalhe técnico de cada peça já implementada.
