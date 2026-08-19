# Portal Inference (`portal/`)

App Streamlit externo para o engenheiro de MLOps acompanhar e agir sobre a
[House Pricing API](../app/README.md). Consome a API **só via HTTP** (`httpx`) — nunca importa `app/`
diretamente, roda como processo/container independente.

## Como rodar

### Local

```bash
pip install -r portal/requirements.txt
PORTAL_API_BASE_URL=http://localhost:8000 PORTAL_API_KEY=change-me-dev-key \
    streamlit run portal/Home.py
```

Acesse `http://localhost:8501`. `PORTAL_API_BASE_URL`/`PORTAL_API_KEY` são configuradas só por
variável de ambiente (sem UI de conexão na barra lateral, pedido explícito) — mude e reinicie o
processo/container do portal para trocar de API alvo.

Ocultar/mostrar a barra lateral usa o controle nativo do Streamlit (seta "«" no topo da própria
sidebar) — não um botão customizado. O botão nativo "Deploy" do Streamlit (canto superior direito) é
removido globalmente (`portal/lib/config.py::sidebar_client`), já que este portal não é implantado
pelo Streamlit Cloud.

### Docker

```bash
docker compose up --build
```

Sobe `api` + `portal` juntos (`docker-compose.yml`, raiz do repo) — o portal já aponta para
`http://api:8000` (rede interna do compose).

## Páginas

Portal reduzido a 3 telas — Home, Predições, Inferência em Lote (as demais telas de MLOps —
Importância de Features, Performance & Error Matrix, Drift, Treino, Registro de Modelo, Matriz de
Confiança — foram retiradas do portal a pedido; os endpoints correspondentes continuam existindo na
API, só não têm mais UI aqui).

| Página | O que mostra |
|---|---|
| **Home** (Visão geral) | health (status em português), **apelido da versão ativa** (se tiver, senão o id interno), uptime, contrato do modelo, lista de páginas com link clicável direto |
| **Predições** | preencher campo a campo OU colar uma linha inteira (ordem das colunas + valores, separados por vírgula) → registrar **valor real de venda** → comparativo previsto×registrado → preço/segmento previsto + Confidence Score + explicação SHAP local |
| **Inferência em Lote** | upload de CSV → checagem de colunas (com opção de informar a ordem manualmente) → predição em lote → detalhe por linha → gráficos por categoria → gráfico "o quanto o modelo acertou" com legenda, limiar de acerto ajustável e distribuição segmentada por Acerto/Erro (geral ou por categoria do imóvel) → registro de valor real em lote |

Nomenclatura: a tela de Predições chama `price_band` de **"Segmento do Imóvel"**; a tela de
Inferência em Lote usa **"Categoria do Imóvel"** especificamente (pedido do usuário). O cluster físico
(`property_cluster`) não aparece em nenhuma das duas — informação secundária que poluía a leitura
rápida do resultado principal.

## Predições e valor real (feedback)

A página **Predições** guarda a predição na sessão do navegador assim que o preço é calculado; o
formulário para registrar o valor real de venda (`POST /feedback`, precisa de API key) fica logo
**entre** o formulário de atributos e o resultado da predição. O valor registrado fica disponível pra
API usar em avaliação de performance e em retraining do modelo — sem tela própria no portal pra
disparar isso manualmente (ver seção seguinte sobre o gatilho automático).

## Gatilho automático de retraining — desligado

A API tem um pipeline semi-automático de retraining (gatilho por volume de feedback acumulado →
retraining → avaliação vs. modelo ativo → critério de promoção → **sempre** aprovação humana antes de
ativar, P6 — nunca promove sozinho). Esse gatilho automático está **desligado por padrão**
(`AutoRetrainConfig.enabled = False`, `app/domain/entities/auto_retrain_config.py`) — sem tela de
**Registro de modelo** no portal pra ligar/desligar ou ajustar o limite, a configuração só pode ser
lida/alterada direto pela API (`GET`/`PUT /api/v1/training/auto-retrain-config`).

## Inferência em lote (CSV)

Colunas obrigatórias: as mesmas 18 de `future_unseen_examples.csv` (`bedrooms`, `bathrooms`,
`sqft_living`, ..., `sqft_lot15` — ver `portal/lib/config.py::REQUIRED_CSV_COLUMNS`). O upload:

1. Lê o CSV pelo cabeçalho do próprio arquivo, por padrão. Se o arquivo não tiver cabeçalho (ou tiver
   nomes diferentes dos esperados), preencha o campo **"Ordem das colunas do CSV"** com os nomes
   corretos, na mesma ordem das colunas do arquivo, separados por vírgula.
2. Checa se todas as colunas obrigatórias estão presentes — se faltar alguma, mostra exatamente quais
   antes de tentar qualquer inferência.
3. Detecta automaticamente uma coluna extra de valor real (`price`, `actual_price`, `valor_real`, ...)
   — se encontrar, mostra um checkbox **"Registrar automaticamente como feedback ao concluir"**
   (marcado por padrão) logo acima do botão de rodar o lote; também dá pra escolher a coluna
   manualmente ou registrar depois via o botão "Registrar valores reais em lote" mais embaixo. Em
   qualquer um dos dois casos, os pares (imóvel, preço real) ficam disponíveis pra API usar em
   retraining, sem passo manual extra.
4. Valida tipo/valor de cada célula das colunas obrigatórias antes de chamar a API; linhas inválidas
   podem ser reportadas e (opcionalmente) ignoradas, mantendo as válidas.
5. Erros de validação de negócio (ex.: `grade` fora do intervalo 1-13) vêm da própria API — a regra
   não é duplicada no portal (P5) — e são reportados por linha do CSV original.
6. Resultado fica em `st.session_state` — clicar numa linha da tabela para ver o detalhe individual
   (mesma informação de uma predição feita na página Predições) não perde o restante da tela: o botão
   "Rodar inferência em lote" só *dispara* o cálculo, quem sustenta a tela depois é o estado salvo, não
   o valor momentâneo do botão (bug corrigido — antes, selecionar uma linha reexecutava o script e
   `st.stop()` apagava tudo, já que o clique do botão "resetava" na mesma passada).
7. Se houver coluna de valor real, a sessão "O quanto o modelo acertou" traz: gráfico previsto×real com
   legenda de cor (Acerto/Erro conforme um limiar de erro % ajustável por slider), contagem/percentual
   de acerto, e um seletor pra ver a distribuição do erro Geral, segmentada por Acerto/Erro, ou
   segmentada por Acerto/Erro × Categoria do Imóvel.

## Autenticação

O portal usa a mesma API key das rotas protegidas da API (`X-API-Key`) — nas telas atuais, isso cobre o
registro de valor real de venda (feedback), individual e em lote. Configurável só via
`PORTAL_API_KEY` (variável de ambiente do processo/container do portal).
