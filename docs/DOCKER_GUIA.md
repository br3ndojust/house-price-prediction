# Guia — Criar e Rodar os Containers (API + Portal)

Passo a passo para buildar e subir a [House Pricing API](../app/README.md) e o
[Portal Inference](../portal/README.md) via Docker. Ver `docker-compose.yml` (raiz) e
`app/Dockerfile`/`portal/Dockerfile`. **CI/CD não faz parte deste guia** — é build/execução local.

## 0. Pré-requisitos

- Docker Desktop instalado e **rodando** (`docker version` deve responder sem erro de conexão).
- Pipeline de ML já executado localmente pelo menos até a fase 14 — os containers **não geram** os
  artefatos, só os consomem via bind mount (`artifacts/` e `data/` não são versionados em git):

  ```bash
  pip install -r requirements.txt
  python scripts/validate_data.py
  python scripts/build_dataset.py
  python scripts/run_eda.py
  python scripts/create_split.py
  python scripts/analyze_geographic_coverage.py
  python scripts/run_augmentation_experiment.py
  python scripts/run_segmentation.py
  python scripts/generate_features.py
  python scripts/register_features.py
  python scripts/validate_features.py
  python scripts/promote_features.py
  python scripts/finalize_feature_set.py
  python scripts/train_model.py
  python scripts/evaluate_model.py
  python scripts/finalize_model.py
  python scripts/write_production_contract.py
  ```

  Confirme que existem `artifacts/model_final.pkl`, `artifacts/spatial_index.pkl`,
  `artifacts/property_cluster_model.pkl`, `artifacts/production_contract.yaml` e
  `data/trusted/features_contextual.parquet` antes de seguir.

## 1. (Opcional) configurar a API key

Por padrão os containers usam `change-me-dev-key`. Para trocar, crie um `.env` na raiz do repo (mesmo
diretório do `docker-compose.yml`):

```bash
echo "APP_API_KEY=minha-chave-segura" > .env
```

`docker compose` lê `.env` automaticamente e injeta a mesma chave na API (`APP_API_KEY`) e no portal
(`PORTAL_API_KEY`) — variável `${APP_API_KEY:-change-me-dev-key}` no `docker-compose.yml`.

## 2. Build + subir (caminho recomendado — `docker compose`)

Na raiz do repo:

```bash
docker compose up --build
```

O que acontece:

1. Builda `house-pricing-api:local` a partir de `app/Dockerfile` (contexto = raiz do repo, porque a
   API reusa `src/`, `scripts/`, `configs/`).
2. Builda `house-pricing-portal:local` a partir de `portal/Dockerfile`.
3. Sobe `api` (porta `8000`), espera o healthcheck (`GET /api/v1/health/ready`) ficar `healthy`.
4. Só então sobe `portal` (porta `8501`), já apontando para `http://api:8000` (rede interna do
   compose) — não precisa configurar URL manualmente.

Para rodar em background (sem prender o terminal):

```bash
docker compose up --build -d
```

## 3. Verificar que subiu

```bash
curl http://localhost:8000/api/v1/health/ready
# {"status":"ready","model_loaded":true}
```

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc
- Portal: http://localhost:8501
- Métricas Prometheus: http://localhost:8000/metrics

Teste rápido de predição:

```bash
curl -X POST http://localhost:8000/api/v1/predictions \
  -H "Content-Type: application/json" \
  -d '{"bedrooms":3,"bathrooms":2.0,"sqft_living":1800,"sqft_lot":5000,"floors":1.0,
       "waterfront":0,"view":0,"condition":3,"grade":7,"sqft_above":1800,"sqft_basement":0,
       "yr_built":1990,"yr_renovated":0,"zipcode":98042,"lat":47.6,"long":-122.3,
       "sqft_living15":1800,"sqft_lot15":5000}'
```

## 4. Acompanhar logs

```bash
docker compose logs -f api        # só a API
docker compose logs -f portal     # só o portal
docker compose logs -f            # os dois juntos
```

## 5. Parar / derrubar

```bash
docker compose stop               # para os containers, mantém volumes/imagens
docker compose down               # remove os containers (mantém o volume app_db, i.e. o SQLite)
docker compose down -v            # remove containers E o volume app_db (apaga o histórico de predições)
```

## 6. Rebuild depois de mudar código

`docker compose up` reusa camadas de imagem em cache. Depois de alterar algo em `app/` ou `portal/`:

```bash
docker compose up --build          # rebuilda só o que mudou (cache de camada do Docker)
docker compose build --no-cache api  # força rebuild total só da API, se precisar descartar o cache
```

Não precisa rebuildar para mudanças em `artifacts/`/`data/` (são bind mount, refletem na hora) nem
para promover um modelo novo (`POST /model/promote` recarrega em memória via `reload()`, sem restart).

## 7. Alternativa — build/run manual, sem `docker compose`

Útil para debugar um serviço isolado.

```bash
# API
docker build -t house-pricing-api:local -f app/Dockerfile .
docker run --rm -p 8000:8000 \
  -e APP_API_KEY=change-me-dev-key \
  -v "$(pwd)/artifacts:/app/artifacts" \
  -v "$(pwd)/data:/app/data" \
  -v app_db:/app/app_data \
  house-pricing-api:local

# Portal (em outro terminal, com a API já rodando)
docker build -t house-pricing-portal:local -f portal/Dockerfile .
docker run --rm -p 8501:8501 \
  -e PORTAL_API_BASE_URL=http://host.docker.internal:8000 \
  -e PORTAL_API_KEY=change-me-dev-key \
  house-pricing-portal:local
```

`host.docker.internal` é o hostname que o Docker Desktop expõe para o container alcançar o host —
necessário aqui porque, fora do `docker compose`, os dois containers não compartilham a mesma rede
nomeada automaticamente.

## 8. Solução de problemas

| Sintoma | Causa provável | Solução |
|---|---|---|
| `error during connect ... dockerDesktopLinuxEngine` | Docker Desktop não está rodando | Abra o Docker Desktop e espere ele inicializar antes de rodar `docker compose` |
| API sobe mas `health/ready` fica `not_ready` / container reinicia | `artifacts/model_final.pkl` (ou outro artefato) ausente no host | Rode o pipeline (passo 0) antes de subir os containers |
| `zipcodes em house sem demografia correspondente` ao chamar `/predictions` | zipcode do payload não existe em `data/raw/zipcode_demographics.csv` | Use um zipcode real do dataset (ex.: `98042`) |
| Portal não conecta na API | `PORTAL_API_BASE_URL` errado (ex.: `localhost` em vez de `api` dentro do compose) | Confirme que está usando `docker compose up` — a URL `http://api:8000` só funciona na rede interna do compose |
| Porta `8000`/`8501` já em uso | Outro processo local (ex.: `uvicorn`/`streamlit` rodando fora do Docker) na mesma porta | Pare o processo local ou mude o mapeamento de porta em `docker-compose.yml` (`"8001:8000"`) |
| Mudança em `app/`/`portal/` não aparece | Build usou cache de camada antiga | `docker compose up --build` (ou `build --no-cache` para forçar) |

## 9. Limpeza completa

```bash
docker compose down -v
docker image rm house-pricing-api:local house-pricing-portal:local
```
