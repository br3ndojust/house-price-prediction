# House Pricing API (`app/`)

API FastAPI que coloca `artifacts/model_final.pkl` (fase 13 do pipeline de ML, MAE em `val` **US$
72.517**) atrás de uma camada de inferência, monitoramento e retreinamento — cobrindo o que
[`docs/07_deploy_strategy.md`](../docs/07_deploy_strategy.md) e
[`docs/08_continuous_learning.md`](../docs/08_continuous_learning.md) desenharam como documento.

## Sumário

- [Como rodar](#como-rodar)
- [Arquitetura](#arquitetura--clean-architecture)
- [Endpoints](#endpoints-apiv1)
- [Persistência (SQLite)](#persistência-sqlite)
- [Swagger / OpenAPI](#swagger--openapi)
- [Observabilidade](#observabilidade)
- [Decisões de escopo e honestidade técnica](#decisões-de-escopo-e-honestidade-técnica)
- [Testes](#testes)

## Como rodar

### Local (sem Docker)

Pré-requisito: o pipeline de ML já rodado localmente pelo menos até a fase 14 (`artifacts/model_final.pkl`,
`artifacts/spatial_index.pkl`, `artifacts/property_cluster_model.pkl`, `artifacts/production_contract.yaml`,
`data/trusted/*`, `data/raw/zipcode_demographics.csv` — ver README raiz, seção "Início rápido"). Esses
arquivos **não são versionados em git** (gerados localmente pelo pipeline).

```bash
pip install -r app/requirements.txt
python scripts/build_confidence_matrix.py   # opcional — habilita confidence_score/category nas respostas
uvicorn app.main:app --reload --port 8000
```

Acesse `http://localhost:8000/docs` (Swagger UI). API key padrão de desenvolvimento:
`change-me-dev-key` (ver `app/core/config.py`, sobrescrever via `.env`/`APP_API_KEY`). Sem o passo
`build_confidence_matrix.py`, a API sobe normalmente — `confidence_score`/`confidence_category` só
ficam `null` até a calibração existir.

### Docker (API + Portal)

```bash
docker compose up --build
```

Sobe `api` (porta 8000) e `portal` (porta 8501). `artifacts/` e `data/` são montados como bind mount
(o container lê os artefatos já gerados localmente, nunca os reconstrói na imagem); o SQLite persiste
num volume nomeado (`app_db`) entre restarts. Passo a passo completo (pré-requisitos, verificação,
logs, troubleshooting, build manual sem compose): [`docs/DOCKER_GUIA.md`](../docs/DOCKER_GUIA.md).

## Arquitetura 

```
app/
├── domain/            # entidades + value objects + ports (Protocols) — zero import de framework
├── application/       # use cases (1 classe por caso de uso) — orquestra ports, sem FastAPI/SQLAlchemy
├── infrastructure/     # adapters: model (artifacts), persistence (SQLite), training, drift, explainability
├── api/v1/             # routers + schemas Pydantic — só aqui HTTP existe
├── core/               # config (pydantic-settings), security (API key), container (composition root)
├── tests/               # unit (domain/application/infrastructure) + integration (routers via TestClient)
└── main.py             # app factory
```

Regra de dependência (P5, reprodutibilidade/isolamento): `domain` não depende de nada; `application`
depende só de `domain`; `infrastructure` implementa os `ports`; `api` depende de `application`. Nunca o
contrário — `core/container.py` é o único lugar onde a `api` conhece `infrastructure` concretamente.

**Reuso do pipeline de ML (P5 — nenhuma lógica duplicada):**

| Adapter | Reusa de `src/scripts/` |
|---|---|
| `infrastructure/model/artifact_model_repository.py` | `src/pipelines/inference.py::predict_price` (mesma função da demo da fase 14) |
| `infrastructure/training/pipeline_training_service.py` | `src/scripts/train_model.py::run_model_selection_pipeline` + `src/scripts/finalize_model.py::run_final_refit_pipeline` (refatoradas em funções chamáveis para este propósito) |
| `infrastructure/drift/feature_drift_evaluator.py` | `src/evaluation/geographic_coverage.py::feature_space_coverage` (fase 04) |
| `infrastructure/explainability/shap_explainer.py` | `shap.TreeExplainer` sobre o XGBoost ativo (nova dependência) |
| `app/application/use_cases/get_error_matrix.py` | `src/evaluation/error_matrix.py` (fase 09/11) |
| `infrastructure/confidence/confidence_scorer.py` | `src/evaluation/confidence.py` (Matriz de Confiança — ver [`docs/09_confidence_matrix.md`](../docs/09_confidence_matrix.md)) |

## Endpoints (`/api/v1`)

| Grupo | Endpoint | Método | Auth | Descrição |
|---|---|---|---|---|
| Inferência | `/predictions` | POST | não | 1 imóvel → preço + banda + cluster |
| Inferência | `/predictions/batch` | POST | não | lote de imóveis |
| Health | `/health/live` `/health/ready` `/health/detailed` | GET | não | liveness/readiness/detalhe |
| Performance | `/performance/summary` | GET | não | volume, distribuição por banda/cluster (`model_version` opcional filtra por versão) |
| Performance | `/performance/error-matrix` | GET | não | Error Matrix real sobre predições com feedback (`model_version` opcional) |
| Performance | `/performance/predictions` | GET | não | lista predições recentes (`with_feedback=true` filtra só as que já têm valor real; `model_version` opcional) |
| Performance | `/performance/model-versions` | GET | não | versões de modelo com pelo menos 1 predição registrada |
| Feedback | `/feedback` | POST | **sim** | preço de venda real de uma predição — também checa o gatilho automático de retraining (P6: nunca promove sozinho) |
| Modelo | `/model/info` `/model/versions` | GET | não | contrato ativo / histórico de versões (`cv_mae`/`test_mae`/`val_mae`, `performance_metrics` — RMSE/MAE/MAPE/R² por TRAIN/TEST/VAL, `n_features`, `feature_cols`, `recommended_for_promotion`, `promotion_eval`) |
| Modelo | `/model/promote` | POST | **sim** | aprova e ativa um candidato (P6) |
| Treino | `/training/jobs` | POST | **sim** | dispara retraining em background (dataset oficial, sem dado novo) |
| Treino | `/training/jobs/from-feedback` | POST | **sim** | retraining somando as predições que já têm valor real registrado (fase 16) |
| Treino | `/training/jobs/from-upload` | POST (multipart) | **sim** | retraining somando um CSV de dado novo rotulado (fase 16) |
| Treino | `/training/jobs` `/training/jobs/{id}` | GET | não | histórico / status de um job |
| Treino | `/training/auto-retrain-config` | GET | não | limite/status do gatilho automático de retraining + acumulado desde o último disparo |
| Treino | `/training/auto-retrain-config` | PUT | **sim** | ajusta o limite (padrão 500) e liga/desliga o gatilho automático |
| Drift | `/drift/evaluate` | POST | **sim** | computa relatório de drift novo (PSI por feature + Prediction Drift) |
| Drift | `/drift/reports` `/drift/reports/latest` | GET | não | histórico / mais recente (`model_version` = versão ativa quando o relatório foi gerado) |
| Explicabilidade | `/model/feature-importance` | GET | não | ranking global (gain-based) |
| Explicabilidade | `/model/feature-statistics` | GET | não | correlação com `price` + quartis de cada feature oficial (TRAIN) |
| Explicabilidade | `/model/feature-sample` | GET | não | amostra de TRAIN (feature+`price`) para gráfico de dispersão |
| Explicabilidade | `/model/feature-snapshots` | GET | não | modelos registrados com snapshot de importância/correlação gravado |
| Explicabilidade | `/model/feature-snapshots/{version}` | GET | não | snapshot completo (importância+correlação+quartis) de um modelo específico |
| Explicabilidade | `/predictions/explain` | POST | não | SHAP local para uma predição |
| Confiança | `/predictions/confidence` | POST | não | Confidence Score (0-100) + breakdown completo — [`docs/09_confidence_matrix.md`](../docs/09_confidence_matrix.md) |
| Confiança | `/model/confidence-calibration` | GET | não | metodologia + cortes de categoria calibrados em TEST |
| Observabilidade | `/metrics` (raiz) | GET | não | Prometheus |

Auth: header `X-API-Key` (ver `app/core/security.py`). `POST /predictions` e `/predictions/batch`
também retornam `confidence_score`/`confidence_category` diretamente na resposta (null se a
calibração ainda não foi gerada).

**Seleção de features no retreino:** os três endpoints `/training/jobs`, `/training/jobs/from-feedback`
e `/training/jobs/from-upload` aceitam um `feature_cols` opcional (JSON `{"feature_cols": [...]}` nos
dois primeiros; campo de formulário `feature_cols` separado por vírgula no upload) — subconjunto das
20 features oficiais (`GET /model/feature-importance` lista as válidas). Omitido = usa as 20 oficiais;
informado e válido = candidato experimental com esse subconjunto (nunca reescreve
`data/trusted/feature_metadata.json`, é só mais uma versão pra avaliar/promover como qualquer outra,
P6); nome desconhecido = 422 com a lista de features válidas. `GET /model/versions` reporta o
`feature_cols`/`n_features` real de cada versão (inclusive o modelo original, seedado com as 20
oficiais) — assim uma feature nunca "some" das opções de retreino só porque um candidato anterior foi
treinado com um subconjunto menor.

**Métrica de seleção no retreino:** os mesmos três endpoints também aceitam um `metric` opcional
(`mae` | `rmse` | `mape` | `r2` — ver `src/models/train.py::METRIC_KEYS`). Decide só qual candidato
(Ridge/XGBoost, cada combinação de hiperparâmetros) este job elege como vencedor da seleção via
`GroupKFold`; **nunca** muda o critério oficial de promoção (P4: continua MAE em `val`, checado uma
única vez). Omitido/`"mae"` = comportamento padrão/oficial; métrica desconhecida = 422.

## Persistência (SQLite)

`predictions` (inclui `confidence_score`/`confidence_category`), `feedback` (FK), `training_jobs`,
`drift_reports` (inclui `prediction_drift_json` e `model_version` — qual versão estava ativa quando o
relatório foi gerado), `model_versions` (inclui `n_train_rows`, `n_features`, `feature_cols`,
`triggered_by`, `recommended_for_promotion` e `promotion_eval` — avaliação comparativa contra o modelo
ativo, ver seção acima), `feature_snapshots` (importância gain-based + correlação com `price` + quartis
por feature, gravado uma vez na criação de cada `ModelVersion` — `FeatureSnapshotService`, P4: nunca
recalculado depois, histórico estável mesmo que o `.pkl` do candidato seja removido), `auto_retrain_config`
(linha única — limite/status do gatilho automático de retraining, `SqlAutoRetrainConfigRepository`) —
criadas via `Base.metadata.create_all()` no
startup (**sem Alembic** — simplificação consciente e documentada, coerente com SQLite/escopo do
exercício). `create_all()` só cria tabelas que não existem, nunca adiciona coluna a uma tabela já
existente — um `app_data/app.db` (ou volume Docker `app_db`) de uma execução anterior a essas colunas
precisa ser apagado em dev (`rm app_data/app.db` local, `docker compose down -v` no Docker) para pegar
o schema novo; um schema change real em produção precisaria de migração de verdade. Exceção: a linha
`model_final` (seed do modelo original) se auto-corrige sozinha a cada startup —
`Container._seed_baseline_model_version()` sempre recalcula os valores e faz backfill dos campos que
estiverem `NULL` numa linha já existente (`SqlModelVersionRepository.backfill_missing_fields`), sem
sobrescrever `status`/`notes` reais — não precisa apagar o banco só pra essa linha específica pegar um
campo novo.

## Swagger / OpenAPI

- **Interativo:** `GET /docs` (Swagger UI) e `GET /redoc` (ReDoc) — servidos nativamente pelo FastAPI,
  gerados a partir dos `response_model`/docstrings/tags de cada router.
- **Estático:** `python src/scripts/export_openapi.py` gera `docs/api/openapi.json` (útil para revisar o
  contrato em PR sem subir o servidor).

## Observabilidade

- `GET /metrics` — Prometheus (`http_requests_total`, `http_request_duration_seconds`,
  `predictions_total`, `drift_status`), via `prometheus-client`.
- Logging estruturado JSON (`app/infrastructure/observability/logging_config.py`) com `request_id` por
  requisição (header `X-Request-ID` + middleware em `main.py`) — cada linha de requisição já mostra
  método, rota, status e duração (`POST /api/v1/predictions/batch -> 200 (16234ms)`); jobs de treino
  logam início/fim de cada fase (seleção de modelo, refit) e o resultado final.

### Acompanhar logs pelo terminal (sem UI dedicada — funciona hoje)

```bash
docker compose logs -f api        # inferência, treino, drift — tudo que passa pela API
docker compose logs -f portal     # o portal em si (raramente necessário)
docker compose logs -f            # os dois juntos

# sem Docker (uvicorn local): a saída já vai pro stdout do terminal onde rodou `uvicorn app.main:app`

# filtrar só as linhas de um job de treino específico:
docker compose logs api | grep "training job <id>"
```

Não tem viewer de log dentro do portal — se isso virar necessário (volume grande, precisar filtrar por
`request_id` interativamente), o caminho natural é agregar os logs num serviço externo (Loki/ELK) e
linkar do portal, não reimplementar um `tail -f` na UI.

## Decisões de escopo e honestidade técnica

Seguindo a cultura de honestidade do projeto (P4), alguns pontos onde esta implementação simplifica
conscientemente o desenho de `docs/07`/`docs/08`, documentados aqui em vez de escondidos:

- **CI/CD não implementado nesta etapa** (pedido explícito do usuário) — `docs/02_execution_plan_api.md`
  descreve `ci.yml`/`cd.yml`; nenhum dos dois existe em `.github/workflows/` ainda.
- **Retraining (`POST /training/jobs`)** reexecuta as fases 10 (seleção de modelo) + **11 (avaliação
  em `test`, com o modelo ainda fitado só em `train`)** + 13 (refit + checagem em `val`) sobre
  `data/trusted/features_contextual.parquet` já travado — o mesmo pipeline do treino original. A fase
  11 roda `src/scripts/evaluate_model.py::run_test_evaluation_pipeline` **antes** da fase 13, porque a
  fase 13 refita em `train`+`test` combinados e sobrescreve o mesmo artefato — depois disso, avaliar
  em `test` deixaria de ser holdout. `cv_mae` (fase 10, TRAIN)/`test_mae` (fase 11)/`val_mae` (fase
  13) juntos em `ModelVersion` dão o trio treino/teste/val pra sinalizar overfitting (TRAIN bem melhor
  que TEST/VAL) ou underfitting (os três ruins) — exposto em `GET /model/versions`, sem tela própria no
  portal desde a redução a 3 páginas (`docs/PROJETO_VISAO_GERAL.md`, seção 12.9).
  `performance_metrics` (JSON, `ModelVersion`) tem o quadro completo — RMSE, MAE, MAPE e R² (espaço
  `price_log`) pra cada uma das três partições — `r2_log_score` (`src/evaluation/error_matrix.py`) é a
  única fonte dessa fórmula, reusada por
  TRAIN/TEST/VAL pra manter os três comparáveis.
  Fora do fluxo experimental original, isso significa que `val` deixa de ser "tocado uma única vez"
  no sentido estrito das fases 00-14: cada novo candidato reavalia `val`, igual a qualquer pipeline de
  reavaliação contínua real — mas nenhuma decisão de feature/modelo/hipótese usa esse `val`, só mede o
  candidato já travado (ver docstring de `src/scripts/finalize_model.py::run_final_refit_pipeline`).
- **Acumulação de dado novo rotulado (fase 16, `docs/08_continuous_learning.md` seções 2-3) — agora
  implementada:** `POST /training/jobs/from-feedback` soma ao dataset oficial as predições que já
  receberam `POST /feedback` (venda real capturada); `POST /training/jobs/from-upload` soma um lote
  novo enviado via CSV. Os dois reusam a mesma engenharia de features de
  `src/pipelines/inference.py` através de `src/pipelines/training_data.py::build_labeled_rows` (P5) —
  banda de preço e cluster físico do dado novo são classificados com os cortes **já travados** (nunca
  redefinidos a partir do dado novo, P6). O dado combinado nunca sobrescreve
  `data/trusted/features_contextual.parquet` — fica num parquet temporário só daquela execução
  (`app_data/train_augmented_<job_id>.parquet`). Cada linha nova recebe `split` sorteado
  aleatoriamente entre `train`/`test` (`test_size` opcional nos dois endpoints, padrão 0.2) — nunca
  100% em `train` como na primeira versão dessa funcionalidade; `val` nunca recebe dado novo (P1).
- **Latência** é exposta via `/metrics` (Prometheus, histogram), não duplicada em
  `/performance/summary` (que reporta volume/distribuição via SQLite).
- **SHAP local** só suporta modelos baseados em árvore (`shap.TreeExplainer`) — o modelo de produção
  atual é sempre XGBoost, mas se um retraining futuro promover um candidato Ridge, `/predictions/explain`
  retorna erro 409 explícito em vez de um resultado incorreto.
- **Conversão de SHAP para dólar é aproximada** (`expm1` no ponto de referência, não perfeitamente
  aditiva) — toda resposta de `/predictions/explain` inclui a ressalva no campo `approx_dollar_note`.
- **Bug de performance encontrado e corrigido: timeout em lotes grandes (`POST /predictions/batch`
  com >1000 imóveis).** Causa raiz: `ConfidenceScorer` refitava a árvore de vizinhos mais próximos
  (`NearestNeighbors` sobre ~15 mil linhas de TRAIN) **a cada imóvel do lote**, em vez de uma vez por
  processo — `src/evaluation/geographic_coverage.py` agora expõe `fit_feature_space_index`/
  `apply_feature_space_distance` (fit 1x, consulta N vezes), e `ConfidenceScorer` cacheia o índice na
  carga. SQLite também passou para `PRAGMA journal_mode=WAL` + `synchronous=NORMAL` (cada predição do
  lote fazia um commit individual; em modo `DELETE` — o padrão — isso sozinho já custava a maior parte
  do tempo em lotes de centenas de linhas). Resultado: 500 imóveis foi de ~23s para ~16s por chamada;
  o timeout do cliente (portal) para `/predictions/batch` e `/training/jobs/from-upload` também subiu
  de 15s (padrão) para 120s. **Limite conhecido, não escondido:** cada linha do lote ainda passa pelo
  pipeline de inferência/confiança um por vez (`PredictBatch` chama `PredictPrice.execute()` em loop,
  não vetorizado) — ~30-85ms/linha; lotes de muitos milhares de linhas continuam lentos (minutos), só
  não estouram mais o timeout do cliente. Vetorizar `PredictBatch` de ponta a ponta (1 chamada de
  modelo/KNN para o lote inteiro, não N chamadas) é o próximo ganho de performance óbvio, se
  necessário.
- **`POST /model/promote` agora falha com 404 claro (não 500 cru)** quando o `.pkl` do candidato não
  existe mais em disco — o registro em `model_versions` sobrevive no SQLite (persistido em volume),
  mas o arquivo em `artifacts/candidates/` é solto e desaparece se alguém limpar aquele diretório
  (aconteceu de verdade nesta sessão — `artifacts/` é bind mount compartilhado entre o host e o
  container; um `rm -rf artifacts/candidates/` rodado no host durante teste local apagou um candidato
  que uma instância Docker rodando em paralelo tinha acabado de treinar). `PromoteModel` agora checa a
  existência do arquivo **antes** de marcar a versão como `active`, então o estado no banco nunca fica
  inconsistente com o disco.
- **Matriz de Confiança:** os cortes de categoria (0-100) são calibrados em TEST e cada categoria
  respeitou seu teto de erro prometido quando verificada em VAL — mas a ORDEM fina das 4 categorias
  não ficou perfeitamente monotônica em VAL (achado documentado, não escondido). Ver
  [`docs/09_confidence_matrix.md`](../docs/09_confidence_matrix.md), seção "Limitação observada", para
  a leitura completa e a recomendação de uso (confiar no teto de erro + na matriz 2×2, tratar a ordem
  fina como indicativa).

## Testes

```bash
pytest app/tests -q                     # unit + integration (usa SQLite temporário isolado, nunca app_data/ real)
pytest app/tests/unit -q                # só unit (rápido, sem I/O de rede)
pytest app/tests/integration -q         # integration via TestClient, artefatos reais (precisa do pipeline já rodado)
python src/scripts/export_openapi.py        # gera docs/api/openapi.json
```
