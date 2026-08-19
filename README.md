# Previsão de Preços de Casas — Seattle

Sistema de previsão de preço de imóveis residenciais, construído para generalizar para zipcodes nunca
vistos, com objetivo de medir onde ela falha e transformar essa incerteza em parte do produto.

## Onde começar a ler

0. **[`docs/PROJETO_VISAO_GERAL.md`](docs/PROJETO_VISAO_GERAL.md)** — visão geral completa do projeto:
   timeline de todas as fases com número real por decisão, fluxo de features, modelo final, deploy e
   aprendizado contínuo. **Ponto de entrada recomendado para avaliação técnica** — dá o quadro inteiro
   sem precisar abrir mais nada.
1. **[`docs/GUIA_EXECUCAO_NOTEBOOKS.md`](docs/GUIA_EXECUCAO_NOTEBOOKS.md)** — passo a passo pra rodar
   o projeto do zero só com notebooks (avaliação/reprodução), incluindo em qual fase a API já pode
   ser testada.
2. **[`docs/Modelo_Precificacao_Imoveis.pptx`](docs/Modelo_Precificacao_Imoveis.pptx)** — relatório visual para
   público de negócio (Entregável 5).

3. **`notebooks/`** — narrativa científica de cada fase (00-15, incl. 3 iterações do loop de hipóteses
   em 12), executados de ponta a ponta chamando o pipeline em processo (sem `subprocess`), com dados,
   gráficos e métricas reais inline.
4. **[`docs/notebooks/`](docs/notebooks/)** — um `.md` por notebook (objetivo, metodologia, decisões-chave,
   entradas/saídas, resultado).
5. **`reports/phase_reports/`** — relatório detalhado de cada fase, com as figuras geradas como
   evidência (inclui extensões exploratórias de nas fases 02/04/06/07/08/09).

## Os 4 entregáveis oficiais 

| Entregável | Onde está |
|---|---|
| 1. Entendimento dos dados | Documento: [`docs/Analise_Entendimento_Dados.docx`](docs/Analise_Entendimento_Dados.docx) Notebooks: fases 00-02, 06 (`notebooks/`, `reports/phase_reports/`) |
| 2. Variáveis importantes, modelo, generalização | Documento: [`docs/Desenvolvimento_Modelo_ML.docx`](docs/Desenvolvimento_Modelo_ML.docx) Notebooks: fases 03-13, 15 (Matriz de Confiança) |
| 3. Estratégia de deploy | Documento: [`docs/Estrategia_Deploy.docx`](docs/Estrategia_Deploy.docx) + [`app/`](app/README.md) (implementação, exceto CI/CD) |
| 4. Aprendizado contínuo | Documento: [`docs/Aprendizado_Continuo.docx`](docs/Aprendizado_Continuo.docx) + `/training`, `/model/promote`, `/feedback` em [`app/`](app/README.md) (implementação local) |
| 5. Comunicação com stakeholders | Documento: [`docs/Modelo_Precificacao_Imoveis.pptx`](docs/Modelo_Precificacao_Imoveis.pptx) |

## Estrutura

```
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

docs/                 # documentação técnica e entregáveis

app/                  # API de inferência FastAPI 
portal/               # Portal de inferência Streamlit```
---

```

## Início rápido

```bash
pip install -r requirements.txt
python src/scripts/validate_data.py            # fase 00
python src/scripts/build_dataset.py            # fase 01
python src/scripts/run_eda.py                  # fase 02
python src/scripts/create_split.py             # fase 03
python src/scripts/analyze_geographic_coverage.py  # fase 04
python src/scripts/run_augmentation_experiment.py  # fase 05
python src/scripts/run_segmentation.py         # fase 06
python src/scripts/generate_features.py        # fases 07-08
python src/scripts/register_features.py
python src/scripts/validate_features.py        # fase 09
python src/scripts/promote_features.py
python src/scripts/finalize_feature_set.py
python src/scripts/train_model.py              # fase 10
python src/scripts/evaluate_model.py           # fase 11
python src/scripts/finalize_model.py           # fase 13 — val tocado uma única vez
python src/scripts/write_production_contract.py  # fase 14
python src/scripts/predict.py
python src/scripts/build_confidence_matrix.py     # Matriz de Confiança (pós-plano) — docs/09_confidence_matrix.md
python src/scripts/run_hypothesis_drop_comps_knn_price.py  # loop de hipóteses, fase 12 iter3
python -m pytest tests/ -q
```


**Alternativa equivalente, notebook a notebook:** todo script acima expõe uma função `run_<fase>_pipeline()`
(mesma lógica, mesmos artefatos gravados) que os notebooks em `notebooks/` chamam diretamente — sem
`subprocess`, com os dados/gráficos/métricas reais de cada fase visíveis inline. Rodar
`notebooks/00_validate_data.ipynb` → `notebooks/15_confidence_matrix.ipynb` em ordem (via Jupyter ou
`python -m nbconvert --to notebook --execute --inplace notebooks/NN_*.ipynb`)  Ver
[`docs/notebooks/`](docs/notebooks/) para a documentação de cada notebook (objetivo, metodologia,
decisões, resultado).

## Resultado

Modelo final: XGBoost (`n_estimators=400, max_depth=3, learning_rate=0.05`), 20 features, refit em
`train`+`test`. **MAE em `val` (checagem única, sagrada) = \$72.517** (MAPE 14,1%). Erro 2-4x maior em
imóveis de luxo/waterfront — limitação conhecida e documentada.

## API de serving + Portal Inference (`app/`, `portal/`)

Camada de teste sobre o modelo final: API FastAPI + portal Streamlit para inferências. **CI/CD ainda não foi implementado nesta etapa**, por decisão explícita de escopo. Inclui uma **Matriz/Score
de Confiança** por predição (0-100, calibrada em TEST e verificada em VAL —
[`docs/09_confidence_matrix.md`](docs/09_confidence_matrix.md)).

```bash
pip install -r app/requirements.txt && pip install -r portal/requirements.txt
uvicorn app.main:app --reload --port 8000    # Swagger UI em http://localhost:8000/docs
streamlit run portal/Home.py                  # portal em http://localhost:8501
# ou, via Docker (API + portal juntos):
docker compose up --build
```

Documentação completa de cada processo: [`app/README.md`](app/README.md) (API — arquitetura, endpoints,
decisões de escopo), [`portal/README.md`](portal/README.md) (portal — páginas, autenticação) e
[`docs/DOCKER_GUIA.md`](docs/DOCKER_GUIA.md) (passo a passo de build/execução dos containers).

