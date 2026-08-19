# Feature Registry — schema

Recriado do zero (o schema de referência da iteração anterior não está disponível neste repositório;
estrutura de colunas replicada a partir da descrição em `roadmap/project_roadmap.md`, seção 3).

Arquivo: `data/processing/feature_registry.csv`

| coluna | tipo | descrição |
|---|---|---|
| `feature` | str | nome exato da coluna gerada |
| `layer` | enum | `RAW` \| `DERIVED` \| `CONTEXTUAL` |
| `family` | enum | mecanismo representado: `SIZE`, `QUALITY`, `LOCATION`, `ACCESSIBILITY`, `AGE`, `CONDITION`, `VIEW`, `WATERFRONT`, `NEIGHBORHOOD`, `RELATIVE_POSITION`, `COMPARABLE_MARKET`, `DEMOGRAPHICS`, `SEGMENT` |
| `source` | str | coluna(s) de origem |
| `target_derived` | bool | se depende de `price`/`price_log` (precisa fit-só-em-train) |
| `spatial` | bool | se usa `lat`/`long`/`zipcode`/geografia |
| `hypothesis` | str | mecanismo de formação de preço que a feature representa — obrigatório, preenchido **antes** da implementação (P2) |
| `phase` | str | fase do roadmap em que nasceu (ex: `05`, `06`) |
| `first_iteration` | str | identificador do experimento/commit onde entrou pela primeira vez |
| `status` | enum | `active` \| `candidate` \| `rejected` \| `legacy` |

Regra sem exceção (P2, blueprint seção 8): nenhuma linha nasce sem `hypothesis` preenchida.
