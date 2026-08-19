# Fase 05 — augmentation_experiment (Notebook: notebooks/05_augmentation_experiment.ipynb)

## Objetivo
Data augmentation melhora a robustez do `train` sem contaminar `test`/`val`?

## Metodologia / por quê
Só `train` é aumentado (P1); `test` é sempre o real, íntegro, usado como cheque de zipcode não visto
(relatado, não decide sozinho); `val` não entra aqui. Duas técnicas próprias (implementadas do zero, sem
depender do pacote `smogn` do PyPI nem de código de projeto anterior) são comparadas contra um baseline
sem augmentation:
- **Técnica A — SMOGN** (`src/augmentation/smogn.py`): super-representa a cauda do alvo (`price_log`
  muito alto/baixo) via interpolação SMOTE-like entre vizinhos raros, ou ruído gaussiano quando o
  vizinho mais próximo é "comum".
- **Técnica B — perturbação controlada** (`src/augmentation/perturbation.py`): varia `sqft_living`
  ±2-5%/`sqft_lot`±5%, mantém `grade`/`waterfront`/`view`/`zipcode` fixos, ajusta `price` por
  elasticidade local (`price_log ~ log(sqft_living)` em `train`) — não copia o preço original.

**Critério de aceite, definido antes do experimento (P4):** decisão principal por MAE médio em
`GroupKFold(3)/zipcode` dentro do `train` (aumentado ou não) — nunca por `test` isolado. Adota-se a
técnica vencedora só se reduzir o MAE de CV em ≥2% sobre o baseline (evita adotar ruído). O MAE em
`test` (geral e no subconjunto "atípico" identificado na fase 04) é reportado como evidência
complementar, não decisiva.

## Decisões-chave
- Regra de decisão pré-registrada (MAE de CV, limiar de 2%) para evitar mover a régua depois de ver o
  resultado.
- A variante **materializada** em `data/trusted/train_for_pipeline.parquet` segue
  `configs/augmentation.yaml` (`adopted`), uma config humana/versionada — não necessariamente o
  `cv_recommended` mecânico desta fase. Isso existe porque a fase 12 mostrou, com o pipeline completo,
  que o vencedor do critério de CV (Técnica B) piorava a generalização real em `test` (~+8,8%); a
  decisão foi revertida para `none` sem reescrever o histórico desta fase (P4/P6).
- O script sempre roda a comparação completa das 3 variantes (para fins de relatório/reavaliação
  futura), independentemente de qual seja a variante adotada.

## Entradas
- `configs/model.yaml`
- `data/processed/house_clean.parquet`
- `data/processed/split_assignment.parquet`
- `configs/augmentation.yaml`

## Saídas
- `reports/augmentation_experiment.json`
- `data/trusted/train_for_pipeline.parquet`
- `reports/figures/05_augmentation_comparison.png`

## Resultado
| Variante | n train | MAE CV (decisivo) | MAE test geral | MAE test atípico |
|---|---|---|---|---|
| Baseline (sem augmentation) | 15.100 | 107.515 ± 14.836 | 97.416 | 116.334 |
| Técnica A — SMOGN | 18.137 | 112.366 ± 7.243 (+4,5%, pior) | 96.830 | 117.190 |
| Técnica B — perturbação controlada | 22.650 | 95.276 ± 14.475 (-11,4%) | 97.397 | 116.144 |

Técnica A não atinge o critério de CV (rejeitada). Técnica B atinge o critério de CV
(`cv_recommended: technique_b_controlled_perturbation`) mas o ganho quase não aparece em `test`; a fase
12 confirmou piora real de generalização ao materializar essa técnica. **Decisão vigente
(`configs/augmentation.yaml`): `adopted: none`.** `reports/augmentation_experiment.json["adopted"]` =
`"baseline_no_augmentation"`, e `data/trusted/train_for_pipeline.parquet` contém as 15.100 linhas
originais, sem augmentation.

## Ver também
- Config: `configs/augmentation.yaml`
- Relatório de fase: `reports/phase_reports/05_augmentation_experiment.md`
