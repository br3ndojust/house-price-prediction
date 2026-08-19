# Fase 06 — market_property_segmentation (Notebook: notebooks/06_market_property_segmentation.ipynb)

## Objetivo
Como o mercado se segmenta por preço e por perfil físico?

## Metodologia / por quê
`train` vem de `data/trusted/train_for_pipeline.parquet` (decisão da fase 05 — original ou aumentado,
conforme o experimento de augmentation), nunca do split bruto diretamente; `test`/`val` vêm sempre do
split bruto, reais. Duas segmentações independentes são ajustadas só em `train`:
- **Bandas de preço**: quartis de `price_log` (`Entry`/`Standard`/`Premium`/`Luxury`), com breakpoints
  fixados em `train` e aplicados a todo o dataset.
- **Cluster de perfil físico**: k-means sobre features físicas (sem `price`/`lat`/`long`/`zipcode`, para
  não vazar o próprio alvo nem geografia no cluster), com varredura de `k` em `range(2, 7)` por
  silhouette para escolher o `k` final.
A hipótese (P3) é que os dois eixos capturam mecanismos distintos — banda de preço não deveria ser
redutível a "qual cluster físico o imóvel pertence".

## Decisões-chave
- Fit (quartis de preço e k-means) feito exclusivamente em `train`; aplicado depois a `test`/`val` via
  `apply_price_quartile`/`apply_property_cluster` (sem refit).
- `k` do cluster físico escolhido por silhouette máxima na varredura 2-6, não fixado a priori.
- Merge com o split sempre por `(id, date)`, nunca `id` isolado — `id` sozinho se repete em revendas
  (achado das fases 00/01).
- Pós fase 12: roda sobre o `train` sem augmentation (15.100 linhas) — a decisão de reverter a
  augmentation da fase 05 se propaga para cá automaticamente via
  `data/trusted/train_for_pipeline.parquet`.

## Entradas
- `data/processed/house_clean.parquet`
- `data/processed/split_assignment.parquet`
- `data/trusted/train_for_pipeline.parquet`

## Saídas
- `data/trusted/house_segments.parquet`
- `data/trusted/price_quartiles.json`
- `artifacts/property_cluster_model.pkl`
- `reports/segmentation_report.json`
- `reports/figures/06_segmentation.png`

## Resultado
- **Bandas de preço:** `Entry` 5.259, `Standard` 5.940, `Premium` 5.496, `Luxury` 4.917 imóveis.
- **Cluster físico, `best_k=3`** (silhouette máxima na varredura 2-6): cluster 0 — antigo/compacto,
  11.651 imóveis (`sqft_living` mediano 1.540, `grade` 7, idade mediana 58 anos); cluster 1 —
  moderno/grande, 9.798 imóveis (`sqft_living` mediano 2.530, `grade` 8, idade mediana 17 anos);
  cluster 2 — luxo físico raro, 163 imóveis (`sqft_living` mediano 2.850, `grade` 9, `waterfront`/`view`
  altos, idade mediana 54 anos).
- Números idênticos aos da execução anterior à reestruturação — esperado, já que sem augmentation
  `train` é exatamente o mesmo dataset.

## Ver também
- Relatório de fase: `reports/phase_reports/06_market_property_segmentation.md`
