# Fase 03 — canonical_split (Notebook: notebooks/03_canonical_split.ipynb)

## Objetivo
Como garantir que a avaliação reflita zipcode nunca visto?

## Metodologia / por quê
`run_split_pipeline` usa `src/validation/split.py::canonical_split`, que aplica `GroupShuffleSplit`
duas vezes (train vs. resto, depois test vs. val dentro do resto) agrupando por `zipcode`
(`configs/split.yaml`) — nunca um `train_test_split` aleatório por linha, porque isso deixaria o mesmo
zipcode aparecer em treino e teste e superestimaria a generalização geográfica (P1). Roda depois da
fase 02 (EDA completo sobre 100% dos dados) — decisões descritivas já foram tomadas sem depender do
split; aqui o split é fixado e passa a ser a base de todas as fases seguintes. A função também checa
explicitamente se alguma revenda (mesmo `id`, `date` diferente) acaba dividida entre partições
diferentes.

## Decisões-chave
- Split por grupo (`zipcode`), nunca aleatório por linha — validação geográfica é a principal (P1).
- Proporções alvo 70/15/15 (`train`/`test`/`val`), mas o resultado real varia em torno disso por causa
  do tamanho desigual dos zipcodes.
- Papel de cada partição fixado para o resto do roteiro: `train` ajusta o modelo (e recebe
  augmentation, se a fase 05 confirmar); `test` é o cheque repetível de zipcode não visto durante o
  desenvolvimento (fases 06-12), nunca decide sozinho entre candidatos; `val` só é tocado **uma única
  vez**, na fase 13.
- Checagem explícita de que revendas do mesmo imóvel nunca ficam divididas entre partições diferentes.

## Entradas
- `data/processed/house_clean.parquet`
- `configs/split.yaml`

## Saídas
- `data/processed/split_assignment.parquet`
- `data/processed/split_metadata.json`
- `reports/figures/03_split_distribution.png`

## Resultado
- `train`: 15.100 linhas (69,87%), 49 zipcodes.
- `test`: 2.644 linhas (12,23%), 10 zipcodes.
- `val`: 3.868 linhas (17,90%), 11 zipcodes.
- `group_overlap`: `[]` — nenhum zipcode aparece em mais de uma partição.
- `resales_spanning_splits`: `0` — nenhuma revenda dividida entre partições.

## Ver também
- Config: `configs/split.yaml`
- Relatório de fase: `reports/phase_reports/03_canonical_split.md`
