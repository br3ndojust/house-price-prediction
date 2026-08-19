# Fase 01 — data_understanding (Notebook: notebooks/01_data_understanding.ipynb)

## Objetivo
O que os dados representam e estão prontos para uso (limpeza + merge)?

## Metodologia / por quê
`run_dataset_build_pipeline` lê os 2 CSVs de imóveis e demografia, limpa (`src/data/clean.py`) e faz
merge (`src/data/merge.py`), gravando `data/processed/house_clean.parquet`. A limpeza remove apenas
linhas fisicamente implausíveis (erro de digitação), nunca outliers de mercado reais — a distinção é
explícita: um preço/área extremo mas fisicamente consistente permanece no dataset, só o que é
impossível fisicamente é removido. `price_log = log1p(price)` é adotado como alvo de modelagem porque
reduz a assimetria (skewness) da distribuição de forma verificável nos dados reais. O merge com
demografia usa `zipcode` só como chave de junção — nunca como feature categórica direta no modelo (P1).

## Decisões-chave
- 1 linha removida (`id=2402100895`, `bedrooms=33` com `sqft_living=1620`, ~49 sqft/quarto) — erro de
  digitação, não sinal de mercado real.
- 353 linhas de revenda mantidas (177 imóveis com `id` repetido, `date`/`price` diferentes) — cada
  venda é um evento de precificação independente, não uma duplicata a remover.
- `price_log = log1p(price)` adotado como alvo (skew bruto ≈4,02 → skew log ≈0,43).
- Merge 1:1 com `zipcode_demographics.csv`: 70/70 zipcodes cobertos nos dois lados, sem órfãos.

## Entradas
- `data/raw/kc_house_data.csv`
- `data/raw/zipcode_demographics.csv`

## Saídas
- `data/processed/house_clean.parquet`
- `reports/build_dataset_log.json`
- `reports/figures/01_price_log_transform.png`

## Resultado
- `data/processed/house_clean.parquet`: 21.612 linhas × 48 colunas (`n_rows_in=21613` → 1 linha
  removida).
- `n_resales=353` linhas de revenda mantidas como observações independentes.
- Skew de `price` ≈4,02 cai para ≈0,43 em `price_log` (histogramas comparativos gerados).

## Ver também
- Relatório de fase: `reports/phase_reports/01_data_understanding.md`
