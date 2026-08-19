"""Fase 16 (pós-fechamento) — constrói linhas de TREINO rotuladas a partir de dado novo real (venda
efetiva capturada via feedback, ou upload de lote rotulado) — docs/08_continuous_learning.md, seções
2-3 ("acúmulo de dado novo rotulado").

Reusa a MESMA engenharia de features de `src/pipelines/inference.py::predict_price` (P5) — a diferença
é que aqui o preço é conhecido (rótulo), não previsto: em vez de rodar o modelo, grava `price`/
`price_log` reais e classifica banda/cluster com os artefatos de segmentação já travados (nunca
redefine os cortes de banda/cluster a partir do dado novo — P6, contrato de produção é atômico).
"""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from src.data.merge import merge_demographics
from src.features.comparable import apply_comps_knn_price
from src.features.neighborhood import add_dist_to_seattle_center
from src.features.raw import add_property_age
from src.features.relative_position import apply_local_grade_percentile
from src.segmentation.price_bands import SEMANTIC_LABELS, apply_price_quartile
from src.segmentation.property_clusters import PHYSICAL_PROFILE_COLUMNS, apply_property_cluster


def build_labeled_rows(
    raw_df: pd.DataFrame,
    actual_prices: np.ndarray,
    demographics: pd.DataFrame,
    spatial_index,
    price_breakpoints: np.ndarray,
    cluster_scaler,
    cluster_kmeans,
    as_of_date: date | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
) -> pd.DataFrame:
    """`raw_df` no mesmo schema de `future_unseen_examples.csv` (sem id/date/price). Retorna linhas
    com `split` sorteado aleatoriamente entre `train`/`test` (proporção `test_size`) — nunca tudo em
    `train`, pra que dado novo também alimente a partição usada no refit final (fase 13), igual ao
    dataset original. `val` nunca recebe dado novo (P1: sagrado, tocado uma única vez)."""
    as_of = as_of_date or date.today()
    df = raw_df.copy().reset_index(drop=True)
    df["date"] = as_of.strftime("%Y%m%dT000000")
    df["price"] = np.asarray(actual_prices, dtype=float)
    df["price_log"] = np.log1p(df["price"])

    df = merge_demographics(df, demographics)
    df = add_property_age(df)
    df = add_dist_to_seattle_center(df)
    df["comps_knn_price"] = apply_comps_knn_price(df, spatial_index)
    df["local_grade_percentile"] = apply_local_grade_percentile(df, spatial_index)

    quartile = apply_price_quartile(df["price_log"], price_breakpoints)
    df["price_band"] = [SEMANTIC_LABELS[str(q)] for q in quartile]
    df["property_cluster"] = apply_property_cluster(df[PHYSICAL_PROFILE_COLUMNS], cluster_scaler, cluster_kmeans)

    rng = np.random.RandomState(random_state)
    is_test = rng.random_sample(len(df)) < test_size
    df["split"] = np.where(is_test, "test", "train")
    return df
