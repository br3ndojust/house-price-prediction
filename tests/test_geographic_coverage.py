import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.evaluation.geographic_coverage import (
    apply_feature_space_distance,
    data_scarcity_buckets,
    distribution_comparison,
    feature_space_coverage,
    fit_feature_space_index,
    nearest_train_zip_distance,
    zip_representation_summary,
)


def test_zip_representation_summary_basic():
    df = pd.DataFrame({
        "zipcode": [1, 1, 1, 2],
        "price": [100.0, 200.0, 300.0, 400.0],
    })
    out = zip_representation_summary(df)
    row1 = out[out["zipcode"] == 1].iloc[0]
    assert row1["n"] == 3
    assert row1["median_price"] == 200.0
    row2 = out[out["zipcode"] == 2].iloc[0]
    assert row2["n"] == 1


def test_distribution_comparison_returns_per_split_rows():
    df = pd.DataFrame({
        "split": ["train"] * 5 + ["test"] * 5,
        "price_log": list(range(10)),
    })
    out = distribution_comparison(df, "split", ["price_log"])
    assert set(out["split"]) == {"train", "test"}
    assert (out["n"] == 5).all()


def test_nearest_train_zip_distance_zero_for_same_location():
    train = pd.DataFrame({"zipcode": [1, 2], "lat": [47.6, 47.7], "long": [-122.3, -122.2]})
    other = pd.DataFrame({"zipcode": [3], "lat": [47.6], "long": [-122.3]})
    out = nearest_train_zip_distance(train, other)
    assert np.isclose(out["distance_to_nearest_train_zip"].iloc[0], 0.0)
    assert out["nearest_train_zipcode"].iloc[0] == 1


def test_nearest_train_zip_distance_far_for_distant_location():
    train = pd.DataFrame({"zipcode": [1], "lat": [47.6], "long": [-122.3]})
    other = pd.DataFrame({"zipcode": [2], "lat": [50.0], "long": [-100.0]})
    out = nearest_train_zip_distance(train, other)
    assert out["distance_to_nearest_train_zip"].iloc[0] > 1.0


def test_feature_space_coverage_fits_on_train_only():
    train = pd.DataFrame({"id": [1, 2, 3], "x": [0.0, 1.0, 2.0]})
    other = pd.DataFrame({"id": [4], "x": [1.0]})
    out = feature_space_coverage(train, other, ["x"])
    assert "nearest_train_neighbor_distance" in out.columns
    assert len(out) == 1


def test_fit_feature_space_index_matches_one_shot_feature_space_coverage():
    """O índice cacheado (fit 1x, aplica N vezes) precisa dar o MESMO resultado do one-shot
    `feature_space_coverage` — é um refactor de performance, não deve mudar o número (P5/P4:
    `ConfidenceScorer` migrou pra essa API pra não refitar a árvore a cada predição de um lote)."""
    train = pd.DataFrame({"id": [1, 2, 3, 4], "x": [0.0, 1.0, 2.0, 10.0], "y": [0.0, 0.5, 1.0, 5.0]})
    other = pd.DataFrame({"id": [5, 6], "x": [1.2, 9.0], "y": [0.6, 4.5]})

    one_shot = feature_space_coverage(train, other, ["x", "y"])

    index = fit_feature_space_index(train, ["x", "y"])
    cached = apply_feature_space_distance(index, other)

    assert np.allclose(one_shot["nearest_train_neighbor_distance"].to_numpy(), cached)


def test_apply_feature_space_distance_reused_across_multiple_single_row_calls():
    """Consultar linha a linha (como um lote de predições faz) precisa bater com consultar tudo de
    uma vez — a árvore é fitada uma única vez e reusada."""
    train = pd.DataFrame({"x": [0.0, 5.0, 10.0]})
    other = pd.DataFrame({"x": [1.0, 4.0, 11.0]})
    index = fit_feature_space_index(train, ["x"])

    batch_distances = apply_feature_space_distance(index, other)
    row_by_row = [apply_feature_space_distance(index, other.iloc[[i]])[0] for i in range(len(other))]

    assert np.allclose(batch_distances, row_by_row)


def test_data_scarcity_buckets_classification():
    train = pd.DataFrame({"zipcode": [1] * 50 + [2] * 200 + [3] * 600})
    out = data_scarcity_buckets(train)
    b = out.set_index("zipcode")["scarcity_bucket"]
    assert b[1] == "LOW"
    assert b[2] == "MEDIUM"
    assert b[3] == "HIGH"
