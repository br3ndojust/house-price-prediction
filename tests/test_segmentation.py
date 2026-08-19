import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.segmentation.price_bands import apply_price_quartile, fit_price_quartiles
from src.segmentation.property_clusters import (
    PHYSICAL_PROFILE_COLUMNS,
    add_property_age,
    apply_property_cluster,
    fit_property_cluster,
)


def test_fit_price_quartiles_uses_train_only():
    train = pd.Series(np.arange(100))
    bp = fit_price_quartiles(train)
    assert len(bp) == 3
    assert (bp[0] < bp[1] < bp[2])


def test_apply_price_quartile_assigns_four_labels():
    train = pd.Series(np.arange(100))
    bp = fit_price_quartiles(train)
    labels = apply_price_quartile(pd.Series(np.arange(100)), bp)
    assert set(labels.unique()) == {"Q1", "Q2", "Q3", "Q4"}


def test_apply_price_quartile_extrapolates_beyond_train_range():
    """Val/test podem ter price_log fora do range de train — não pode quebrar (P1: fit só train, apply
    em qualquer dado)."""
    train = pd.Series(np.arange(10, 20))
    bp = fit_price_quartiles(train)
    out_of_range = apply_price_quartile(pd.Series([0, 100]), bp)
    assert out_of_range.iloc[0] == "Q1"
    assert out_of_range.iloc[1] == "Q4"


def test_add_property_age():
    df = pd.DataFrame({"date": ["20150301T000000"], "yr_built": [2000]})
    out = add_property_age(df)
    assert out["property_age"].iloc[0] == 15


def _toy_physical_df(n=60, seed=0):
    rng = np.random.RandomState(seed)
    return pd.DataFrame({
        "sqft_living": rng.normal(1800, 400, n),
        "sqft_lot": rng.normal(6000, 1500, n),
        "bedrooms": rng.randint(1, 6, n),
        "bathrooms": rng.uniform(1, 4, n),
        "floors": rng.choice([1, 1.5, 2], n),
        "grade": rng.randint(4, 11, n),
        "condition": rng.randint(1, 6, n),
        "view": rng.randint(0, 5, n),
        "waterfront": rng.choice([0, 1], n, p=[0.95, 0.05]),
        "property_age": rng.randint(0, 100, n),
    })


def test_property_cluster_no_leakage_columns():
    assert not ({"price", "lat", "long", "zipcode"} & set(PHYSICAL_PROFILE_COLUMNS))


def test_fit_apply_property_cluster_roundtrip():
    train = _toy_physical_df()
    scaler, km = fit_property_cluster(train, k=3, random_state=42)
    labels = apply_property_cluster(train, scaler, km)
    assert set(labels) <= {0, 1, 2}
    assert len(labels) == len(train)
