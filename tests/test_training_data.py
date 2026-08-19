import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.features.comparable import fit_spatial_index
from src.pipelines.training_data import build_labeled_rows
from src.segmentation.property_clusters import PHYSICAL_PROFILE_COLUMNS, fit_property_cluster


def _toy_setup():
    rng = np.random.RandomState(0)
    n = 200
    train = pd.DataFrame({
        "lat": rng.uniform(47.3, 47.8, n),
        "long": rng.uniform(-122.5, -121.9, n),
        "zipcode": rng.choice([98101, 98102, 98103], n),
        "grade": rng.randint(4, 12, n),
        "price_log": rng.normal(13, 0.4, n),
        "sqft_living": rng.uniform(800, 3000, n),
        "sqft_lot": rng.uniform(2000, 10000, n),
        "bedrooms": rng.randint(1, 6, n),
        "bathrooms": rng.uniform(1, 4, n),
        "floors": rng.choice([1.0, 1.5, 2.0], n),
        "condition": rng.randint(1, 6, n),
        "view": rng.randint(0, 5, n),
        "waterfront": 0,
        "yr_built": rng.randint(1950, 2020, n),
    })
    train["property_age"] = 2026 - train["yr_built"]
    spatial_index = fit_spatial_index(train, k=10)
    scaler, km = fit_property_cluster(train, k=3, random_state=42)
    demographics = pd.DataFrame({"zipcode": [98101, 98102, 98103], "medn_hshld_incm_amt": [80000] * 3})
    price_breakpoints = np.quantile(train["price_log"], [0.25, 0.5, 0.75])
    return spatial_index, scaler, km, demographics, price_breakpoints


def _raw_row():
    return pd.DataFrame({
        "bedrooms": [3], "bathrooms": [2.0], "sqft_living": [1800], "sqft_lot": [5000],
        "floors": [1.0], "waterfront": [0], "view": [0], "condition": [3], "grade": [7],
        "sqft_above": [1800], "sqft_basement": [0], "yr_built": [1990], "yr_renovated": [0],
        "zipcode": [98101], "lat": [47.6], "long": [-122.3], "sqft_living15": [1800],
        "sqft_lot15": [5000],
    })


def test_build_labeled_rows_uses_real_price_as_label():
    spatial_index, scaler, km, demographics, breakpoints = _toy_setup()
    raw = _raw_row()
    out = build_labeled_rows(raw, np.array([450000.0]), demographics, spatial_index, breakpoints,
                              scaler, km, as_of_date=date(2026, 1, 1))
    assert out.loc[0, "price"] == 450000.0
    assert np.isclose(out.loc[0, "price_log"], np.log1p(450000.0))
    assert out.loc[0, "split"] == "train"
    assert out.loc[0, "price_band"] in {"Entry", "Standard", "Premium", "Luxury"}
    assert out.loc[0, "property_cluster"] in {0, 1, 2}


def test_build_labeled_rows_has_all_model_feature_columns():
    spatial_index, scaler, km, demographics, breakpoints = _toy_setup()
    raw = _raw_row()
    out = build_labeled_rows(raw, np.array([450000.0]), demographics, spatial_index, breakpoints,
                              scaler, km, as_of_date=date(2026, 1, 1))
    for col in ["property_age", "dist_to_seattle_center", "comps_knn_price", "local_grade_percentile"]:
        assert col in out.columns
    for col in PHYSICAL_PROFILE_COLUMNS:
        assert col in out.columns


def test_build_labeled_rows_multiple_properties():
    spatial_index, scaler, km, demographics, breakpoints = _toy_setup()
    raw = pd.concat([_raw_row(), _raw_row()], ignore_index=True)
    out = build_labeled_rows(raw, np.array([300000.0, 900000.0]), demographics, spatial_index,
                              breakpoints, scaler, km, as_of_date=date(2026, 1, 1))
    assert len(out) == 2
    assert list(out["price"]) == [300000.0, 900000.0]


def test_build_labeled_rows_test_size_controls_split_ratio():
    spatial_index, scaler, km, demographics, breakpoints = _toy_setup()
    raw = pd.concat([_raw_row()] * 20, ignore_index=True)
    prices = np.full(20, 450000.0)

    all_train = build_labeled_rows(raw, prices, demographics, spatial_index, breakpoints, scaler, km,
                                    as_of_date=date(2026, 1, 1), test_size=0.0)
    assert (all_train["split"] == "train").all()

    all_test = build_labeled_rows(raw, prices, demographics, spatial_index, breakpoints, scaler, km,
                                   as_of_date=date(2026, 1, 1), test_size=1.0)
    assert (all_test["split"] == "test").all()

    mixed = build_labeled_rows(raw, prices, demographics, spatial_index, breakpoints, scaler, km,
                                as_of_date=date(2026, 1, 1), test_size=0.5, random_state=7)
    assert set(mixed["split"].unique()) <= {"train", "test"}
    assert mixed["split"].eq("test").sum() > 0
    assert mixed["split"].eq("train").sum() > 0
