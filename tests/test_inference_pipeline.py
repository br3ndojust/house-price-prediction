import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.features.comparable import fit_spatial_index
from src.models.train import build_model
from src.pipelines.inference import predict_price


def _toy_setup():
    rng = np.random.RandomState(0)
    n = 200
    train = pd.DataFrame({
        "lat": rng.uniform(47.3, 47.8, n),
        "long": rng.uniform(-122.5, -121.9, n),
        "zipcode": rng.choice([98101, 98102, 98103], n),
        "grade": rng.randint(4, 12, n),
        "price_log": rng.normal(13, 0.4, n),
        "dist_to_seattle_center": rng.uniform(0, 1, n),
        "comps_knn_price": rng.normal(13, 0.3, n),
        "local_grade_percentile": rng.uniform(0, 1, n),
        "property_age": rng.randint(0, 80, n),
    })
    feature_cols = ["grade", "dist_to_seattle_center", "comps_knn_price", "local_grade_percentile",
                    "property_age"]
    model = build_model("ridge", {"alpha": 1.0})
    model.fit(train[feature_cols], train["price_log"])
    model_bundle = {"model": model, "feature_cols": feature_cols, "target": "price_log"}
    spatial_index = fit_spatial_index(train, k=10)
    demographics = pd.DataFrame({"zipcode": [98101, 98102, 98103], "medn_hshld_incm_amt": [80000] * 3})
    return model_bundle, spatial_index, demographics


def test_predict_price_returns_positive_dollar_values():
    model_bundle, spatial_index, demographics = _toy_setup()
    raw = pd.DataFrame({
        "bedrooms": [3], "bathrooms": [2.0], "sqft_living": [1800], "sqft_lot": [5000],
        "floors": [1.0], "waterfront": [0], "view": [0], "condition": [3], "grade": [7],
        "sqft_above": [1800], "sqft_basement": [0], "yr_built": [1990], "yr_renovated": [0],
        "zipcode": [98101], "lat": [47.6], "long": [-122.3], "sqft_living15": [1800],
        "sqft_lot15": [5000],
    })
    preds = predict_price(raw, model_bundle, spatial_index, demographics, as_of_date=date(2026, 1, 1))
    assert len(preds) == 1
    assert preds[0] > 0


def test_predict_price_deterministic():
    model_bundle, spatial_index, demographics = _toy_setup()
    raw = pd.DataFrame({
        "bedrooms": [3, 4], "bathrooms": [2.0, 2.5], "sqft_living": [1800, 2400],
        "sqft_lot": [5000, 6000], "floors": [1.0, 2.0], "waterfront": [0, 0], "view": [0, 1],
        "condition": [3, 4], "grade": [7, 9], "sqft_above": [1800, 2000],
        "sqft_basement": [0, 400], "yr_built": [1990, 2005], "yr_renovated": [0, 0],
        "zipcode": [98101, 98102], "lat": [47.6, 47.65], "long": [-122.3, -122.25],
        "sqft_living15": [1800, 2200], "sqft_lot15": [5000, 5800],
    })
    p1 = predict_price(raw, model_bundle, spatial_index, demographics, as_of_date=date(2026, 1, 1))
    p2 = predict_price(raw, model_bundle, spatial_index, demographics, as_of_date=date(2026, 1, 1))
    assert np.allclose(p1, p2)


def test_predict_price_does_not_require_id_date_price_columns():
    """future_unseen_examples.csv nao tem id/date/price - o pipeline nao pode exigi-los."""
    model_bundle, spatial_index, demographics = _toy_setup()
    raw = pd.DataFrame({
        "bedrooms": [3], "bathrooms": [2.0], "sqft_living": [1800], "sqft_lot": [5000],
        "floors": [1.0], "waterfront": [0], "view": [0], "condition": [3], "grade": [7],
        "sqft_above": [1800], "sqft_basement": [0], "yr_built": [1990], "yr_renovated": [0],
        "zipcode": [98103], "lat": [47.65], "long": [-122.3], "sqft_living15": [1800],
        "sqft_lot15": [5000],
    })
    assert "id" not in raw.columns and "date" not in raw.columns and "price" not in raw.columns
    preds = predict_price(raw, model_bundle, spatial_index, demographics)
    assert len(preds) == 1
