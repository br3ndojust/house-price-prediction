import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.validation.ablation import run_ablation_cv
from src.validation.leakage import (
    check_determinism,
    check_spatial_index_fit_size,
    train_test_correlation_gap,
)


def test_check_determinism_true_for_pure_function():
    def build_fn(df):
        df = df.copy()
        df["x2"] = df["x"] * 2
        return df

    df = pd.DataFrame({"x": [1, 2, 3]})
    assert check_determinism(build_fn, df) is True


def test_check_determinism_false_for_random_function():
    def build_fn(df):
        df = df.copy()
        df["x2"] = df["x"] + np.random.rand(len(df))
        return df

    df = pd.DataFrame({"x": [1, 2, 3]})
    assert check_determinism(build_fn, df) is False


def test_check_spatial_index_fit_size():
    class FakeIndex:
        price_log = np.zeros(100)

    assert check_spatial_index_fit_size(FakeIndex(), 100) is True
    assert check_spatial_index_fit_size(FakeIndex(), 50) is False


def test_train_test_correlation_gap_flags_suspicious_positive_gap():
    rng = np.random.RandomState(0)
    n = 200
    df = pd.DataFrame({
        "split": ["train"] * 100 + ["test"] * 100,
        "y": rng.normal(0, 1, n),
    })
    # feature quase perfeita em test, fraca em train -> gap positivo suspeito
    df["f"] = np.where(df["split"] == "test", df["y"] + rng.normal(0, 0.01, n), rng.normal(0, 1, n))
    result = train_test_correlation_gap(df, "f", "y")
    assert result["suspicious_leakage"] is True


def test_train_test_correlation_gap_not_suspicious_for_negative_gap():
    rng = np.random.RandomState(0)
    n = 200
    df = pd.DataFrame({
        "split": ["train"] * 100 + ["test"] * 100,
        "y": rng.normal(0, 1, n),
    })
    df["f"] = np.where(df["split"] == "train", df["y"] + rng.normal(0, 0.1, n), rng.normal(0, 1, n))
    result = train_test_correlation_gap(df, "f", "y")
    assert result["suspicious_leakage"] is False


def _toy_train_df(n=300, n_zips=10, seed=0):
    rng = np.random.RandomState(seed)
    zipcode = rng.randint(0, n_zips, n)
    sqft = rng.normal(1800, 400, n)
    noise_feature = rng.normal(0, 1, n)
    price_log = 10 + 0.001 * sqft + rng.normal(0, 0.1, n)
    return pd.DataFrame({
        "zipcode": zipcode, "sqft_living": sqft, "noise_feature": noise_feature,
        "price_log": price_log,
        "price_band": pd.cut(price_log, 2, labels=["low", "high"]),
    })


def test_run_ablation_cv_returns_expected_keys():
    df = _toy_train_df()
    result = run_ablation_cv(df, ["sqft_living"], "price_log", "zipcode", n_splits=3, random_state=42)
    assert "mae_mean" in result
    assert "mae_by_band_mean" in result
    assert result["mae_mean"] > 0


def test_run_ablation_cv_useful_feature_beats_noise_feature():
    df = _toy_train_df()
    baseline = run_ablation_cv(df, ["noise_feature"], "price_log", "zipcode", 3, 42)
    with_signal = run_ablation_cv(df, ["sqft_living"], "price_log", "zipcode", 3, 42)
    assert with_signal["mae_mean"] < baseline["mae_mean"]
