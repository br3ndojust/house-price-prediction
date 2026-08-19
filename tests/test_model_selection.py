import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.models.train import (
    cross_validate_candidate,
    run_model_selection,
    technical_tie_check,
)


def _toy_train_df(n=300, n_zips=15, seed=0):
    rng = np.random.RandomState(seed)
    zipcode = rng.randint(0, n_zips, n)
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    price_log = 12 + 0.5 * x1 + rng.normal(0, 0.2, n)
    return pd.DataFrame({"zipcode": zipcode, "x1": x1, "x2": x2, "price_log": price_log})


def test_cross_validate_candidate_ridge():
    df = _toy_train_df()
    result = cross_validate_candidate(df, ["x1", "x2"], "price_log", "zipcode", "ridge",
                                       {"alpha": 1.0}, n_splits=3, random_state=42)
    assert result["mae_mean"] > 0
    assert -1 <= result["r2_log_mean"] <= 1


def test_run_model_selection_sorted_by_mae():
    df = _toy_train_df()
    grids = {"ridge": {"alpha": [0.1, 10.0]}}
    results = run_model_selection(df, ["x1", "x2"], "price_log", "zipcode", grids, 3, 42)
    maes = [r["mae_mean"] for r in results]
    assert maes == sorted(maes)


def test_technical_tie_check_detects_tie():
    results = [
        {"model": "a", "params": {}, "mae_mean": 100.0, "mae_std": 20.0},
        {"model": "b", "params": {}, "mae_mean": 105.0, "mae_std": 15.0},
    ]
    out = technical_tie_check(results)
    assert out["is_tie"] is True


def test_technical_tie_check_detects_clear_winner():
    results = [
        {"model": "a", "params": {}, "mae_mean": 100.0, "mae_std": 2.0},
        {"model": "b", "params": {}, "mae_mean": 200.0, "mae_std": 2.0},
    ]
    out = technical_tie_check(results)
    assert out["is_tie"] is False
