import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.evaluation.error_matrix import characteristic_cuts, compute_metrics, error_matrix_by


def test_compute_metrics_perfect_prediction():
    y = np.array([100.0, 200.0, 300.0])
    metrics = compute_metrics(y, y)
    assert metrics["mae"] == 0
    assert metrics["rmse"] == 0
    assert metrics["bias"] == 0


def test_compute_metrics_bias_sign():
    y_true = np.array([100.0, 100.0])
    y_pred = np.array([90.0, 90.0])  # modelo subestima -> bias positivo
    metrics = compute_metrics(y_true, y_pred)
    assert metrics["bias"] > 0


def test_compute_metrics_low_sample_flag():
    y = np.arange(10.0)
    assert compute_metrics(y, y)["low_sample"] is True
    y2 = np.arange(25.0)
    assert compute_metrics(y2, y2)["low_sample"] is False


def test_error_matrix_by_group():
    df = pd.DataFrame({
        "band": ["A"] * 25 + ["B"] * 25,
        "y_true": [100.0] * 25 + [200.0] * 25,
        "y_pred": [110.0] * 25 + [190.0] * 25,
    })
    out = error_matrix_by(df, "band", "y_true", "y_pred")
    assert set(out["band"]) == {"A", "B"}
    assert (out["mae"] == 10.0).all()


def test_characteristic_cuts_returns_expected_columns():
    df = pd.DataFrame({
        "waterfront": [1, 0, 0], "grade": [11, 5, 5], "view": [0, 0, 2],
        "yr_renovated": [0, 0, 2010],
        "y_true": [100.0, 200.0, 300.0], "y_pred": [90.0, 210.0, 280.0],
    })
    out = characteristic_cuts(df, "y_true", "y_pred")
    assert "characteristic" in out.columns
    assert set(out["characteristic"]) == {"waterfront=1", "grade>=10", "view>0", "renovated"}
