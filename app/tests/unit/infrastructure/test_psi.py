import numpy as np

from app.infrastructure.drift.feature_drift_evaluator import _psi


def test_psi_zero_for_identical_distributions():
    rng = np.random.RandomState(0)
    data = rng.normal(0, 1, 2000)
    assert _psi(data, data) == 0.0


def test_psi_positive_for_shifted_distribution():
    rng = np.random.RandomState(0)
    expected = rng.normal(0, 1, 2000)
    actual = rng.normal(3, 1, 2000)
    assert _psi(expected, actual) > 0.25
