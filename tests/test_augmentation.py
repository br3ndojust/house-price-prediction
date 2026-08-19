import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.augmentation.perturbation import controlled_perturbation_augment, estimate_size_elasticity
from src.augmentation.smogn import relevance_mask, smogn_augment


def _toy_train(n=300, seed=0):
    rng = np.random.RandomState(seed)
    sqft = rng.normal(1800, 400, n).clip(500, 6000)
    price_log = 10 + 0.6 * np.log(sqft) + rng.normal(0, 0.15, n)
    return pd.DataFrame({
        "id": np.arange(n),
        "zipcode": rng.choice([98101, 98102, 98103], n),
        "sqft_living": sqft,
        "sqft_lot": rng.normal(6000, 1500, n).clip(1000, 20000),
        "grade": rng.randint(4, 12, n),
        "waterfront": rng.choice([0, 1], n, p=[0.95, 0.05]),
        "view": rng.randint(0, 5, n),
        "price_log": price_log,
        "price": np.expm1(price_log),
    })


def test_relevance_mask_flags_tails_only():
    target = pd.Series(np.arange(100))
    mask = relevance_mask(target, low_q=0.1, high_q=0.9)
    assert mask.sum() < len(target)
    assert mask.iloc[0] and mask.iloc[-1]
    assert not mask.iloc[50]


def test_smogn_augment_adds_rows_and_preserves_original():
    df = _toy_train()
    feature_cols = ["sqft_living", "sqft_lot", "grade", "waterfront", "view"]
    out = smogn_augment(df, feature_cols, "price_log", k=5, tail_boost=1.0, random_state=42)
    assert len(out) > len(df)
    assert (out.iloc[: len(df)]["id"] == df["id"]).all()


def test_smogn_augment_synthetic_zipcode_always_inherited_from_real_row():
    df = _toy_train()
    feature_cols = ["sqft_living", "sqft_lot", "grade", "waterfront", "view"]
    out = smogn_augment(df, feature_cols, "price_log", k=5, tail_boost=2.0, random_state=1)
    synthetic = out.iloc[len(df):]
    assert set(synthetic["zipcode"]) <= set(df["zipcode"])
    assert (synthetic["id"] == -1).all()


def test_smogn_augment_deterministic():
    df = _toy_train()
    feature_cols = ["sqft_living", "sqft_lot", "grade", "waterfront", "view"]
    out1 = smogn_augment(df, feature_cols, "price_log", k=5, tail_boost=1.0, random_state=7)
    out2 = smogn_augment(df, feature_cols, "price_log", k=5, tail_boost=1.0, random_state=7)
    pd.testing.assert_frame_equal(out1, out2)


def test_estimate_size_elasticity_positive_for_realistic_data():
    df = _toy_train()
    elasticity = estimate_size_elasticity(df)
    assert elasticity > 0  # preco cresce com tamanho


def test_controlled_perturbation_does_not_copy_original_price():
    df = _toy_train()
    out = controlled_perturbation_augment(df, augment_fraction=1.0, random_state=42)
    synthetic = out.iloc[len(df):].reset_index(drop=True)
    original = df.reset_index(drop=True)
    # preco sintetico difere do preco da linha original correspondente (nao e copia)
    assert not np.isclose(synthetic["price_log"].to_numpy(), original["price_log"].to_numpy()).all()


def test_controlled_perturbation_keeps_grade_waterfront_view_zipcode_fixed():
    df = _toy_train()
    out = controlled_perturbation_augment(df, augment_fraction=1.0, random_state=42)
    synthetic = out.iloc[len(df):].reset_index(drop=True)
    original = df.reset_index(drop=True)
    assert (synthetic["grade"].to_numpy() == original["grade"].to_numpy()).all()
    assert (synthetic["waterfront"].to_numpy() == original["waterfront"].to_numpy()).all()
    assert (synthetic["view"].to_numpy() == original["view"].to_numpy()).all()
    assert (synthetic["zipcode"].to_numpy() == original["zipcode"].to_numpy()).all()


def test_controlled_perturbation_size_change_within_expected_range():
    df = _toy_train()
    out = controlled_perturbation_augment(df, size_pct_range=(0.02, 0.05), augment_fraction=1.0,
                                           random_state=42)
    synthetic = out.iloc[len(df):].reset_index(drop=True)
    original = df.reset_index(drop=True)
    pct_change = (synthetic["sqft_living"] - original["sqft_living"]).abs() / original["sqft_living"]
    assert (pct_change >= 0.015).all() and (pct_change <= 0.055).all()
