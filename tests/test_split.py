import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.validation.split import canonical_split, split_summary


def _toy_df(n_groups=20, rows_per_group=10):
    rows = []
    for g in range(n_groups):
        for _ in range(rows_per_group):
            rows.append({"zipcode": g})
    return pd.DataFrame(rows)


def test_canonical_split_no_group_overlap():
    df = _toy_df()
    split = canonical_split(df, "zipcode", 0.7, 0.15, 0.15, random_state=42)
    summary = split_summary(df, split, "zipcode")
    assert summary["group_overlap"] == []


def test_canonical_split_proportions_reasonable():
    df = _toy_df(n_groups=50, rows_per_group=20)
    split = canonical_split(df, "zipcode", 0.7, 0.15, 0.15, random_state=42)
    summary = split_summary(df, split, "zipcode")
    assert 0.55 < summary["train"]["pct_rows"] < 0.85
    assert summary["test"]["n_rows"] > 0
    assert summary["val"]["n_rows"] > 0


def test_canonical_split_deterministic():
    df = _toy_df()
    split_a = canonical_split(df, "zipcode", 0.7, 0.15, 0.15, random_state=42)
    split_b = canonical_split(df, "zipcode", 0.7, 0.15, 0.15, random_state=42)
    assert (split_a == split_b).all()


def test_canonical_split_rejects_bad_proportions():
    df = _toy_df()
    with pytest.raises(ValueError):
        canonical_split(df, "zipcode", 0.7, 0.4, 0.4, random_state=42)


def test_resales_never_split_across_partitions():
    """Mesma propriedade (id) sempre no mesmo zipcode -> nunca cai em partes diferentes do split."""
    df = pd.DataFrame({
        "id": [1, 1, 2, 2, 3],
        "zipcode": [10, 10, 11, 11, 12],
    })
    full = pd.concat([_toy_df(n_groups=10, rows_per_group=5), df], ignore_index=True)
    split = canonical_split(full, "zipcode", 0.7, 0.15, 0.15, random_state=1)
    resale_check = pd.DataFrame({"id": full["id"], "split": split}).dropna()
    grouped = resale_check.groupby("id")["split"].nunique()
    assert (grouped == 1).all()
