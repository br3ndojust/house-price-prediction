import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.clean import clean_kc_house_data, drop_implausible_rows
from src.data.merge import merge_demographics


def _toy_house():
    return pd.DataFrame({
        "id": [1, 2, 3],
        "bedrooms": [3, 33, 4],
        "sqft_living": [1500, 1620, 2000],
        "zipcode": [98101, 98101, 98102],
        "price": [500000, 640000, 700000],
    })


def test_drop_implausible_rows_removes_only_bedroom_outlier():
    out, log = drop_implausible_rows(_toy_house())
    assert len(out) == 2
    assert 2 not in out["id"].values
    assert log["dropped_implausible_bedrooms_ids"] == [2]


def test_clean_kc_house_data_adds_price_log():
    out, log = clean_kc_house_data(_toy_house())
    assert "price_log" in out.columns
    assert np.isclose(out.loc[out["id"] == 1, "price_log"].iloc[0], np.log1p(500000))
    assert log["n_rows_out"] == 2


def test_merge_demographics_preserves_row_count():
    house = _toy_house().drop(columns=[]).iloc[:2].assign(zipcode=[98101, 98102])
    demo = pd.DataFrame({"zipcode": [98101, 98102], "medn_hshld_incm_amt": [80000, 95000]})
    merged = merge_demographics(house, demo)
    assert len(merged) == len(house)
    assert "medn_hshld_incm_amt" in merged.columns


def test_merge_demographics_raises_on_missing_zipcode():
    house = pd.DataFrame({"zipcode": [98101, 99999]})
    demo = pd.DataFrame({"zipcode": [98101]})
    with pytest.raises(ValueError, match="sem demografia"):
        merge_demographics(house, demo)


def test_merge_demographics_raises_on_duplicated_demo_zipcode():
    house = pd.DataFrame({"zipcode": [98101]})
    demo = pd.DataFrame({"zipcode": [98101, 98101]})
    with pytest.raises(ValueError, match="não é 1:1"):
        merge_demographics(house, demo)
