import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.contracts import (
    ContractReport,
    check_future_unseen,
    check_kc_house_data,
    check_zipcode_demographics,
)

KC_CFG = {
    "expected_columns": ["id", "price", "bedrooms", "sqft_living", "zipcode"],
    "key_column": "id",
    "non_negative_columns": ["price"],
    "range_checks": {},
}


def test_kc_house_data_flags_duplicate_key():
    df = pd.DataFrame({"id": [1, 1], "price": [100, 200], "bedrooms": [3, 3],
                        "sqft_living": [1000, 1000], "zipcode": [98101, 98101]})
    report = ContractReport()
    check_kc_house_data(df, KC_CFG, report)
    assert not report.ok
    assert any(v.check == "cardinality" for v in report.violations)


def test_kc_house_data_flags_negative_price():
    df = pd.DataFrame({"id": [1, 2], "price": [-5, 200], "bedrooms": [3, 3],
                        "sqft_living": [1000, 1000], "zipcode": [98101, 98101]})
    report = ContractReport()
    check_kc_house_data(df, KC_CFG, report)
    assert not report.ok
    assert any(v.check == "range" and v.severity == "error" for v in report.violations)


def test_kc_house_data_clean_passes():
    df = pd.DataFrame({"id": [1, 2], "price": [100, 200], "bedrooms": [3, 4],
                        "sqft_living": [1000, 1200], "zipcode": [98101, 98102]})
    report = ContractReport()
    check_kc_house_data(df, KC_CFG, report)
    assert report.ok


def test_zipcode_demographics_flags_duplicate_key():
    df = pd.DataFrame({"zipcode": [98101, 98101]})
    report = ContractReport()
    check_zipcode_demographics(df, {"key_column": "zipcode", "min_rows": 1}, report)
    assert not report.ok


def test_future_unseen_flags_price_leakage():
    df = pd.DataFrame({"price": [100], "bedrooms": [3]})
    report = ContractReport()
    check_future_unseen(df, {"forbidden_columns": ["price", "id", "date"]}, report)
    assert not report.ok
    assert any(v.check == "leakage" for v in report.violations)


def test_future_unseen_clean_passes():
    df = pd.DataFrame({"bedrooms": [3], "sqft_living": [1000]})
    report = ContractReport()
    check_future_unseen(df, {"forbidden_columns": ["price", "id", "date"]}, report)
    assert report.ok


def test_real_raw_files_pass_contract():
    """Contrato real contra os arquivos em data/raw/ — só warnings esperados (outlier bedrooms=33)."""
    import yaml

    cfg = yaml.safe_load(Path("configs/data_contract.yaml").read_text(encoding="utf-8"))
    kc = pd.read_csv(cfg["kc_house_data"]["path"])
    demo = pd.read_csv(cfg["zipcode_demographics"]["path"])
    future = pd.read_csv(cfg["future_unseen_examples"]["path"])

    report = ContractReport()
    check_kc_house_data(kc, cfg["kc_house_data"], report)
    check_zipcode_demographics(demo, cfg["zipcode_demographics"], report)
    check_future_unseen(future, cfg["future_unseen_examples"], report)
    assert report.ok, [v for v in report.violations if v.severity == "error"]
