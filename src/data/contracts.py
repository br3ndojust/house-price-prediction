"""Schema/null/cardinality contract checks for the 3 raw input files (P1, fase 00)."""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass
class ContractViolation:
    file: str
    check: str
    detail: str
    severity: str = "error"  # "error" blocks the pipeline, "warning" is informational


@dataclass
class ContractReport:
    violations: list[ContractViolation] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not any(v.severity == "error" for v in self.violations)

    def add(self, file: str, check: str, detail: str, severity: str = "error") -> None:
        self.violations.append(ContractViolation(file, check, detail, severity))


def check_kc_house_data(df: pd.DataFrame, cfg: dict, report: ContractReport) -> None:
    name = "kc_house_data"
    missing = set(cfg["expected_columns"]) - set(df.columns)
    if missing:
        report.add(name, "schema", f"missing columns: {sorted(missing)}")

    key = cfg["key_column"]
    key_cols = key if isinstance(key, list) else [key]
    dup_mask = df.duplicated(subset=key_cols)
    if dup_mask.any():
        report.add(name, "cardinality", f"{dup_mask.sum()} duplicated {key_cols}")
    if len(key_cols) == 1:
        id_col = key_cols[0]
        repeated = df[id_col].duplicated(keep=False).sum()
        if repeated:
            report.add(
                name, "cardinality",
                f"{repeated} rows share a non-unique {id_col} but differ elsewhere "
                f"(likely repeat sales of the same property) — investigate in fase 01",
                severity="warning",
            )

    null_counts = df.isnull().sum()
    nulls = null_counts[null_counts > 0]
    if not nulls.empty:
        report.add(name, "nulls", nulls.to_dict().__repr__())

    for col in cfg.get("non_negative_columns", []):
        if col in df.columns and (df[col] < 0).any():
            report.add(name, "range", f"{col} has negative values")

    for col, (lo, hi) in cfg.get("range_checks", {}).items():
        if col in df.columns:
            out = df[(df[col] < lo) | (df[col] > hi)]
            if not out.empty:
                report.add(
                    name, "range", f"{col}: {len(out)} rows outside [{lo}, {hi}]",
                    severity="warning",
                )

    # Known anomaly, flagged not fixed here (fixed in fase 01 data_understanding).
    if "bedrooms" in df.columns:
        extreme = df[df["bedrooms"] > 15]
        if not extreme.empty:
            report.add(
                name, "outlier",
                f"{len(extreme)} row(s) with bedrooms>15 (ids: {extreme['id'].tolist()})",
                severity="warning",
            )


def check_zipcode_demographics(df: pd.DataFrame, cfg: dict, report: ContractReport) -> None:
    name = "zipcode_demographics"
    key = cfg["key_column"]
    if df[key].duplicated().any():
        report.add(name, "cardinality", f"{df[key].duplicated().sum()} duplicated {key}")
    if len(df) < cfg["min_rows"]:
        report.add(name, "row_count", f"only {len(df)} rows, expected >= {cfg['min_rows']}")
    null_counts = df.isnull().sum()
    nulls = null_counts[null_counts > 0]
    if not nulls.empty:
        report.add(name, "nulls", nulls.to_dict().__repr__())


def check_future_unseen(df: pd.DataFrame, cfg: dict, report: ContractReport) -> None:
    name = "future_unseen_examples"
    present_forbidden = [c for c in cfg.get("forbidden_columns", []) if c in df.columns]
    if present_forbidden:
        report.add(name, "leakage", f"forbidden columns present: {present_forbidden}")
    null_counts = df.isnull().sum()
    nulls = null_counts[null_counts > 0]
    if not nulls.empty:
        report.add(name, "nulls", nulls.to_dict().__repr__())
