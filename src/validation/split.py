"""Fase 02 — split canônico por zipcode (P1: GroupShuffleSplit, val é sagrado a partir daqui)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


def canonical_split(df: pd.DataFrame, group_column: str, train: float, test: float, val: float,
                     random_state: int) -> pd.Series:
    """Retorna uma Series `split` alinhada a `df.index` com valores 'train'/'test'/'val'.

    Split é feito por grupo (`group_column`, ex: zipcode) — todas as linhas de um mesmo grupo caem
    inteiramente do mesmo lado, nunca um zipcode aparece em mais de uma partição.
    """
    if not np.isclose(train + test + val, 1.0):
        raise ValueError(f"train+test+val deve somar 1.0, recebido {train + test + val}")

    groups = df[group_column]

    gss1 = GroupShuffleSplit(n_splits=1, train_size=train, random_state=random_state)
    train_idx, rest_idx = next(gss1.split(df, groups=groups))

    rest_groups = groups.iloc[rest_idx]
    test_share_of_rest = test / (test + val)
    gss2 = GroupShuffleSplit(n_splits=1, train_size=test_share_of_rest, random_state=random_state)
    test_rel_idx, val_rel_idx = next(gss2.split(rest_idx, groups=rest_groups))
    test_idx = rest_idx[test_rel_idx]
    val_idx = rest_idx[val_rel_idx]

    split = pd.Series("train", index=df.index)
    split.iloc[test_idx] = "test"
    split.iloc[val_idx] = "val"
    return split


def split_summary(df: pd.DataFrame, split: pd.Series, group_column: str) -> dict:
    out = {}
    for part in ["train", "test", "val"]:
        mask = split == part
        out[part] = {
            "n_rows": int(mask.sum()),
            "pct_rows": round(float(mask.mean()), 4),
            "n_groups": int(df.loc[mask, group_column].nunique()),
        }
    groups_by_part = {p: set(df.loc[split == p, group_column]) for p in ["train", "test", "val"]}
    overlap = (
        groups_by_part["train"] & groups_by_part["test"]
        | groups_by_part["train"] & groups_by_part["val"]
        | groups_by_part["test"] & groups_by_part["val"]
    )
    out["group_overlap"] = sorted(overlap)
    return out
