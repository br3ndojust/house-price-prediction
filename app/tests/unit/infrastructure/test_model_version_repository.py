import pytest

from app.domain.entities.model_version import ModelVersion
from app.infrastructure.persistence.db import create_all_tables
from app.infrastructure.persistence.model_version_repository import SqlModelVersionRepository


@pytest.fixture(autouse=True)
def _tables():
    create_all_tables()


def test_backfill_missing_fields_fills_only_null_columns():
    repo = SqlModelVersionRepository()
    repo.add(ModelVersion(
        version="v1", algorithm="xgboost", hyperparameters={}, artifact_path="artifacts/v1.pkl",
        status="active", val_mae=72000.0, notes="já tinha valor",
    ))

    repo.backfill_missing_fields(
        "v1",
        n_train_rows=15000, test_mae=104000.0,
        val_mae=999999.0,  # já preenchido — não pode sobrescrever
        performance_metrics_json={"train": {"mae": 1.0}},
    )

    updated = repo.get("v1")
    assert updated.n_train_rows == 15000
    assert updated.test_mae == 104000.0
    assert updated.val_mae == 72000.0  # preservado, não sobrescrito
    assert updated.performance_metrics == {"train": {"mae": 1.0}}


def test_backfill_missing_fields_noop_when_version_unknown():
    repo = SqlModelVersionRepository()
    repo.backfill_missing_fields("does-not-exist", n_train_rows=1)  # não deve lançar
    assert repo.get("does-not-exist") is None
