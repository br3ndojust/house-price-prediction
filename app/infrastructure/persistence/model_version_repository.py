from __future__ import annotations

from datetime import datetime

from sqlalchemy import select, update

from app.domain.entities.model_version import ModelVersion
from app.infrastructure.persistence.db import session_scope
from app.infrastructure.persistence.models import ModelVersionRecord


def _to_entity(record: ModelVersionRecord) -> ModelVersion:
    return ModelVersion(
        version=record.version,
        algorithm=record.algorithm,
        hyperparameters=record.hyperparameters_json,
        artifact_path=record.artifact_path,
        status=record.status,
        cv_mae=record.cv_mae,
        test_mae=record.test_mae,
        val_mae=record.val_mae,
        trained_on=record.trained_on,
        n_train_rows=record.n_train_rows,
        n_features=record.n_features,
        feature_cols=record.feature_cols_json,
        triggered_by=record.triggered_by,
        training_job_id=record.training_job_id,
        recommended_for_promotion=record.recommended_for_promotion,
        promotion_eval=record.promotion_eval_json,
        performance_metrics=record.performance_metrics_json,
        created_at=record.created_at,
        promoted_at=record.promoted_at,
        notes=record.notes,
    )


class SqlModelVersionRepository:
    def add(self, version: ModelVersion) -> None:
        with session_scope() as session:
            session.add(
                ModelVersionRecord(
                    version=version.version,
                    algorithm=version.algorithm,
                    hyperparameters_json=version.hyperparameters,
                    artifact_path=version.artifact_path,
                    status=version.status,
                    cv_mae=version.cv_mae,
                    test_mae=version.test_mae,
                    val_mae=version.val_mae,
                    trained_on=version.trained_on,
                    n_train_rows=version.n_train_rows,
                    n_features=version.n_features,
                    feature_cols_json=version.feature_cols,
                    triggered_by=version.triggered_by,
                    training_job_id=version.training_job_id,
                    recommended_for_promotion=version.recommended_for_promotion,
                    promotion_eval_json=version.promotion_eval,
                    performance_metrics_json=version.performance_metrics,
                    created_at=version.created_at,
                    notes=version.notes,
                )
            )

    def backfill_missing_fields(self, version: str, **fields) -> None:
        """Preenche só as colunas de `fields` que estão `NULL` no registro hoje — nunca sobrescreve
        um valor já gravado (ex.: `status`/`notes` alterados por uma promoção real). Usado pelo seed
        do modelo original (`Container._seed_baseline_model_version`) pra recuperar de um schema que
        evoluiu depois que a linha já existia (sem Alembic — colunas novas nascem sempre `NULL` numa
        linha antiga, `create_all()` não faz backfill sozinho)."""
        with session_scope() as session:
            record = session.get(ModelVersionRecord, version)
            if record is None:
                return
            for field, value in fields.items():
                if getattr(record, field, None) is None and value is not None:
                    setattr(record, field, value)

    def get(self, version: str) -> ModelVersion | None:
        with session_scope() as session:
            record = session.get(ModelVersionRecord, version)
            return _to_entity(record) if record else None

    def list_all(self) -> list[ModelVersion]:
        with session_scope() as session:
            stmt = select(ModelVersionRecord).order_by(ModelVersionRecord.created_at.desc())
            return [_to_entity(r) for r in session.scalars(stmt).all()]

    def get_active(self) -> ModelVersion | None:
        with session_scope() as session:
            record = session.scalar(select(ModelVersionRecord).where(ModelVersionRecord.status == "active"))
            return _to_entity(record) if record else None

    def promote(self, version: str, alias: str | None = None) -> ModelVersion:
        with session_scope() as session:
            target = session.get(ModelVersionRecord, version)
            if target is None:
                raise ValueError(f"versão {version} não encontrada")
            session.execute(
                update(ModelVersionRecord)
                .where(ModelVersionRecord.status == "active")
                .values(status="superseded")
            )
            target.status = "active"
            target.promoted_at = datetime.utcnow()
            if alias:
                target.notes = alias
            session.flush()
            return _to_entity(target)
