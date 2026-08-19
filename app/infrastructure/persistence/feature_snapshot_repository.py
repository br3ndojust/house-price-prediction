from __future__ import annotations

from sqlalchemy import select

from app.domain.entities.feature_snapshot import FeatureSnapshot
from app.infrastructure.persistence.db import session_scope
from app.infrastructure.persistence.models import FeatureSnapshotRecord


def _to_entity(record: FeatureSnapshotRecord) -> FeatureSnapshot:
    return FeatureSnapshot(
        version=record.version,
        stats=record.stats_json,
        n_train_rows=record.n_train_rows,
        created_at=record.created_at,
    )


class SqlFeatureSnapshotRepository:
    def add(self, snapshot: FeatureSnapshot) -> None:
        with session_scope() as session:
            session.add(
                FeatureSnapshotRecord(
                    version=snapshot.version,
                    stats_json=snapshot.stats,
                    n_train_rows=snapshot.n_train_rows,
                    created_at=snapshot.created_at,
                )
            )

    def get(self, version: str) -> FeatureSnapshot | None:
        with session_scope() as session:
            record = session.get(FeatureSnapshotRecord, version)
            return _to_entity(record) if record else None

    def list_all(self) -> list[FeatureSnapshot]:
        with session_scope() as session:
            stmt = select(FeatureSnapshotRecord).order_by(FeatureSnapshotRecord.created_at.desc())
            return [_to_entity(r) for r in session.scalars(stmt).all()]
