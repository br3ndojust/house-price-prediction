from __future__ import annotations

from datetime import datetime

from app.domain.entities.auto_retrain_config import AutoRetrainConfig
from app.infrastructure.persistence.db import session_scope
from app.infrastructure.persistence.models import AutoRetrainConfigRecord

_SINGLETON_ID = "singleton"


class SqlAutoRetrainConfigRepository:
    def get(self) -> AutoRetrainConfig:
        with session_scope() as session:
            record = session.get(AutoRetrainConfigRecord, _SINGLETON_ID)
            if record is None:
                record = AutoRetrainConfigRecord(id=_SINGLETON_ID)
                session.add(record)
                session.flush()
            return AutoRetrainConfig(
                threshold=record.threshold, enabled=record.enabled,
                consumed_feedback_count=record.consumed_feedback_count, updated_at=record.updated_at,
            )

    def save(self, config: AutoRetrainConfig) -> None:
        with session_scope() as session:
            record = session.get(AutoRetrainConfigRecord, _SINGLETON_ID)
            if record is None:
                record = AutoRetrainConfigRecord(id=_SINGLETON_ID)
                session.add(record)
            record.threshold = config.threshold
            record.enabled = config.enabled
            record.consumed_feedback_count = config.consumed_feedback_count
            record.updated_at = datetime.utcnow()
