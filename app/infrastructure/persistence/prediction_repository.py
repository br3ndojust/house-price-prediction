"""Implementa os ports PredictionLogger + FeedbackRepository sobre SQLite/SQLAlchemy."""
from __future__ import annotations

import dataclasses
import json
import uuid
from datetime import datetime, timedelta

from sqlalchemy import func, select

from app.domain.entities.prediction import Prediction
from app.domain.entities.property import Property
from app.infrastructure.persistence.db import session_scope
from app.infrastructure.persistence.models import FeedbackRecord, PredictionRecord


def _to_entity(record: PredictionRecord) -> Prediction:
    prop = Property(**json.loads(record.property_json))
    return Prediction(
        property=prop,
        predicted_price=record.predicted_price,
        price_band=record.price_band,
        property_cluster=record.property_cluster,
        model_version=record.model_version,
        confidence_score=record.confidence_score,
        confidence_category=record.confidence_category,
        id=record.id,
        created_at=record.created_at,
        actual_price=record.feedback.actual_price if record.feedback else None,
        feedback_recorded_at=record.feedback.recorded_at if record.feedback else None,
    )


class SqlPredictionRepository:
    def log(self, prediction: Prediction) -> str:
        prediction_id = prediction.id or str(uuid.uuid4())
        with session_scope() as session:
            record = PredictionRecord(
                id=prediction_id,
                created_at=prediction.created_at,
                property_json=json.dumps(dataclasses.asdict(prediction.property)),
                predicted_price=prediction.predicted_price,
                price_band=prediction.price_band,
                property_cluster=prediction.property_cluster,
                model_version=prediction.model_version,
                confidence_score=prediction.confidence_score,
                confidence_category=prediction.confidence_category,
            )
            session.add(record)
        return prediction_id

    def get(self, prediction_id: str) -> Prediction | None:
        with session_scope() as session:
            record = session.get(PredictionRecord, prediction_id)
            return _to_entity(record) if record else None

    def list_recent(self, limit: int = 100, model_version: str | None = None) -> list[Prediction]:
        with session_scope() as session:
            stmt = select(PredictionRecord).order_by(PredictionRecord.created_at.desc()).limit(limit)
            if model_version:
                stmt = stmt.where(PredictionRecord.model_version == model_version)
            return [_to_entity(r) for r in session.scalars(stmt).all()]

    def list_with_feedback(self, model_version: str | None = None) -> list[Prediction]:
        with session_scope() as session:
            stmt = select(PredictionRecord).join(FeedbackRecord)
            if model_version:
                stmt = stmt.where(PredictionRecord.model_version == model_version)
            return [_to_entity(r) for r in session.scalars(stmt).all()]

    def summary(self, model_version: str | None = None) -> dict:
        def _filtered(stmt):
            return stmt.where(PredictionRecord.model_version == model_version) if model_version else stmt

        with session_scope() as session:
            total = session.scalar(_filtered(select(func.count()).select_from(PredictionRecord))) or 0
            since = datetime.utcnow() - timedelta(hours=24)
            last_24h = session.scalar(
                _filtered(select(func.count()).select_from(PredictionRecord)).where(
                    PredictionRecord.created_at >= since
                )
            ) or 0
            by_band = dict(
                session.execute(
                    _filtered(select(PredictionRecord.price_band, func.count())).group_by(
                        PredictionRecord.price_band
                    )
                ).all()
            )
            by_cluster = dict(
                session.execute(
                    _filtered(select(PredictionRecord.property_cluster, func.count())).group_by(
                        PredictionRecord.property_cluster
                    )
                ).all()
            )
            with_feedback = session.scalar(
                _filtered(select(func.count()).select_from(FeedbackRecord).join(PredictionRecord))
            ) or 0
            by_confidence = dict(
                session.execute(
                    _filtered(select(PredictionRecord.confidence_category, func.count()))
                    .where(PredictionRecord.confidence_category.is_not(None))
                    .group_by(PredictionRecord.confidence_category)
                ).all()
            )
        return {
            "total_predictions": total,
            "predictions_last_24h": last_24h,
            "predictions_with_feedback": with_feedback,
            "by_price_band": by_band,
            "by_property_cluster": {str(k): v for k, v in by_cluster.items()},
            "by_confidence_category": by_confidence,
            "latency_note": "latência de requisição é exposta via GET /metrics (Prometheus, histogram "
                             "http_request_duration_seconds), não duplicada aqui.",
        }

    def distinct_model_versions(self) -> list[str]:
        with session_scope() as session:
            rows = session.execute(
                select(PredictionRecord.model_version).distinct().order_by(PredictionRecord.model_version)
            ).all()
            return [r[0] for r in rows if r[0]]


class SqlFeedbackRepository:
    def record(self, prediction_id: str, actual_price: float) -> None:
        with session_scope() as session:
            existing = session.scalar(
                select(FeedbackRecord).where(FeedbackRecord.prediction_id == prediction_id)
            )
            if existing:
                existing.actual_price = actual_price
                existing.recorded_at = datetime.utcnow()
            else:
                session.add(FeedbackRecord(prediction_id=prediction_id, actual_price=actual_price))
