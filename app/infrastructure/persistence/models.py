"""Tabelas SQLAlchemy 2.0 — predictions, feedback (FK), training_jobs, drift_reports, model_versions.

Criadas via `Base.metadata.create_all()` no startup (sem Alembic — simplificação consciente,
documentada, coerente com SQLite/escopo do exercício, ver README.md do app)."""
from __future__ import annotations

from datetime import datetime

from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class PredictionRecord(Base):
    __tablename__ = "predictions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)
    property_json: Mapped[str] = mapped_column(Text)
    predicted_price: Mapped[float] = mapped_column(Float)
    price_band: Mapped[str] = mapped_column(String(20), index=True)
    property_cluster: Mapped[int] = mapped_column(Integer, index=True)
    model_version: Mapped[str] = mapped_column(String(64))
    confidence_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    confidence_category: Mapped[str | None] = mapped_column(String(40), nullable=True, index=True)

    feedback: Mapped["FeedbackRecord | None"] = relationship(
        back_populates="prediction", uselist=False, cascade="all, delete-orphan"
    )


class FeedbackRecord(Base):
    __tablename__ = "feedback"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    prediction_id: Mapped[str] = mapped_column(ForeignKey("predictions.id"), unique=True)
    actual_price: Mapped[float] = mapped_column(Float)
    recorded_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    prediction: Mapped[PredictionRecord] = relationship(back_populates="feedback")


class TrainingJobRecord(Base):
    __tablename__ = "training_jobs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    status: Mapped[str] = mapped_column(String(20), default="pending", index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    started_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    result_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    triggered_by: Mapped[str | None] = mapped_column(String(120), nullable=True)


class DriftReportRecord(Base):
    __tablename__ = "drift_reports"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)
    n_samples: Mapped[int] = mapped_column(Integer)
    psi_json: Mapped[dict] = mapped_column(JSON)
    overall_status: Mapped[str] = mapped_column(String(20))
    mean_feature_space_distance: Mapped[float | None] = mapped_column(Float, nullable=True)
    prediction_drift_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    model_version: Mapped[str | None] = mapped_column(String(64), nullable=True)


class ModelVersionRecord(Base):
    __tablename__ = "model_versions"

    version: Mapped[str] = mapped_column(String(64), primary_key=True)
    algorithm: Mapped[str] = mapped_column(String(40))
    hyperparameters_json: Mapped[dict] = mapped_column(JSON)
    artifact_path: Mapped[str] = mapped_column(String(255))
    status: Mapped[str] = mapped_column(String(20), default="candidate", index=True)
    cv_mae: Mapped[float | None] = mapped_column(Float, nullable=True)
    test_mae: Mapped[float | None] = mapped_column(Float, nullable=True)
    val_mae: Mapped[float | None] = mapped_column(Float, nullable=True)
    trained_on: Mapped[str | None] = mapped_column(String(120), nullable=True)
    n_train_rows: Mapped[int | None] = mapped_column(Integer, nullable=True)
    n_features: Mapped[int | None] = mapped_column(Integer, nullable=True)
    feature_cols_json: Mapped[list | None] = mapped_column(JSON, nullable=True)
    triggered_by: Mapped[str | None] = mapped_column(String(120), nullable=True)
    training_job_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    recommended_for_promotion: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    promotion_eval_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    performance_metrics_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    promoted_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)


class FeatureSnapshotRecord(Base):
    __tablename__ = "feature_snapshots"

    version: Mapped[str] = mapped_column(String(64), primary_key=True)
    stats_json: Mapped[dict] = mapped_column(JSON)
    n_train_rows: Mapped[int] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class AutoRetrainConfigRecord(Base):
    """Linha única (`id="singleton"`) — configuração do gatilho automático de retraining por volume
    de dado novo (pipeline semi-automático, P6: sempre com aprovação humana antes de ativar)."""
    __tablename__ = "auto_retrain_config"

    id: Mapped[str] = mapped_column(String(20), primary_key=True, default="singleton")
    threshold: Mapped[int] = mapped_column(Integer, default=500)
    enabled: Mapped[bool] = mapped_column(Boolean, default=False)
    consumed_feedback_count: Mapped[int] = mapped_column(Integer, default=0)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
