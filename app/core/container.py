"""Composition root — instancia adapters e injeta nos use cases (clean architecture: só aqui `api`
conhece `infrastructure` concretamente)."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from app.application.use_cases.auto_retrain import (
    CheckAutoRetrainTrigger,
    GetAutoRetrainConfig,
    UpdateAutoRetrainConfig,
)
from app.application.use_cases.evaluate_drift import EvaluateDrift, GetLatestDriftReport, ListDriftReports
from app.application.use_cases.explain_prediction import ExplainPrediction
from app.application.use_cases.get_error_matrix import GetErrorMatrix
from app.application.use_cases.get_feature_snapshot import GetFeatureSnapshot, ListFeatureSnapshots
from app.application.use_cases.list_prediction_model_versions import ListPredictionModelVersions
from app.application.use_cases.get_feature_statistics import GetFeatureSample, GetFeatureStatistics
from app.application.use_cases.get_global_feature_importance import GetGlobalFeatureImportance
from app.application.use_cases.get_health import GetHealth
from app.application.use_cases.get_model_info import GetModelInfo, ListModelVersions
from app.application.use_cases.get_performance_summary import GetPerformanceSummary
from app.application.use_cases.get_prediction_confidence import (
    GetConfidenceCalibrationSummary,
    GetPredictionConfidence,
)
from app.application.use_cases.list_predictions import ListPredictions
from app.application.use_cases.predict_batch import PredictBatch
from app.application.use_cases.predict_price import PredictPrice
from app.application.use_cases.promote_model import PromoteModel
from app.application.use_cases.record_feedback import RecordFeedback
from app.application.use_cases.trigger_training import GetTrainingJob, ListTrainingJobs, TriggerTraining
from app.application.use_cases.trigger_training_from_new_data import (
    TriggerTrainingFromFeedback,
    TriggerTrainingFromUpload,
)
from app.core.config import settings
from app.domain.entities.model_version import ModelVersion
from app.infrastructure.confidence.confidence_scorer import ConfidenceScorer
from app.infrastructure.drift.feature_drift_evaluator import FeatureDriftEvaluator
from app.infrastructure.explainability.feature_snapshot_service import FeatureSnapshotService
from app.infrastructure.explainability.shap_explainer import ShapExplainabilityService
from app.infrastructure.model.artifact_model_repository import ArtifactModelRepository
from app.infrastructure.persistence.auto_retrain_config_repository import SqlAutoRetrainConfigRepository
from app.infrastructure.persistence.feature_snapshot_repository import SqlFeatureSnapshotRepository
from app.infrastructure.persistence.model_version_repository import SqlModelVersionRepository
from app.infrastructure.persistence.prediction_repository import SqlFeedbackRepository, SqlPredictionRepository
from app.infrastructure.training.pipeline_training_service import PipelineTrainingService


class Container:
    """Singleton por processo (via `get_container`) — adapters stateful (modelo carregado em memória,
    SHAP explainer cacheado) precisam sobreviver entre requisições."""

    def __init__(self) -> None:
        self.model_repository = ArtifactModelRepository()
        self.prediction_logger = SqlPredictionRepository()
        self.feedback_repository = SqlFeedbackRepository()
        self.model_version_repository = SqlModelVersionRepository()
        self.feature_snapshot_repository = SqlFeatureSnapshotRepository()
        self.feature_snapshot_service = FeatureSnapshotService(self.feature_snapshot_repository)
        self.auto_retrain_config_repository = SqlAutoRetrainConfigRepository()
        self._seed_baseline_model_version()
        self.explainability_service = ShapExplainabilityService(self.model_repository)
        self.drift_evaluator = FeatureDriftEvaluator(
            self.model_repository, self.prediction_logger, self.model_version_repository
        )
        self.training_service = PipelineTrainingService(
            self.model_version_repository, self.feature_snapshot_service
        )
        self.confidence_service = ConfidenceScorer()

        self.predict_price = PredictPrice(self.model_repository, self.prediction_logger, self.confidence_service)
        self.predict_batch = PredictBatch(self.model_repository, self.prediction_logger, self.confidence_service)
        self.record_feedback = RecordFeedback(self.feedback_repository, self.prediction_logger)
        self.get_performance_summary = GetPerformanceSummary(self.prediction_logger)
        self.list_predictions = ListPredictions(self.prediction_logger)
        self.get_error_matrix = GetErrorMatrix(self.prediction_logger)
        self.list_prediction_model_versions = ListPredictionModelVersions(self.prediction_logger)
        self.evaluate_drift = EvaluateDrift(self.drift_evaluator)
        self.list_drift_reports = ListDriftReports(self.drift_evaluator)
        self.get_latest_drift_report = GetLatestDriftReport(self.drift_evaluator)
        self.trigger_training = TriggerTraining(self.training_service)
        self.trigger_training_from_feedback = TriggerTrainingFromFeedback(
            self.prediction_logger, self.model_repository, self.training_service
        )
        self.trigger_training_from_upload = TriggerTrainingFromUpload(self.model_repository, self.training_service)
        self.get_auto_retrain_config = GetAutoRetrainConfig(self.auto_retrain_config_repository)
        self.update_auto_retrain_config = UpdateAutoRetrainConfig(self.auto_retrain_config_repository)
        self.check_auto_retrain_trigger = CheckAutoRetrainTrigger(
            self.auto_retrain_config_repository, self.prediction_logger, self.trigger_training_from_feedback
        )
        self.get_training_job = GetTrainingJob(self.training_service)
        self.list_training_jobs = ListTrainingJobs(self.training_service)
        self.promote_model = PromoteModel(self.model_version_repository, self.model_repository)
        self.get_model_info = GetModelInfo(self.model_repository)
        self.list_model_versions = ListModelVersions(self.model_version_repository)
        self.get_health = GetHealth(self.model_repository)
        self.get_global_feature_importance = GetGlobalFeatureImportance(self.explainability_service)
        self.get_feature_statistics = GetFeatureStatistics(self.explainability_service)
        self.get_feature_sample = GetFeatureSample(self.explainability_service)
        self.get_feature_snapshot = GetFeatureSnapshot(self.feature_snapshot_repository)
        self.list_feature_snapshots = ListFeatureSnapshots(self.feature_snapshot_repository)
        self.explain_prediction = ExplainPrediction(self.model_repository, self.explainability_service)
        self.get_prediction_confidence = GetPredictionConfidence(self.model_repository, self.confidence_service)
        self.get_confidence_calibration_summary = GetConfidenceCalibrationSummary(self.confidence_service)

    def promote(self, version: str, alias: str | None = None):
        result = self.promote_model.execute(version, alias)
        self.explainability_service.refresh()
        return result

    def _seed_baseline_model_version(self) -> None:
        """Registra `model_final.pkl` (fases 10+13 do pipeline de ML, fora da API) como uma
        `ModelVersion` selecionável em Registro de Modelo — sem isso, o modelo original nunca
        aparecia na lista pra ser escolhido de volta depois que um candidato fosse promovido (não
        dava pra "voltar" pro modelo original pelo portal). Roda uma vez por processo (Container é
        singleton via `get_container`), idempotente entre restarts.

        Sem Alembic (P4, decisão documentada): quando esse schema ganha uma coluna nova
        (`test_mae`, `performance_metrics`, ...), uma linha `model_final` já existente de uma sessão
        anterior nunca ganha esse valor sozinha — `create_all()` só cria tabela/coluna que falta, não
        faz backfill de dado. Por isso este método SEMPRE recalcula os valores abaixo e, se a linha já
        existir, faz backfill só dos campos ainda `NULL` (nunca sobrescreve status/notes reais de uma
        promoção manual)."""
        existing = self.model_version_repository.get("model_final")
        model_meta = self.model_repository.active_contract().get("model", {})
        n_features = self.model_repository.active_contract().get("feature_schema", {}).get("n_features")
        feature_metadata_path = settings.data_dir / "trusted" / "feature_metadata.json"
        feature_cols = None
        if feature_metadata_path.exists():
            feature_cols = json.loads(feature_metadata_path.read_text(encoding="utf-8")).get("model_features")

        n_train_rows = None
        val_check_path = settings.project_root / "reports" / "val_final_check.json"
        if val_check_path.exists():
            try:
                n_train_rows = json.loads(val_check_path.read_text(encoding="utf-8")).get("n_train_test")
            except (ValueError, OSError):
                n_train_rows = None

        def _load_json(path: Path) -> dict:
            if not path.exists():
                return {}
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except (ValueError, OSError):
                return {}

        train_report = _load_json(settings.project_root / "reports" / "model_selection_report.json")
        test_report = _load_json(settings.project_root / "reports" / "error_matrix.json")
        val_report = _load_json(val_check_path)
        train_winner = train_report.get("winner", {})
        test_mae = test_report.get("global", {}).get("mae")

        # relatórios do treino original (pré-fase 16) não têm todas as métricas — RMSE/MAPE de TRAIN
        # e R² de TEST/VAL nasceram só com essa sessão (portal, sessão de performance). Melhor exibir
        # "—" do que inventar um número que a execução original nunca calculou (P4).
        performance_metrics = {
            "train": {
                "mae": train_winner.get("mae_mean"), "rmse": train_winner.get("rmse_mean"),
                "mape": train_winner["mape_mean"] / 100 if train_winner.get("mape_mean") is not None else None,
                "r2_log": train_winner.get("r2_log_mean"),
            },
            "test": {
                "mae": test_report.get("global", {}).get("mae"), "rmse": test_report.get("global", {}).get("rmse"),
                "mape": test_report.get("global", {}).get("mape"), "r2_log": test_report.get("global", {}).get("r2_log"),
            },
            "val": {
                "mae": val_report.get("global", {}).get("mae"), "rmse": val_report.get("global", {}).get("rmse"),
                "mape": val_report.get("global", {}).get("mape"), "r2_log": val_report.get("global", {}).get("r2_log"),
            },
        }

        if existing is not None:
            self.model_version_repository.backfill_missing_fields(
                "model_final",
                cv_mae=model_meta.get("cv_mae_dollar"),
                test_mae=test_mae,
                val_mae=model_meta.get("val_mae_dollar"),
                trained_on=model_meta.get("trained_on"),
                n_train_rows=n_train_rows,
                n_features=n_features,
                feature_cols_json=feature_cols,
                performance_metrics_json=performance_metrics,
                triggered_by="pipeline_ml_original",
            )
        else:
            status = "active" if self.model_version_repository.get_active() is None else "candidate"
            self.model_version_repository.add(
                ModelVersion(
                    version="model_final",
                    algorithm=model_meta.get("algorithm", "xgboost"),
                    hyperparameters=model_meta.get("hyperparameters", {}),
                    artifact_path=str(settings.artifacts_dir / "model_final.pkl"),
                    status=status,
                    cv_mae=model_meta.get("cv_mae_dollar"),
                    test_mae=test_mae,
                    val_mae=model_meta.get("val_mae_dollar"),
                    trained_on=model_meta.get("trained_on"),
                    n_train_rows=n_train_rows,
                    n_features=n_features,
                    feature_cols=feature_cols,
                    triggered_by="pipeline_ml_original",
                    performance_metrics=performance_metrics,
                    notes="Modelo original",
                )
            )
        model_final_path = settings.artifacts_dir / "model_final.pkl"
        features_parquet_path = settings.data_dir / "trusted" / "features_contextual.parquet"
        if model_final_path.exists() and features_parquet_path.exists():
            self.feature_snapshot_service.build_and_persist_from_artifact(
                "model_final", str(model_final_path), str(features_parquet_path)
            )


@lru_cache
def get_container() -> Container:
    return Container()
