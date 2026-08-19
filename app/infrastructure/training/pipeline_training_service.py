"""Adapter do port TrainingService — chama scripts/train_model.py + scripts/finalize_model.py
refatorados (funções chamáveis, P5: sem duplicar lógica, sem subprocess).

Escopo explícito (honestidade técnica, ver README.md do app): o job de retraining desta API reexecuta
as fases 10 (seleção de modelo) + 13 (refit final + checagem em `val`) sobre o dataset já travado em
`data/trusted/features_contextual.parquet` — o mesmo pipeline usado no treino original, mas sem um
mecanismo de acumulação de dado novo rotulado (isso é `docs/08_continuous_learning.md` seções 2-3,
uma integração externa fora do escopo desta API). Nunca ativa o candidato sozinho (P6) — só
`/model/promote` faz isso, via `PromoteModel`.
"""
from __future__ import annotations

import json
import logging
import sys
import uuid
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
from sqlalchemy import select

from src.evaluation.promotion_criterion import check_promotion_criterion, score_model_on_val

from app.core.config import settings
from app.domain.entities.model_version import ModelVersion
from app.domain.entities.training_job import TrainingJob, TrainingJobStatus
from app.infrastructure.persistence.db import session_scope
from app.infrastructure.persistence.model_version_repository import SqlModelVersionRepository
from app.infrastructure.persistence.models import TrainingJobRecord

sys.path.insert(0, str(settings.project_root))

logger = logging.getLogger("app.training")


def _to_entity(record: TrainingJobRecord) -> TrainingJob:
    return TrainingJob(
        id=record.id,
        status=TrainingJobStatus(record.status),
        created_at=record.created_at,
        started_at=record.started_at,
        finished_at=record.finished_at,
        result=record.result_json,
        error=record.error,
        triggered_by=record.triggered_by,
    )


class PipelineTrainingService:
    def __init__(self, model_version_repository: SqlModelVersionRepository | None = None,
                 feature_snapshot_service=None):
        self._versions = model_version_repository or SqlModelVersionRepository()
        self._feature_snapshots = feature_snapshot_service
        # dado novo (feedback/upload) fica em memória até o BackgroundTask rodar — mesmo processo,
        # não precisa de fila externa (decisão já registrada em docs/02_execution_plan_api.md)
        self._staged_extra_rows: dict[str, pd.DataFrame] = {}
        self._staged_feature_cols: dict[str, list[str]] = {}
        self._staged_metric: dict[str, str] = {}

    def create_job(self, triggered_by: str | None) -> TrainingJob:
        job_id = str(uuid.uuid4())
        with session_scope() as session:
            session.add(TrainingJobRecord(id=job_id, status=TrainingJobStatus.PENDING, triggered_by=triggered_by))
        logger.info(f"training job {job_id} criado (triggered_by={triggered_by})")
        return TrainingJob(id=job_id, status=TrainingJobStatus.PENDING, triggered_by=triggered_by)

    def stage_extra_rows(self, job_id: str, extra_rows: pd.DataFrame) -> None:
        """Dado novo rotulado (fase 16) a ser concatenado ao TRAIN oficial quando `run(job_id)` disparar
        — nunca sobrescreve `data/trusted/features_contextual.parquet` (P1), grava num parquet
        temporário só para esta execução."""
        self._staged_extra_rows[job_id] = extra_rows

    def stage_feature_cols(self, job_id: str, feature_cols: list[str]) -> None:
        """Subconjunto de features escolhido pelo usuário (tela Treino) para este job — quando
        ausente, o job usa o conjunto oficial travado (`data/trusted/feature_metadata.json`)."""
        self._staged_feature_cols[job_id] = feature_cols

    def stage_metric(self, job_id: str, metric: str) -> None:
        """Métrica escolhida pelo usuário (tela Treino) para ranquear a seleção de modelo deste job —
        quando ausente, usa `mae` (métrica oficial, P4)."""
        self._staged_metric[job_id] = metric

    def get_job(self, job_id: str) -> TrainingJob | None:
        with session_scope() as session:
            record = session.get(TrainingJobRecord, job_id)
            return _to_entity(record) if record else None

    def list_jobs(self) -> list[TrainingJob]:
        with session_scope() as session:
            stmt = select(TrainingJobRecord).order_by(TrainingJobRecord.created_at.desc())
            return [_to_entity(r) for r in session.scalars(stmt).all()]

    def run(self, job_id: str) -> dict:
        from src.scripts.evaluate_model import run_test_evaluation_pipeline
        from src.scripts.finalize_model import run_final_refit_pipeline
        from src.scripts.train_model import run_model_selection_pipeline

        with session_scope() as session:
            record = session.get(TrainingJobRecord, job_id)
            record.status = TrainingJobStatus.RUNNING
            record.started_at = datetime.utcnow()
            triggered_by = record.triggered_by

        candidates_dir = settings.artifacts_dir / "candidates"
        candidates_dir.mkdir(parents=True, exist_ok=True)
        candidate_path = candidates_dir / f"model_candidate_{job_id}.pkl"

        official_parquet = settings.data_dir / "trusted" / "features_contextual.parquet"
        extra_rows = self._staged_extra_rows.pop(job_id, None)
        feature_cols_override = self._staged_feature_cols.pop(job_id, None)
        metric = self._staged_metric.pop(job_id, "mae")
        n_extra_rows = 0
        n_extra_test = 0
        if extra_rows is not None and len(extra_rows):
            n_extra_rows = len(extra_rows)
            n_extra_test = int((extra_rows["split"] == "test").sum())
            combined = pd.concat([pd.read_parquet(official_parquet), extra_rows], ignore_index=True)
            features_parquet = settings.app_data_dir / f"train_augmented_{job_id}.parquet"
            combined.to_parquet(features_parquet)
            features_parquet = str(features_parquet)
            notes = (
                f"Retraining via API — {n_extra_rows} linha(s) de dado novo (fase 16) somadas ao dataset "
                f"oficial, split sorteado aleatoriamente: {n_extra_rows - n_extra_test} em train, "
                f"{n_extra_test} em test."
            )
        else:
            features_parquet = str(official_parquet)
            notes = "Retraining via API — reexecuta fases 10+13 sobre dataset já travado."
        if feature_cols_override:
            notes += f" Features customizadas ({len(feature_cols_override)}/20): {', '.join(feature_cols_override)}."
        if metric != "mae":
            notes += f" Seleção ranqueada por {metric.upper()} (não a métrica oficial MAE)."

        logger.info(
            f"training job {job_id} iniciado — {n_extra_rows} linha(s) de dado novo, fonte={features_parquet}, "
            f"features={'customizadas (' + str(len(feature_cols_override)) + ')' if feature_cols_override else 'oficiais (20)'}, "
            f"metrica={metric}"
        )

        try:
            logger.info(f"training job {job_id} — fase 10 (seleção de modelo, GroupKFold) rodando...")
            selection_out = run_model_selection_pipeline(
                features_parquet=features_parquet,
                model_cfg_path=str(settings.project_root / "configs" / "model.yaml"),
                feature_metadata_path=str(settings.data_dir / "trusted" / "feature_metadata.json"),
                candidate_artifact_path=str(candidate_path),
                feature_cols_override=feature_cols_override,
                metric=metric,
            )
            logger.info(
                f"training job {job_id} — fase 10 concluída: {selection_out['winner']['model']} "
                f"mae_cv=${selection_out['winner']['mae_mean']:,.0f}"
            )

            # fase 11 — MAE em TEST, com o artefato ainda fitado só em TRAIN (fase 10). Tem que rodar
            # ANTES da fase 13, que refita em TRAIN+TEST e sobrescreve o mesmo artefato — depois disso
            # o modelo já viu TEST no fit, avaliar nele deixaria de ser holdout. Junto com cv_mae
            # (TRAIN) e val_mae (VAL, abaixo), dá o trio treino/teste/val pra sinalizar
            # overfitting (treino muito melhor que teste) ou underfitting (os três ruins).
            logger.info(f"training job {job_id} — fase 11 (avaliação em test) rodando...")
            test_eval = run_test_evaluation_pipeline(
                candidate_artifact_path=str(candidate_path), features_parquet=features_parquet,
            )
            logger.info(f"training job {job_id} — fase 11 concluída: mae_test=${test_eval['global']['mae']:,.0f}")

            logger.info(f"training job {job_id} — fase 13 (refit final + checagem em val) rodando...")
            refit_out = run_final_refit_pipeline(
                candidate_artifact_path=str(candidate_path),
                features_parquet=features_parquet,
                final_artifact_path=str(candidate_path),
            )
            logger.info(f"training job {job_id} — fase 13 concluída: mae_val=${refit_out['global']['mae']:,.0f}")

            # trio treino/teste/val (RMSE/MAE/MAPE/R² log) pra sessão de performance do Registro de
            # Modelo — mape do CV (fase 10) vem em pontos percentuais (x100), normaliza pra fração
            # igual test/val (compute_metrics, src/evaluation/error_matrix.py) antes de gravar.
            winner = selection_out["winner"]
            performance_metrics = {
                "train": {
                    "mae": winner.get("mae_mean"), "rmse": winner.get("rmse_mean"),
                    "mape": winner["mape_mean"] / 100 if winner.get("mape_mean") is not None else None,
                    "r2_log": winner.get("r2_log_mean"),
                },
                "test": {
                    "mae": test_eval["global"].get("mae"), "rmse": test_eval["global"].get("rmse"),
                    "mape": test_eval["global"].get("mape"), "r2_log": test_eval["global"].get("r2_log"),
                },
                "val": {
                    "mae": refit_out["global"].get("mae"), "rmse": refit_out["global"].get("rmse"),
                    "mape": refit_out["global"].get("mape"), "r2_log": refit_out["global"].get("r2_log"),
                },
            }

            # avaliação vs. modelo ativo (critério objetivo de promoção, pré-registrado — P4): nunca
            # bloqueia o job se falhar, só fica sem recomendação (best-effort, igual ao snapshot)
            promotion_eval = None
            recommended_for_promotion = None
            active_version = self._versions.get_active()
            if active_version is not None and Path(active_version.artifact_path).exists():
                try:
                    active_bundle = joblib.load(active_version.artifact_path)
                    val_df = pd.read_parquet(features_parquet)
                    val_df = val_df[val_df["split"] == "val"].reset_index(drop=True)
                    active_score = score_model_on_val(active_bundle, val_df)
                    criterion = check_promotion_criterion(refit_out["by_price_band"], active_score["by_price_band"])
                    promotion_eval = {
                        "compared_to_version": active_version.version,
                        "candidate_by_price_band": refit_out["by_price_band"],
                        "active_by_price_band": active_score["by_price_band"],
                        "criterion": criterion,
                    }
                    recommended_for_promotion = criterion["recommended"]
                    logger.info(
                        f"training job {job_id} — avaliação vs. {active_version.version}: "
                        f"recomendado={recommended_for_promotion}"
                    )
                except Exception:
                    logger.exception(
                        f"training job {job_id} — falha ao avaliar critério de promoção vs. modelo "
                        "ativo, seguindo sem recomendação"
                    )

            result = {
                "cv_results_top": selection_out["cv_results"][0],
                "technical_tie_check": selection_out["technical_tie_check"],
                "test_check": test_eval,
                "val_check": refit_out,
                "candidate_artifact_path": str(candidate_path),
                "n_extra_rows_from_new_data": n_extra_rows,
                "n_extra_rows_test_split": n_extra_test,
                "feature_cols_used": feature_cols_override,  # None = conjunto oficial (20 features travadas)
                "metric_used": metric,
                "promotion_eval": promotion_eval,
                "performance_metrics": performance_metrics,
            }

            if feature_cols_override:
                feature_cols_used = feature_cols_override
            else:
                official = json.loads(
                    (settings.data_dir / "trusted" / "feature_metadata.json").read_text(encoding="utf-8")
                )
                feature_cols_used = official["model_features"]
            n_features = len(feature_cols_used)

            version_id = f"candidate-{job_id[:8]}"
            self._versions.add(
                ModelVersion(
                    version=version_id,
                    algorithm=refit_out["model_name"],
                    hyperparameters=refit_out["params"],
                    artifact_path=str(candidate_path),
                    status="candidate",
                    cv_mae=selection_out["winner"]["mae_mean"],
                    test_mae=test_eval["global"]["mae"],
                    val_mae=refit_out["global"]["mae"],
                    trained_on=refit_out["trained_on"],
                    n_train_rows=refit_out.get("n_train_test"),
                    n_features=n_features,
                    feature_cols=feature_cols_used,
                    triggered_by=triggered_by,
                    training_job_id=job_id,
                    recommended_for_promotion=recommended_for_promotion,
                    promotion_eval=promotion_eval,
                    performance_metrics=performance_metrics,
                    notes=notes,
                )
            )
            if self._feature_snapshots is not None:
                self._feature_snapshots.build_and_persist_from_artifact(
                    version_id, str(candidate_path), features_parquet
                )

            with session_scope() as session:
                record = session.get(TrainingJobRecord, job_id)
                record.status = TrainingJobStatus.DONE
                record.finished_at = datetime.utcnow()
                record.result_json = result
            logger.info(f"training job {job_id} concluído — candidato {version_id} pronto para deployment")
            return result
        except Exception as exc:  # job de background: nunca deixa exceção silenciosa sem estado persistido
            logger.exception(f"training job {job_id} falhou: {exc}")
            with session_scope() as session:
                record = session.get(TrainingJobRecord, job_id)
                record.status = TrainingJobStatus.FAILED
                record.finished_at = datetime.utcnow()
                record.error = str(exc)
            raise
