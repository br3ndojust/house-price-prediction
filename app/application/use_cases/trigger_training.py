from __future__ import annotations

from app.domain.entities.training_job import TrainingJob
from app.domain.ports.training_service import TrainingService


class TriggerTraining:
    """Só cria o registro do job (pending) — o agendamento em background é responsabilidade da camada
    `api` (BackgroundTasks é detalhe de framework, não pertence à application, P5)."""

    def __init__(self, training_service: TrainingService):
        self._training = training_service

    def execute(self, triggered_by: str | None = None, feature_cols: list[str] | None = None,
                metric: str | None = None) -> TrainingJob:
        job = self._training.create_job(triggered_by)
        if feature_cols:
            self._training.stage_feature_cols(job.id, feature_cols)
        if metric:
            self._training.stage_metric(job.id, metric)
        return job


class GetTrainingJob:
    def __init__(self, training_service: TrainingService):
        self._training = training_service

    def execute(self, job_id: str) -> TrainingJob | None:
        return self._training.get_job(job_id)


class ListTrainingJobs:
    def __init__(self, training_service: TrainingService):
        self._training = training_service

    def execute(self) -> list[TrainingJob]:
        return self._training.list_jobs()
