"""Pipeline semi-automático de retraining — gatilho por volume de dado novo (feedback acumulado),
sempre com aprovação humana antes de ativar (P6). Ver docs/entregavel_04_aprendizado_continuo.md,
seção 10, pra honestidade sobre o que é implementação real vs. estratégia."""
from __future__ import annotations

import logging

from app.domain.entities.auto_retrain_config import AutoRetrainConfig
from app.domain.entities.training_job import TrainingJob
from app.domain.ports.prediction_logger import PredictionLogger

logger = logging.getLogger("app.auto_retrain")


class GetAutoRetrainConfig:
    def __init__(self, config_repository):
        self._repo = config_repository

    def execute(self) -> AutoRetrainConfig:
        return self._repo.get()


class UpdateAutoRetrainConfig:
    def __init__(self, config_repository):
        self._repo = config_repository

    def execute(self, threshold: int | None = None, enabled: bool | None = None) -> AutoRetrainConfig:
        config = self._repo.get()
        if threshold is not None:
            if threshold < 1:
                raise ValueError("threshold precisa ser >= 1")
            config.threshold = threshold
        if enabled is not None:
            config.enabled = enabled
        self._repo.save(config)
        return self._repo.get()


class CheckAutoRetrainTrigger:
    """Chamado depois de cada `POST /feedback` — dispara um retraining automático quando o volume de
    feedback novo acumulado desde o último disparo cruza `threshold`. Nunca lança: um gatilho
    automático é bookkeeping best-effort, não pode derrubar o registro de feedback que o chamou."""

    def __init__(self, config_repository, prediction_logger: PredictionLogger, trigger_training_from_feedback):
        self._repo = config_repository
        self._logger = prediction_logger
        self._trigger = trigger_training_from_feedback

    def execute(self) -> TrainingJob | None:
        try:
            config = self._repo.get()
            if not config.enabled:
                return None
            total_feedback = self._logger.summary().get("predictions_with_feedback", 0)
            new_since_last = total_feedback - config.consumed_feedback_count
            if new_since_last < config.threshold:
                return None

            job, n_extra = self._trigger.execute(triggered_by="auto:volume_threshold")
            config.consumed_feedback_count = total_feedback
            self._repo.save(config)
            logger.info(
                f"gatilho automático disparado — {new_since_last} feedback(s) novo(s) acumulado(s) "
                f"(limite {config.threshold}), job {job.id} criado com {n_extra} linha(s)"
            )
            return job
        except Exception:
            logger.exception("falha ao checar/disparar gatilho automático de retraining — feedback já foi registrado")
            return None
