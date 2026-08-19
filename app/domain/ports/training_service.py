from __future__ import annotations

from typing import Protocol

import pandas as pd

from app.domain.entities.training_job import TrainingJob


class TrainingService(Protocol):
    """Orquestra o ciclo de vida do job de treino (execução + persistência do status).

    Reusa src/scripts/train_model.py + src/scripts/finalize_model.py refatorados em funções chamáveis (P5) —
    nunca promove o candidato sozinho (P6): o job só produz `artifacts/model_candidate.pkl` + métricas;
    ativação real passa por PromoteModel (`/model/promote`, com API key).
    """

    def create_job(self, triggered_by: str | None) -> TrainingJob:
        ...

    def run(self, job_id: str) -> None:
        """Executa o job (chamado via BackgroundTasks), atualiza status/resultado ao final."""
        ...

    def get_job(self, job_id: str) -> TrainingJob | None:
        ...

    def list_jobs(self) -> list[TrainingJob]:
        ...

    def stage_extra_rows(self, job_id: str, extra_rows: pd.DataFrame) -> None:
        """Dado novo rotulado (fase 16) a somar ao TRAIN oficial quando `run(job_id)` disparar."""
        ...

    def stage_feature_cols(self, job_id: str, feature_cols: list[str]) -> None:
        """Subconjunto de features escolhido pelo usuário (tela Treino) para este job — ausente =
        usa o conjunto oficial travado."""
        ...
