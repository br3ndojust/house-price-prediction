from __future__ import annotations

from typing import Protocol

from app.domain.entities.prediction import Prediction


class PredictionLogger(Protocol):
    def log(self, prediction: Prediction) -> str:
        """Persiste a predição, retorna o id gerado."""
        ...

    def get(self, prediction_id: str) -> Prediction | None:
        ...

    def list_recent(self, limit: int = 100, model_version: str | None = None) -> list[Prediction]:
        ...

    def list_with_feedback(self, model_version: str | None = None) -> list[Prediction]:
        ...

    def summary(self, model_version: str | None = None) -> dict:
        """Volume, latência agregada, distribuição por banda — fonte de /performance/summary."""
        ...

    def distinct_model_versions(self) -> list[str]:
        """Versões de modelo com pelo menos 1 predição registrada — fonte do seletor "Geral/por
        modelo" da tela Performance & Error Matrix."""
        ...
