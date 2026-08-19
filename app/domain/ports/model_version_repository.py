from __future__ import annotations

from typing import Protocol

from app.domain.entities.model_version import ModelVersion


class ModelVersionRepository(Protocol):
    def add(self, version: ModelVersion) -> None:
        ...

    def get(self, version: str) -> ModelVersion | None:
        ...

    def list_all(self) -> list[ModelVersion]:
        ...

    def get_active(self) -> ModelVersion | None:
        ...

    def promote(self, version: str, alias: str | None = None) -> ModelVersion:
        """Marca `version` como active (demove a anterior para superseded). Nunca automático (P6) —
        só chamado a partir do use case PromoteModel, que exige API key. `alias` opcional sobrescreve
        `notes` da versão promovida (nome amigável, ex.: "modelo-outono-2026")."""
        ...
