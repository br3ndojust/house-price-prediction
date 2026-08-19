from __future__ import annotations

from app.domain.entities.model_version import ModelVersion
from app.domain.ports.model_repository import ModelRepository
from app.domain.ports.model_version_repository import ModelVersionRepository


class GetModelInfo:
    def __init__(self, model_repository: ModelRepository):
        self._models = model_repository

    def execute(self) -> dict:
        return self._models.active_contract()


class ListModelVersions:
    def __init__(self, model_version_repository: ModelVersionRepository):
        self._versions = model_version_repository

    def execute(self) -> list[ModelVersion]:
        return self._versions.list_all()
