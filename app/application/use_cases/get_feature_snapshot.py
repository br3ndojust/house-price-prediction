from __future__ import annotations

from app.domain.entities.feature_snapshot import FeatureSnapshot
from app.domain.ports.feature_snapshot_repository import FeatureSnapshotRepository


class GetFeatureSnapshot:
    def __init__(self, repository: FeatureSnapshotRepository):
        self._repo = repository

    def execute(self, version: str) -> FeatureSnapshot | None:
        return self._repo.get(version)


class ListFeatureSnapshots:
    def __init__(self, repository: FeatureSnapshotRepository):
        self._repo = repository

    def execute(self) -> list[FeatureSnapshot]:
        return self._repo.list_all()
