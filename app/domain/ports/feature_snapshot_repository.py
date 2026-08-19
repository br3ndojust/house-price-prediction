from __future__ import annotations

from typing import Protocol

from app.domain.entities.feature_snapshot import FeatureSnapshot


class FeatureSnapshotRepository(Protocol):
    def add(self, snapshot: FeatureSnapshot) -> None:
        ...

    def get(self, version: str) -> FeatureSnapshot | None:
        ...

    def list_all(self) -> list[FeatureSnapshot]:
        ...
