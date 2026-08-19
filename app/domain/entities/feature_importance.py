from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class FeatureImportance:
    feature: str
    importance: float
    rank: int
    layer: str | None = None
    hypothesis: str | None = None
    used_by_active_model: bool = True
