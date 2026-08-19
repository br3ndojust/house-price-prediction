from __future__ import annotations

import time

from app.domain.ports.model_repository import ModelRepository

_PROCESS_START = time.monotonic()


class GetHealth:
    def __init__(self, model_repository: ModelRepository):
        self._models = model_repository

    def liveness(self) -> dict:
        return {"status": "alive"}

    def readiness(self) -> dict:
        ready = self._models.is_ready()
        return {"status": "ready" if ready else "not_ready", "model_loaded": ready}

    def detailed(self) -> dict:
        contract = self._models.active_contract() if self._models.is_ready() else {}
        return {
            "status": "ready" if self._models.is_ready() else "not_ready",
            "uptime_seconds": round(time.monotonic() - _PROCESS_START, 1),
            "active_model_version": self._models.active_version() if self._models.is_ready() else None,
            "model": contract.get("model") if contract else None,
        }
