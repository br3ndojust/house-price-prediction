from __future__ import annotations

from pydantic import BaseModel, Field


class LivenessOut(BaseModel):
    status: str = Field(..., examples=["alive"])


class ReadinessOut(BaseModel):
    status: str = Field(..., examples=["ready"])
    model_loaded: bool


class DetailedHealthOut(BaseModel):
    status: str
    uptime_seconds: float
    active_model_version: str | None
    model: dict | None
