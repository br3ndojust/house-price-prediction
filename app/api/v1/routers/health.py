from __future__ import annotations

from fastapi import APIRouter, Depends

from app.core.container import Container, get_container

from ..schemas.health import DetailedHealthOut, LivenessOut, ReadinessOut

router = APIRouter(prefix="/health", tags=["Health"])


@router.get("/live", response_model=LivenessOut, summary="Liveness — processo de pé")
def live() -> LivenessOut:
    return LivenessOut(status="alive")


@router.get("/ready", response_model=ReadinessOut, summary="Readiness — modelo carregado")
def ready(container: Container = Depends(get_container)) -> ReadinessOut:
    return ReadinessOut(**container.get_health.readiness())


@router.get(
    "/detailed",
    response_model=DetailedHealthOut,
    summary="Versão do modelo ativo, uptime, contrato",
)
def detailed(container: Container = Depends(get_container)) -> DetailedHealthOut:
    return DetailedHealthOut(**container.get_health.detailed())
