from __future__ import annotations

import dataclasses

from fastapi import APIRouter, Depends, HTTPException

from app.core.container import Container, get_container
from app.core.security import require_api_key

from ..schemas.model import ModelInfoOut, ModelVersionOut, PromoteIn

router = APIRouter(prefix="/model", tags=["Modelo"])


@router.get("/info", response_model=ModelInfoOut, summary="Contrato ativo (production_contract.yaml)")
def info(container: Container = Depends(get_container)) -> dict:
    return container.get_model_info.execute()


@router.get("/versions", response_model=list[ModelVersionOut], summary="Histórico de versões/candidatos")
def versions(container: Container = Depends(get_container)) -> list[ModelVersionOut]:
    return [ModelVersionOut(**dataclasses.asdict(v)) for v in container.list_model_versions.execute()]


@router.post(
    "/promote",
    response_model=ModelVersionOut,
    summary="Aprova e ativa um candidato",
    description="Nunca automático (P6) — exige API key e uma versão candidata já treinada.",
    dependencies=[Depends(require_api_key)],
)
def promote(data: PromoteIn, container: Container = Depends(get_container)) -> ModelVersionOut:
    try:
        promoted = container.promote(data.version, data.alias)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return ModelVersionOut(**dataclasses.asdict(promoted))
