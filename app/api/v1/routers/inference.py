from __future__ import annotations

import dataclasses

from fastapi import APIRouter, Depends, HTTPException

from app.core.container import Container, get_container
from app.domain.entities.property import Property

from ..schemas.inference import BatchPredictionIn, BatchPredictionOut, PredictionOut, PropertyIn

router = APIRouter(prefix="/predictions", tags=["Inferência"])


def _to_property(data: PropertyIn) -> Property:
    try:
        return Property(**data.model_dump())
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post(
    "",
    response_model=PredictionOut,
    summary="Prevê o preço de 1 imóvel",
    description=(
        "Recebe os atributos físicos + zipcode de um imóvel (mesmo schema de "
        "`future_unseen_examples.csv`, sem `id`/`date`/`price`) e retorna o preço previsto, a banda "
        "de preço (Entry/Standard/Premium/Luxury) e o cluster de perfil físico previstos — reusa "
        "`src/pipelines/inference.py::predict_price` (paridade treino/produção, P5)."
    ),
)
def predict(data: PropertyIn, container: Container = Depends(get_container)) -> PredictionOut:
    result = container.predict_price.execute(_to_property(data))
    return PredictionOut(**dataclasses.asdict(result))


@router.post(
    "/batch",
    response_model=BatchPredictionOut,
    summary="Prevê o preço de um lote de imóveis",
)
def predict_batch(data: BatchPredictionIn, container: Container = Depends(get_container)) -> BatchPredictionOut:
    properties = [_to_property(p) for p in data.properties]
    results = container.predict_batch.execute(properties)
    return BatchPredictionOut(predictions=[PredictionOut(**dataclasses.asdict(r)) for r in results])
