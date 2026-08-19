from __future__ import annotations

import dataclasses

from fastapi import APIRouter, Depends, HTTPException

from app.core.container import Container, get_container

from ..schemas.confidence import ConfidenceCalibrationSummaryOut, PredictionConfidenceOut
from ..schemas.inference import PropertyIn
from .inference import _to_property

router = APIRouter(tags=["Confiança"])


@router.post(
    "/predictions/confidence",
    response_model=PredictionConfidenceOut,
    summary="Matriz/Score de Confiança detalhado para 1 imóvel",
    description=(
        "Mesmo input de POST /predictions + o breakdown completo do Confidence Score (0-100): erro "
        "esperado, distância ao TRAIN, cobertura no espaço de features, tamanho de amostra do "
        "segmento, flag de segmento difícil (waterfront/grade≥10). Calibrado em TEST, verificado uma "
        "única vez em VAL — ver `docs/09_confidence_matrix.md`. Não loga uma nova predição (leitura, "
        "como /predictions/explain)."
    ),
)
def prediction_confidence(data: PropertyIn, container: Container = Depends(get_container)) -> PredictionConfidenceOut:
    try:
        confidence = container.get_prediction_confidence.execute(_to_property(data))
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return PredictionConfidenceOut(**dataclasses.asdict(confidence))


@router.get(
    "/model/confidence-calibration",
    response_model=ConfidenceCalibrationSummaryOut,
    summary="Metodologia e cortes de categoria da Matriz de Confiança",
    description=(
        "Metodologia, cortes de categoria (alta_min/moderada_min/baixa_min) e quartis de APE "
        "calibrados em TEST — `ready: false` se `artifacts/confidence_calibration.json` ainda não "
        "foi gerado (`python src/scripts/build_confidence_matrix.py`)."
    ),
)
def confidence_calibration(container: Container = Depends(get_container)) -> ConfidenceCalibrationSummaryOut:
    return ConfidenceCalibrationSummaryOut(**container.get_confidence_calibration_summary.execute())
