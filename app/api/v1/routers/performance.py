from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from app.core.container import Container, get_container

from ..schemas.performance import ErrorMatrixOut, PerformanceSummaryOut, PredictionRecordOut

router = APIRouter(prefix="/performance", tags=["Performance"])


@router.get(
    "/summary",
    response_model=PerformanceSummaryOut,
    summary="Volume de predição e distribuição por banda/cluster",
    description="`model_version` opcional filtra tudo (volume, distribuição) só pras predições daquela versão de modelo.",
)
def summary(
    model_version: str | None = Query(None, description="Filtra só predições dessa versão de modelo"),
    container: Container = Depends(get_container),
) -> PerformanceSummaryOut:
    return PerformanceSummaryOut(**container.get_performance_summary.execute(model_version=model_version))


@router.get(
    "/model-versions",
    response_model=list[str],
    summary="Versões de modelo com predição registrada",
    description="Fonte do seletor Geral/por modelo da tela Performance & Error Matrix do portal.",
)
def prediction_model_versions(container: Container = Depends(get_container)) -> list[str]:
    return container.list_prediction_model_versions.execute()


@router.get(
    "/predictions",
    response_model=list[PredictionRecordOut],
    summary="Lista predições recentes (com valor real, quando já registrado)",
    description=(
        "Fonte da tabela de predições da tela Performance & Error Matrix do portal. "
        "`with_feedback=true` filtra só as que já têm `POST /feedback` registrado. `model_version` "
        "opcional filtra só predições daquela versão de modelo."
    ),
)
def list_predictions(
    with_feedback: bool = Query(False, description="Filtra só predições com valor real registrado"),
    limit: int = Query(100, ge=1, le=1000),
    model_version: str | None = Query(None, description="Filtra só predições dessa versão de modelo"),
    container: Container = Depends(get_container),
) -> list[PredictionRecordOut]:
    predictions = container.list_predictions.execute(
        with_feedback=with_feedback, limit=limit, model_version=model_version
    )
    return [
        PredictionRecordOut(
            id=p.id, created_at=p.created_at, predicted_price=p.predicted_price,
            price_band=p.price_band, property_cluster=p.property_cluster,
            model_version=p.model_version, confidence_score=p.confidence_score,
            confidence_category=p.confidence_category, actual_price=p.actual_price,
            feedback_recorded_at=p.feedback_recorded_at,
        )
        for p in predictions
    ]


@router.get(
    "/error-matrix",
    response_model=ErrorMatrixOut,
    summary="Error Matrix real sobre predições com feedback",
    description=(
        "Reusa `src/evaluation/error_matrix.py` (fase 09/11) sobre as predições que já receberam "
        "`POST /feedback` (preço de venda real) — nunca reporta só a métrica agregada (P3/P4). "
        "`model_version` opcional filtra só predições daquela versão de modelo."
    ),
)
def error_matrix(
    model_version: str | None = Query(None, description="Filtra só predições dessa versão de modelo"),
    container: Container = Depends(get_container),
) -> ErrorMatrixOut:
    out = container.get_error_matrix.execute(model_version=model_version)
    return ErrorMatrixOut(
        n=out["n"], global_=out.get("global"), by_price_band=out["by_price_band"],
        by_property_cluster=out["by_property_cluster"], note=out.get("note"),
    )
