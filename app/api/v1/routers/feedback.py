from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException

from app.core.container import Container, get_container
from app.core.security import require_api_key

from ..schemas.feedback import FeedbackIn, FeedbackOut

router = APIRouter(prefix="/feedback", tags=["Feedback"])


@router.post(
    "",
    response_model=FeedbackOut,
    summary="Registra o preço de venda real de uma predição",
    description=(
        "Fecha o ciclo de aprendizado contínuo (docs/08_continuous_learning.md, seção 2). Também "
        "checa o gatilho automático de retraining (GET/PUT /training/auto-retrain-config) — se o "
        "feedback acumulado cruzar o limite configurado, dispara um retraining em background sozinho "
        "(nunca promove sozinho, P6)."
    ),
    dependencies=[Depends(require_api_key)],
)
def record_feedback(
    data: FeedbackIn, background_tasks: BackgroundTasks, container: Container = Depends(get_container)
) -> FeedbackOut:
    try:
        container.record_feedback.execute(data.prediction_id, data.actual_price)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    auto_job = container.check_auto_retrain_trigger.execute()
    if auto_job is not None:
        background_tasks.add_task(container.training_service.run, auto_job.id)

    return FeedbackOut(status="recorded", auto_retrain_job_id=auto_job.id if auto_job else None)
