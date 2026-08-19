from __future__ import annotations

import dataclasses

from fastapi import APIRouter, Depends, HTTPException

from app.core.container import Container, get_container
from app.core.security import require_api_key

from ..schemas.drift import DriftReportOut

router = APIRouter(prefix="/drift", tags=["Drift"])


@router.post(
    "/evaluate",
    response_model=DriftReportOut,
    summary="Computa um relatório de drift novo",
    description=(
        "PSI por feature (predições recentes vs. TRAIN) + distância média no espaço de features "
        "(reusa `src/evaluation/geographic_coverage.py::feature_space_coverage`, fase 04)."
    ),
    dependencies=[Depends(require_api_key)],
)
def evaluate(container: Container = Depends(get_container)) -> DriftReportOut:
    return DriftReportOut(**dataclasses.asdict(container.evaluate_drift.execute()))


@router.get("/reports", response_model=list[DriftReportOut], summary="Histórico de relatórios de drift")
def reports(container: Container = Depends(get_container)) -> list[DriftReportOut]:
    return [DriftReportOut(**dataclasses.asdict(r)) for r in container.list_drift_reports.execute()]


@router.get("/reports/latest", response_model=DriftReportOut, summary="Relatório de drift mais recente")
def latest(container: Container = Depends(get_container)) -> DriftReportOut:
    report = container.get_latest_drift_report.execute()
    if report is None:
        raise HTTPException(status_code=404, detail="nenhum relatório de drift ainda")
    return DriftReportOut(**dataclasses.asdict(report))
