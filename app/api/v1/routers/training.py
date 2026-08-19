from __future__ import annotations

import dataclasses
import io
import json

import numpy as np
import pandas as pd
from fastapi import APIRouter, BackgroundTasks, Body, Depends, Form, HTTPException, UploadFile
from pydantic import ValidationError

from app.core.config import settings
from app.core.container import Container, get_container
from app.core.security import require_api_key
from src.models.train import METRIC_KEYS

from ..schemas.inference import PropertyIn
from ..schemas.training import AutoRetrainConfigIn, AutoRetrainConfigOut, TrainingJobIn, TrainingJobOut

router = APIRouter(prefix="/training", tags=["Treino"])

REQUIRED_TRAINING_COLUMNS = list(PropertyIn.model_fields.keys())


def _validate_metric(metric: str | None) -> str | None:
    if not metric:
        return None
    if metric not in METRIC_KEYS:
        raise HTTPException(
            status_code=422,
            detail=f"métrica desconhecida: '{metric}'. Válidas: {sorted(METRIC_KEYS)}.",
        )
    return metric


def _validate_test_size(test_size: float | None) -> float:
    if test_size is None:
        return 0.2
    if not 0.0 <= test_size <= 1.0:
        raise HTTPException(status_code=422, detail=f"test_size precisa estar entre 0 e 1, recebido: {test_size}")
    return test_size


def _official_feature_cols() -> list[str]:
    path = settings.data_dir / "trusted" / "feature_metadata.json"
    return json.loads(path.read_text(encoding="utf-8"))["model_features"]


def _validate_feature_cols(feature_cols: list[str] | None) -> list[str] | None:
    """`None`/vazio = usa o conjunto oficial (sem override). Se informado, precisa ser um
    subconjunto não vazio das features oficiais — nunca aceita nome arbitrário (P4/P5: a lista de
    features válidas tem uma única fonte, `data/trusted/feature_metadata.json`)."""
    if not feature_cols:
        return None
    official = set(_official_feature_cols())
    unknown = [f for f in feature_cols if f not in official]
    if unknown:
        raise HTTPException(
            status_code=422,
            detail=f"feature(s) desconhecida(s): {unknown}. Features válidas: {sorted(official)}.",
        )
    return feature_cols


@router.post(
    "/jobs",
    response_model=TrainingJobOut,
    summary="Dispara retraining em background",
    description=(
        "Reexecuta fases 10 (seleção de modelo) + 13 (refit + checagem em val) via "
        "`BackgroundTasks` — produz um candidato (`artifacts/candidates/model_candidate_<job_id>.pkl`), "
        "nunca ativa sozinho (P6). Corpo opcional (`feature_cols`) escolhe um subconjunto das 20 "
        "features oficiais para este treino — omitido usa todas. Acompanhe com GET /training/jobs/{id}."
    ),
    dependencies=[Depends(require_api_key)],
)
def create_job(
    background_tasks: BackgroundTasks,
    data: TrainingJobIn | None = Body(default=None),
    container: Container = Depends(get_container),
) -> TrainingJobOut:
    feature_cols = _validate_feature_cols(data.feature_cols if data else None)
    metric = _validate_metric(data.metric if data else None)
    job = container.trigger_training.execute(triggered_by="api", feature_cols=feature_cols, metric=metric)
    background_tasks.add_task(container.training_service.run, job.id)
    return TrainingJobOut(**dataclasses.asdict(job))


@router.post(
    "/jobs/from-feedback",
    response_model=TrainingJobOut,
    summary="Dispara retraining somando o dado novo já rotulado via feedback",
    description=(
        "Usa as predições que já receberam `POST /feedback` (preço de venda real) como dado novo "
        "de treino, somado ao TRAIN oficial (fase 16, docs/08_continuous_learning.md seções 2-3) — "
        "reusa a mesma engenharia de features de `src/pipelines/inference.py`. Nunca sobrescreve "
        "`data/trusted/features_contextual.parquet` nem ativa o candidato sozinho (P6). Corpo "
        "opcional (`feature_cols`) escolhe um subconjunto das 20 features oficiais. `test_size` "
        "(opcional, padrão 0.2) sorteia essa fração do dado novo para `split='test'`, o resto para "
        "`split='train'` — nunca 100% em `train`."
    ),
    dependencies=[Depends(require_api_key)],
)
def create_job_from_feedback(
    background_tasks: BackgroundTasks,
    data: TrainingJobIn | None = Body(default=None),
    container: Container = Depends(get_container),
) -> TrainingJobOut:
    feature_cols = _validate_feature_cols(data.feature_cols if data else None)
    metric = _validate_metric(data.metric if data else None)
    test_size = _validate_test_size(data.test_size if data else None)
    try:
        job, n_extra = container.trigger_training_from_feedback.execute(
            triggered_by="api:from_feedback", feature_cols=feature_cols, metric=metric, test_size=test_size
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    background_tasks.add_task(container.training_service.run, job.id)
    out = TrainingJobOut(**dataclasses.asdict(job))
    out.result = {"n_extra_rows_staged": n_extra}
    return out


@router.post(
    "/jobs/from-upload",
    response_model=TrainingJobOut,
    summary="Dispara retraining com upload de um lote de dado novo rotulado (CSV)",
    description=(
        "CSV com os mesmos 18 atributos de `future_unseen_examples.csv` + uma coluna de preço real "
        "(`value_column`). `column_order` é opcional — preencha só se o arquivo não tiver cabeçalho "
        "ou os nomes não baterem com os esperados (mesma lógica da página Inferência em Lote do "
        "portal). Cada linha é validada com o mesmo schema de `POST /predictions` (P5) antes de "
        "qualquer coisa ser somada ao TRAIN. `feature_cols` (opcional, separado por vírgula) escolhe "
        "um subconjunto das 20 features oficiais. `test_size` (opcional, padrão 0.2) sorteia essa "
        "fração do dado novo para `split='test'`, o resto para `split='train'` — nunca 100% em `train`."
    ),
    dependencies=[Depends(require_api_key)],
)
async def create_job_from_upload(
    background_tasks: BackgroundTasks,
    file: UploadFile,
    value_column: str = Form(..., description="Nome da coluna do CSV com o preço real de venda"),
    column_order: str | None = Form(
        None, description="Nomes das colunas do CSV, na ordem do arquivo, separados por vírgula (opcional)"
    ),
    has_header: bool = Form(True, description="Se `column_order` for informado: a 1ª linha do arquivo é cabeçalho?"),
    feature_cols: str | None = Form(
        None, description="Subconjunto das features oficiais, separado por vírgula (opcional — omitido usa todas)"
    ),
    metric: str | None = Form(
        None, description="Métrica para ranquear os candidatos: mae (padrão) | rmse | mape | r2"
    ),
    test_size: float | None = Form(
        None, description="Fração do dado novo sorteada para split='test' (0-1, padrão 0.2)"
    ),
    container: Container = Depends(get_container),
) -> TrainingJobOut:
    feature_cols_list = _validate_feature_cols(
        [f.strip() for f in feature_cols.split(",") if f.strip()] if feature_cols else None
    )
    metric = _validate_metric(metric)
    test_size = _validate_test_size(test_size)

    raw_bytes = await file.read()
    try:
        if column_order:
            override = [c.strip() for c in column_order.split(",") if c.strip()]
            df = pd.read_csv(io.BytesIO(raw_bytes), header=0 if has_header else None)
            if len(override) != df.shape[1]:
                raise HTTPException(
                    status_code=422,
                    detail=f"column_order tem {len(override)} nome(s), mas o arquivo tem {df.shape[1]} coluna(s).",
                )
            df.columns = override
        else:
            df = pd.read_csv(io.BytesIO(raw_bytes), header=0)
    except pd.errors.ParserError as exc:
        raise HTTPException(status_code=422, detail=f"CSV inválido: {exc}") from exc
    df.columns = [str(c).strip() for c in df.columns]

    missing = [c for c in REQUIRED_TRAINING_COLUMNS if c not in df.columns]
    if missing:
        raise HTTPException(
            status_code=422,
            detail=f"Colunas obrigatórias faltando: {missing}. Colunas encontradas: {list(df.columns)}.",
        )
    if value_column not in df.columns:
        raise HTTPException(status_code=422, detail=f"Coluna de valor real '{value_column}' não encontrada no CSV.")

    validated_rows, row_errors = [], []
    for i, row in enumerate(df[REQUIRED_TRAINING_COLUMNS].to_dict(orient="records")):
        try:
            validated_rows.append(PropertyIn(**row).model_dump())
        except ValidationError as exc:
            for err in exc.errors():
                row_errors.append({"linha_csv": i + 1, "campo": err["loc"][-1], "erro": err["msg"]})
    if row_errors:
        raise HTTPException(status_code=422, detail=row_errors)

    actual_prices = pd.to_numeric(df[value_column], errors="coerce").to_numpy()
    if np.isnan(actual_prices).any():
        bad_rows = [i + 1 for i in np.where(np.isnan(actual_prices))[0]]
        raise HTTPException(
            status_code=422, detail=f"Valor real inválido/vazio nas linhas do CSV: {bad_rows}"
        )

    raw_df = pd.DataFrame(validated_rows)
    try:
        job, n_extra = container.trigger_training_from_upload.execute(
            raw_df, actual_prices, triggered_by="api:from_upload", feature_cols=feature_cols_list,
            metric=metric, test_size=test_size,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    background_tasks.add_task(container.training_service.run, job.id)
    out = TrainingJobOut(**dataclasses.asdict(job))
    out.result = {"n_extra_rows_staged": n_extra}
    return out


@router.get("/jobs", response_model=list[TrainingJobOut], summary="Histórico de jobs")
def list_jobs(container: Container = Depends(get_container)) -> list[TrainingJobOut]:
    return [TrainingJobOut(**dataclasses.asdict(j)) for j in container.list_training_jobs.execute()]


@router.get("/jobs/{job_id}", response_model=TrainingJobOut, summary="Status/métricas/erro de um job")
def get_job(job_id: str, container: Container = Depends(get_container)) -> TrainingJobOut:
    job = container.get_training_job.execute(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"job {job_id} não encontrado")
    return TrainingJobOut(**dataclasses.asdict(job))


@router.get(
    "/auto-retrain-config",
    response_model=AutoRetrainConfigOut,
    summary="Configuração do gatilho automático de retraining (pipeline semi-automático)",
    description=(
        "Dispara `POST /training/jobs/from-feedback` sozinho quando o feedback novo acumulado desde "
        "o último disparo cruza `threshold` (checado a cada `POST /feedback`). Nunca promove sozinho "
        "(P6) — o candidato resultante fica `candidate`, esperando aprovação humana em "
        "`POST /model/promote`, igual qualquer outro."
    ),
)
def get_auto_retrain_config(container: Container = Depends(get_container)) -> AutoRetrainConfigOut:
    config = container.get_auto_retrain_config.execute()
    total_feedback = container.prediction_logger.summary().get("predictions_with_feedback", 0)
    return AutoRetrainConfigOut(
        threshold=config.threshold, enabled=config.enabled,
        consumed_feedback_count=config.consumed_feedback_count,
        new_since_last=max(0, total_feedback - config.consumed_feedback_count),
        updated_at=config.updated_at,
    )


@router.put(
    "/auto-retrain-config",
    response_model=AutoRetrainConfigOut,
    summary="Ajusta o limite/liga-desliga do gatilho automático",
    dependencies=[Depends(require_api_key)],
)
def update_auto_retrain_config(
    data: AutoRetrainConfigIn, container: Container = Depends(get_container)
) -> AutoRetrainConfigOut:
    try:
        container.update_auto_retrain_config.execute(threshold=data.threshold, enabled=data.enabled)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return get_auto_retrain_config(container)
