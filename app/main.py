"""App factory — startup (carrega modelo 1x), routers, exception handlers, observabilidade."""
from __future__ import annotations

import logging
import sys
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.api.v1.routers import (  # noqa: E402
    confidence,
    drift,
    explainability,
    feedback,
    health,
    inference,
    model,
    performance,
    training,
)
from app.core.config import settings  # noqa: E402
from app.core.container import get_container  # noqa: E402
from app.infrastructure.observability.logging_config import configure_logging  # noqa: E402
from app.infrastructure.observability.metrics import REQUEST_COUNT, REQUEST_LATENCY  # noqa: E402
from app.infrastructure.persistence.db import create_all_tables  # noqa: E402

logger = logging.getLogger("app")


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    create_all_tables()
    get_container()  # carrega o modelo 1x no boot (P6: stateless, cache em memória)
    logger.info("app iniciado — modelo carregado")
    yield


app = FastAPI(
    title="House Pricing API",
    description=(
        "API de inferência, monitoramento e retreinamento para o modelo de previsão de preço de "
        "imóveis (Seattle). Serve `artifacts/model_final.pkl` (fase 13 do pipeline de "
        "ML, MAE em `val` $72.517) atrás de uma camada, reusando `src/` sem "
        "duplicar lógica (P5). Ver `app/README.md` para a arquitetura completa e "
        "`docs/07_deploy_strategy.md`/`docs/08_continuous_learning.md` para o desenho original."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def observability_middleware(request: Request, call_next):
    request_id = str(uuid.uuid4())
    start = time.perf_counter()
    response = await call_next(request)
    duration = time.perf_counter() - start

    route_path = request.scope.get("route").path if request.scope.get("route") else request.url.path
    REQUEST_COUNT.labels(request.method, route_path, response.status_code).inc()
    REQUEST_LATENCY.labels(request.method, route_path).observe(duration)

    response.headers["X-Request-ID"] = request_id
    logger.info(
        f"{request.method} {route_path} -> {response.status_code} ({duration * 1000:.0f}ms)",
        extra={"request_id": request_id},
    )
    return response


@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError) -> JSONResponse:
    return JSONResponse(status_code=422, content={"detail": str(exc)})


@app.get("/metrics", tags=["Observabilidade"], summary="Métricas Prometheus")
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


API_V1 = "/api/v1"
app.include_router(inference.router, prefix=API_V1)
app.include_router(health.router, prefix=API_V1)
app.include_router(performance.router, prefix=API_V1)
app.include_router(feedback.router, prefix=API_V1)
app.include_router(model.router, prefix=API_V1)
app.include_router(training.router, prefix=API_V1)
app.include_router(drift.router, prefix=API_V1)
app.include_router(explainability.router, prefix=API_V1)
app.include_router(confidence.router, prefix=API_V1)
