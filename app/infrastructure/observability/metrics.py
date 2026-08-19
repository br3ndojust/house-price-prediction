"""`GET /metrics` (formato Prometheus) — contadores de request/latência/predição por segmento, gauge
de drift, via `prometheus-client`."""
from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

REQUEST_COUNT = Counter(
    "http_requests_total", "Total de requisições HTTP", ["method", "path", "status_code"]
)
REQUEST_LATENCY = Histogram(
    "http_request_duration_seconds", "Latência de requisição HTTP (segundos)", ["method", "path"]
)
PREDICTION_COUNT = Counter(
    "predictions_total", "Total de predições de preço", ["price_band"]
)
DRIFT_STATUS_GAUGE = Gauge(
    "drift_status", "1 se o último relatório de drift tem este status", ["status"]
)
