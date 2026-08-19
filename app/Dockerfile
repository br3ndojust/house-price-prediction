# Build context = raiz do repo (docker-compose.yml usa `context: .`, `dockerfile: app/Dockerfile`) —
# a API reusa src/, scripts/, configs/ do pipeline de ML (P5: nenhuma lógica duplicada/reescrita).

FROM python:3.11-slim AS builder

WORKDIR /build
COPY requirements.txt requirements.txt
COPY app/requirements.txt app/requirements.txt
RUN pip install --user --no-cache-dir -r app/requirements.txt

FROM python:3.11-slim AS runtime

RUN useradd --create-home --uid 1000 appuser
WORKDIR /app

COPY --from=builder --chown=appuser:appuser /root/.local /home/appuser/.local
COPY --chown=appuser:appuser app/ app/
COPY --chown=appuser:appuser src/ src/
COPY --chown=appuser:appuser configs/ configs/
COPY --chown=appuser:appuser requirements.txt .

# artifacts/, data/, app_data/ chegam via volume (docker-compose) — nunca embutidos na imagem
# (P6: model registry é estado operacional, não artefato de build; dataset/DB persistem entre restarts).
RUN mkdir -p artifacts data app_data && chown -R appuser:appuser artifacts data app_data

ENV PATH=/home/appuser/.local/bin:$PATH \
    PYTHONPATH=/app \
    PYTHONUNBUFFERED=1

USER appuser
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/v1/health/live', timeout=3)" || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
