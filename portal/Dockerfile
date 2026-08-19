# Build context = raiz do repo (docker-compose.yml usa `context: .`, `dockerfile: portal/Dockerfile`).
# O portal é um cliente HTTP puro — não copia src/ nem app/ (nunca importa o backend diretamente).

FROM python:3.11-slim AS runtime

RUN useradd --create-home --uid 1000 portaluser
WORKDIR /app

COPY portal/requirements.txt portal/requirements.txt
RUN pip install --no-cache-dir -r portal/requirements.txt

COPY --chown=portaluser:portaluser portal/ portal/

ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1

USER portaluser
EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health', timeout=3)" || exit 1

CMD ["streamlit", "run", "portal/Home.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
