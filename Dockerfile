FROM ghcr.io/astral-sh/uv:0.9.13-python3.13-bookworm-slim AS requirements

WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv export --quiet --frozen --no-dev --format requirements.txt --output-file requirements.txt

FROM python:3.13-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8000

WORKDIR /app

RUN useradd --create-home --shell /usr/sbin/nologin appuser

COPY --from=requirements /app/requirements.txt ./
RUN pip install --no-cache-dir --requirement requirements.txt

COPY --chown=appuser:appuser api ./api
COPY --chown=appuser:appuser core ./core
COPY --chown=appuser:appuser infra ./infra
COPY --chown=appuser:appuser services ./services
COPY --chown=appuser:appuser main.py ./

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2).read()"

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
