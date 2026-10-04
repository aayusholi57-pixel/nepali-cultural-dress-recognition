# syntax=docker/dockerfile:1

FROM python:3.10-slim AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH"

WORKDIR /build

RUN python -m venv "$VIRTUAL_ENV" \
    && pip install --upgrade pip

COPY requirements-docker.txt .

RUN pip install --no-cache-dir -r requirements-docker.txt


FROM python:3.10-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH" \
    AWS_DEFAULT_REGION=ap-southeast-2 \
    AWS_REGION=ap-southeast-2

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --shell /usr/sbin/nologin --uid 10001 appuser

COPY --from=builder /opt/venv /opt/venv

COPY --chown=appuser:appuser api/ ./api/
COPY --chown=appuser:appuser src/ ./src/

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=5)"

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]