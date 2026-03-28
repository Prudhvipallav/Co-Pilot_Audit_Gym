FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl git && rm -rf /var/lib/apt/lists/*

ENV PORT=7860

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/
COPY agents/ ./agents/
COPY training/ ./training/
COPY tests/ ./tests/
COPY config.yaml .

RUN mkdir -p /app/memory

HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
  CMD curl -f http://localhost:7860/health || exit 1

EXPOSE 7860

CMD uvicorn app.main:app --host 0.0.0.0 --port 7860 --workers 1
