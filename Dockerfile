FROM python:3.11-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl git && rm -rf /var/lib/apt/lists/*

# HF Spaces requires port 7860
ENV PORT=7860
ENV GRADIO_SERVER_PORT=7860
ENV GRADIO_SERVER_NAME=0.0.0.0

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all application code
COPY app/ ./app/
COPY demo/ ./demo/
COPY agents/ ./agents/
COPY memory/ ./memory/
COPY training/ ./training/
COPY tests/ ./tests/
COPY config.yaml .

# Create memory directory for runtime
RUN mkdir -p /app/memory

# Health check against FastAPI
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Expose both ports
EXPOSE 7860 8000

# Start FastAPI (background) + Gradio (foreground on 7860)
CMD bash -c "uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1 & sleep 3 && python3 demo/app.py"
