FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/
COPY memory/ ./memory/
COPY config.yaml .
COPY openenv.yaml .
COPY inference.py .
COPY app.py .
COPY assets/ ./assets/

EXPOSE 7860

CMD ["python", "app.py"]
