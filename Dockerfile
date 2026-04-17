# Floor-plan vectorizer — Railway-ready image.
# Uses opencv-python-headless (no X11) to keep the image slim.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    JOBS_DIR=/app/jobs \
    LOG_LEVEL=INFO

# System deps:
#   - tesseract-ocr: optional, used when --ocr-fallback is requested
#   - libgl1 / libglib2.0-0: required by OpenCV even in headless mode on some images
#   - curl: healthcheck convenience
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        tesseract-ocr \
        libgl1 \
        libglib2.0-0 \
        curl \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first for better layer caching.
COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy source.
COPY vectorize_floor ./vectorize_floor
COPY webapp ./webapp
COPY vectorize_floor.py ./vectorize_floor.py

# Prepare runtime dirs.
RUN mkdir -p "$JOBS_DIR" && chown -R 1000:1000 /app
USER 1000:1000

# Railway injects $PORT; fall back to 8080 for local use.
ENV PORT=8080
EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD curl -fsS "http://127.0.0.1:${PORT}/healthz" || exit 1

# Single worker keeps in-memory job state coherent.
# For scaling, move job state to Redis and bump --workers.
CMD ["sh", "-c", "uvicorn webapp.main:app --host 0.0.0.0 --port ${PORT} --workers 1"]
