FROM python:3.11-slim

WORKDIR /app

LABEL org.opencontainers.image.source="https://github.com/unsalemregulener/SignTurk" \
      org.opencontainers.image.title="SignTurk" \
      org.opencontainers.image.description="Real-time Turkish Sign Language recognition with a 3D avatar"

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Model weights come from the GitHub release and are checked against model-assets.json.
# MODELS=live builds a smaller image without the 226-class mode.
ARG MODELS=all
COPY model-assets.json .
COPY tools/download_models.py tools/download_models.py
COPY demo_assets_179/ demo_assets_179/
RUN python tools/download_models.py --bundle ${MODELS}

COPY backend.py database.py models.py live_pipeline.py live226.py landmark_smoother.py rate_limit.py ./
COPY signturk_runtime/ signturk_runtime/
COPY research/model_226/config.json research/model_226/config.json
COPY research/model_226/labels/ research/model_226/labels/
COPY text_processing/ text_processing/
COPY frontend/ frontend/
COPY landmarks/ landmarks/

# The app writes only uploads/ and, without DATABASE_URL, local_dev.db in /app;
# the model files stay owned by root.
RUN useradd --create-home --uid 10001 signturk \
    && mkdir -p uploads \
    && chown signturk:signturk /app /app/uploads
USER signturk

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=4)" || exit 1

CMD ["uvicorn", "backend:app", "--host", "0.0.0.0", "--port", "8000"]
