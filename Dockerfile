FROM ubuntu:22.04

# Avoid interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies (build tools, libgl for OpenCV, curl for healthchecks)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-dev \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependency requirements
COPY backend/requirements.txt /app/backend/requirements.txt

# Install PyTorch CPU wheels specifically compatible with ARM64 / aarch64
# and Python dependencies
RUN pip3 install --no-cache-dir --upgrade pip setuptools wheel && \
    pip3 install --no-cache-dir -r /app/backend/requirements.txt

# Copy application backend, configs, and sample images
COPY backend/ /app/backend/
COPY configs/ /app/configs/
COPY artifacts/roadvision_model_artifacts/model_config.json /app/artifacts/roadvision_model_artifacts/model_config.json
COPY artifacts/roadvision_model_artifacts/class_mapping.json /app/artifacts/roadvision_model_artifacts/class_mapping.json

# Create dedicated directory for persistent model caching and SQLite DB
RUN mkdir -p /models /app/backend/data && chmod -R 777 /models /app/backend/data

# Environment defaults
ENV MODEL_CHECKPOINT_PATH=/models/best_model.pth
ENV PORT=8000
ENV HOST=0.0.0.0

WORKDIR /app/backend

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT}"]
