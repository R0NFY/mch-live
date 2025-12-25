# syntax=docker/dockerfile:1.7
#
# Goal: keep image <200MB by NEVER baking model weights into the image.
# Model weights (OceanAI + HF/transformers + torch caches) must live in a mounted volume at /models.

FROM python:3.11-slim-bookworm AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install minimal runtime system deps (no compilers/build tools).
RUN set -eux; \
    # Debian mirrors sometimes return transient 502s; add retries and prefer HTTPS.
    if [ -f /etc/apt/sources.list.d/debian.sources ]; then \
        sed -i 's|http://deb.debian.org|https://deb.debian.org|g' /etc/apt/sources.list.d/debian.sources; \
        sed -i 's|http://security.debian.org|https://security.debian.org|g' /etc/apt/sources.list.d/debian.sources; \
    elif [ -f /etc/apt/sources.list ]; then \
        sed -i 's|http://deb.debian.org|https://deb.debian.org|g' /etc/apt/sources.list; \
        sed -i 's|http://security.debian.org|https://security.debian.org|g' /etc/apt/sources.list; \
    fi; \
    apt-get update -o Acquire::Retries=5; \
    apt-get install -y --no-install-recommends \
        ffmpeg \
        libgl1 \
        libglib2.0-0 \
        ca-certificates \
    ; \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Configure model volume pattern.
# - MODELS_DIR is the canonical mount point
# - *_CACHE vars route weights to the mounted volume
ENV MODELS_DIR=/models \
    HF_HOME=/models/hf \
    HUGGINGFACE_HUB_CACHE=/models/hf/hub \
    TRANSFORMERS_CACHE=/models/hf/transformers \
    TORCH_HOME=/models/torch \
    XDG_CACHE_HOME=/models/xdg-cache \
    OCEANAI_CACHE_DIR=/models/oceanai-cache \
    OCEANAI_LANG=ru \
    OCEANAI_CORPUS=mupta \
    OCEANAI_DISK=googledisk

FROM ghcr.io/astral-sh/uv:latest AS uv

FROM base AS builder
COPY --from=uv /uv /usr/local/bin/uv

# Install deps with uv; use BuildKit cache mount for instant rebuilds.
COPY requirements.txt /app/requirements.txt
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system -r /app/requirements.txt

FROM base AS runtime

# Copy the installed environment from builder (multi-stage: runtime stays lean).
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy only application code (NO model weights).
COPY handler.py /app/handler.py
COPY main.py /app/main.py
COPY model_paths.py /app/model_paths.py
COPY bigfive.py /app/bigfive.py
COPY validate_env.py /app/validate_env.py
COPY app.py /app/app.py

RUN mkdir -p /tmp

VOLUME ["/models"]
EXPOSE 8080

# Run with gunicorn (1 hour timeout for long videos)
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--timeout", "3600", "app:app"]
