FROM python:3.11-slim-bookworm

# Install minimal system dependencies
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

ENV OCEANAI_LANG=ru \
    OCEANAI_CORPUS=mupta \
    OCEANAI_DISK=googledisk \
    OCEANAI_CACHE_DIR=/app/oceanai-cache

# Copy requirements and install
COPY requirements.txt .
# ENV TEST_MINIMAL=true
# Install CPU-only PyTorch first (lightweight)
RUN pip install --no-cache-dir --retries 20 --timeout 600 \
    torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cpu

# Install generic requirements
RUN pip install --no-cache-dir --retries 20 --timeout 600 -r requirements.txt

# Install OceanAI and scientific deps without re-triggering GPU torch
RUN pip install --no-cache-dir --retries 20 --timeout 600 oceanai==1.0.0a48 --no-deps && \
    pip install --no-cache-dir --retries 20 --timeout 600 \
        scikit-learn pandas numpy librosa opencv-python-headless transformers sentencepiece protobuf==3.20.3

# Pre-download HuggingFace models (Whisper for ASR, BERT for text analysis)
# This avoids runtime "No space left on device" errors on serverless containers
RUN python -c "from transformers import pipeline; pipeline('automatic-speech-recognition', model='openai/whisper-base')" && \
    python -c "from transformers import AutoTokenizer, AutoModel; AutoTokenizer.from_pretrained('bert-base-multilingual-cased'); AutoModel.from_pretrained('bert-base-multilingual-cased')"

# Pre-cache OceanAI weights and helper models to avoid cold-start downloads.
RUN mkdir -p "${OCEANAI_CACHE_DIR}" && \
    python - <<'PY'
import os
from pathlib import Path

from oceanai.modules.lab.build import Run

lang = os.getenv("OCEANAI_LANG", "ru")
corpus_env = os.getenv("OCEANAI_CORPUS")
corpus = corpus_env or ("mupta" if lang.lower().startswith("ru") else "fi")
disk = os.getenv("OCEANAI_DISK", "googledisk")
cache = os.getenv("OCEANAI_CACHE_DIR", "/app/oceanai-cache")

Path(cache).mkdir(parents=True, exist_ok=True)
ocean = Run(lang=lang, metadata=False)
ocean.path_to_save_ = cache
ocean.path_to_logs_ = os.path.join(cache, "logs")

def preload(ocean, corpus, disk, lang):
    ocean.load_audio_model_hc(out=False)
    ocean.load_audio_model_nn(out=False)
    ocean.load_audio_model_weights_hc(
        url=ocean.weights_for_big5_["audio"][corpus]["hc"][disk],
        force_reload=False,
        out=False,
    )
    ocean.load_audio_model_weights_nn(
        url=ocean.weights_for_big5_["audio"][corpus]["nn"][disk],
        force_reload=False,
        out=False,
    )

    ocean.load_video_model_hc(lang=lang, out=False)
    ocean.load_video_model_deep_fe(out=False)
    ocean.load_video_model_nn(out=False)
    ocean.load_video_model_weights_hc(
        url=ocean.weights_for_big5_["video"][corpus]["hc"][disk],
        force_reload=False,
        out=False,
    )
    ocean.load_video_model_weights_deep_fe(
        url=ocean.weights_for_big5_["video"][corpus]["fe"][disk],
        force_reload=False,
        out=False,
    )
    ocean.load_video_model_weights_nn(
        url=ocean.weights_for_big5_["video"][corpus]["nn"][disk],
        force_reload=False,
        out=False,
    )

    ocean.load_text_features(out=False)
    ocean.setup_translation_model(out=False)
    ocean.setup_bert_encoder(force_reload=False, out=False)
    ocean.load_text_model_hc(corpus=corpus, out=False)
    ocean.load_text_model_weights_hc(
        url=ocean.weights_for_big5_["text"][corpus]["hc"][disk],
        force_reload=False,
        out=False,
    )
    ocean.load_text_model_nn(corpus=corpus, out=False)
    ocean.load_text_model_weights_nn(
        url=ocean.weights_for_big5_["text"][corpus]["nn"][disk],
        force_reload=False,
        out=False,
    )

    ocean.load_avt_model_b5(out=False)
    ocean.load_avt_model_weights_b5(
        url=ocean.weights_for_big5_["avt"][corpus]["b5"][disk],
        force_reload=False,
        out=False,
    )

preload(ocean, corpus, disk, lang)
print(f"OceanAI assets cached to {cache} (corpus={corpus}, disk={disk}, lang={lang})")
PY

# Copy handler code
COPY handler.py .

# Create temp directory
RUN mkdir -p /tmp

# Expose port for Serverless Container
EXPOSE 8080

# Install Flask and gunicorn for HTTP server
RUN pip install gunicorn flask

# Create Flask wrapper for the handler
RUN echo 'from flask import Flask, request, jsonify\nfrom handler import handler\napp = Flask(__name__)\n@app.route("/", methods=["POST"])\ndef invoke():\n    event = request.get_json(force=True)\n    result = handler(event, None)\n    return jsonify(result), result.get("statusCode", 200)\n@app.route("/health", methods=["GET"])\ndef health():\n    return "OK", 200' > app.py

# Run with gunicorn (1 hour timeout for long videos)
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--timeout", "3600", "app:app"]
