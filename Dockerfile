FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# DeepFilterNet3 model baked into the image at a fixed path
ENV DF_PRETRAINED_MODELS_PATH=/app/models

# DPDFNet TFLite models — stored alongside DF3 under /app/models/dpdfnet
ENV DPDFNET_MODELS_DIR=/app/models/dpdfnet

# Which DPDFNet models to bake in (comma-separated; see preload_dpdfnet.py)
# Default: all four 16 kHz variants.  Set to "" to skip DPDFNet download.
ENV DPDFNET_DOWNLOAD_MODELS=baseline,dpdfnet2,dpdfnet4,dpdfnet8

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-venv \
    python3-pip \
    ffmpeg \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Create working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
# NOTE: --no-cache-dir ensures requirements.txt changes always trigger a fresh install
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt \
    && pip show faster-whisper | grep Version

# Download Silero VAD model during build.
# trust_repo=True suppresses the interactive trust prompt that PyTorch >=2.x
# would otherwise show in non-TTY environments — without it Docker build fails
# with EOFError because there is no stdin to read the y/N answer from.
RUN python -c "import torch; torch.hub.load('snakers4/silero-vad', 'silero_vad', force_reload=True, trust_repo=True)"

# Download DeepFilterNet3 model into the image (DF_PRETRAINED_MODELS_PATH=/app/models)
# Model files (~30 MB) are baked in so runtime never needs network access for this.
COPY preload_deepfilter.py /tmp/preload_deepfilter.py
RUN python /tmp/preload_deepfilter.py

# Download DPDFNet TFLite models into the image
# (~10–17 MB each depending on variant; set DPDFNET_DOWNLOAD_MODELS="" to skip)
COPY preload_dpdfnet.py /tmp/preload_dpdfnet.py
RUN if [ -n "$DPDFNET_DOWNLOAD_MODELS" ]; then \
        python /tmp/preload_dpdfnet.py; \
    else \
        echo "ℹ️  DPDFNET_DOWNLOAD_MODELS is empty — skipping DPDFNet download"; \
    fi

# Copy application code
COPY . .

# Create directories for uploads and outputs
RUN mkdir -p /app/uploads /app/outputs

# Expose Gradio port
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:7860/ || exit 1

# Run the application
CMD ["python", "app.py"]
