# ⚠️  Base image is ~20 GB on first pull (PyTorch + CUDA + NeMo pre-installed).
#     Subsequent builds use the local cache.
#     Requires NVIDIA GPU + nvidia-container-toolkit on the Docker host.
#
# To use a CPU-only build (slower), replace the FROM line with:
#   FROM python:3.11-slim
# and add these RUN steps before the pip install:
#   RUN apt-get update && apt-get install -y --no-install-recommends \
#       ffmpeg libsndfile1 build-essential && rm -rf /var/lib/apt/lists/*
#   RUN pip install --no-cache-dir torch torchaudio \
#       --index-url https://download.pytorch.org/whl/cpu
#   RUN pip install --no-cache-dir nemo_toolkit[asr]

FROM nvcr.io/nvidia/nemo:25.02

WORKDIR /app

# ffmpeg is needed for audio format conversion (WebM → WAV)
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY server.py .

EXPOSE 8001

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8001"]
