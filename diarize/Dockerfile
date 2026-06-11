FROM python:3.11-slim

# System libs: ffmpeg for audio conversion, libsndfile for resemblyzer/soundfile
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libsndfile1 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
# Install torch CPU first (smaller wheel) so resemblyzer picks it up
RUN pip install --no-cache-dir torch torchaudio \
    --index-url https://download.pytorch.org/whl/cpu
# webrtcvad-wheels provides a pre-built wheel so no C compiler is needed
# (resemblyzer depends on webrtcvad; this satisfies it without source compilation)
RUN pip install --no-cache-dir webrtcvad-wheels
RUN pip install --no-cache-dir -r requirements.txt

COPY server.py .

EXPOSE 8002

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8002"]
