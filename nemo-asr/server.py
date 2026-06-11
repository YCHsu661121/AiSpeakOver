"""
Minimal NeMo ASR server — OpenAI-compatible /v1/audio/transcriptions endpoint.

Receives audio (any format), converts to 16 kHz mono WAV via ffmpeg, then
runs the configured NeMo ASR model for transcription.

Env vars:
  NEMO_MODEL  – pretrained model name (default: stt_zh_conformer_ctc_large)
"""

import os
import subprocess
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse

app = FastAPI(title="NeMo ASR Server")

MODEL_NAME: str = os.environ.get("NEMO_MODEL", "stt_zh_conformer_ctc_large")
_model = None


def get_model():
    global _model
    if _model is not None:
        return _model

    import nemo.collections.asr as nemo_asr  # deferred – heavy import

    print(f"[NeMo] Loading model: {MODEL_NAME}", flush=True)
    _model = nemo_asr.models.ASRModel.from_pretrained(MODEL_NAME)
    _model.eval()
    print("[NeMo] Model ready.", flush=True)
    return _model


@app.on_event("startup")
async def on_startup():
    get_model()


@app.get("/health")
async def health():
    return {"status": "ok", "model": MODEL_NAME}


@app.post("/v1/audio/transcriptions")
async def transcribe(
    file: UploadFile = File(...),
    model: str = Form(None),
    language: str = Form("zh"),
    response_format: str = Form("json"),
):
    audio_bytes = await file.read()
    suffix = Path(file.filename or "audio.webm").suffix or ".webm"

    with tempfile.TemporaryDirectory() as tmpdir:
        in_path  = Path(tmpdir) / f"in{suffix}"
        out_path = Path(tmpdir) / "out.wav"
        in_path.write_bytes(audio_bytes)

        # Convert to 16 kHz mono WAV — NeMo's standard input format
        proc = subprocess.run(
            [
                "ffmpeg", "-y", "-i", str(in_path),
                "-ar", "16000", "-ac", "1", "-f", "wav", str(out_path),
            ],
            capture_output=True,
        )
        if proc.returncode != 0:
            err = proc.stderr.decode(errors="replace")
            return JSONResponse({"error": f"ffmpeg: {err}"}, status_code=500)

        try:
            m = get_model()
            results = m.transcribe([str(out_path)])
            # Result may be a string or a Hypothesis object
            text = results[0] if results else ""
            if hasattr(text, "text"):
                text = text.text
            return {"text": str(text).strip()}
        except Exception as exc:
            return JSONResponse({"error": str(exc)}, status_code=500)
