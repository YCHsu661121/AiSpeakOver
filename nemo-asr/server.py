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
import traceback  # Add traceback for better error logging
from pathlib import Path

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse

app = FastAPI(title="NeMo ASR Server")

MODEL_NAME: str = os.environ.get("NEMO_MODEL", "stt_zh_conformer_ctc_large")
_model = None
_model_error: str | None = None  # set if loading failed


def get_model():
    global _model, _model_error
    if _model is not None:
        return _model
    if _model_error:
        raise RuntimeError(_model_error)

    import nemo.collections.asr as nemo_asr  # deferred – heavy import

    print(f"[NeMo] Loading model: {MODEL_NAME}", flush=True)
    try:
        _model = nemo_asr.models.ASRModel.from_pretrained(MODEL_NAME)
        _model.eval()
        print("[NeMo] Model ready.", flush=True)
    except Exception as exc:
        _model_error = str(exc)
        print(f"[NeMo] ERROR loading model: {exc}", flush=True)
        raise
    return _model


@app.on_event("startup")
async def on_startup():
    """Warm up model in background — don’t block/crash server if it fails."""
    import asyncio
    import concurrent.futures
    loop = asyncio.get_event_loop()
    def _load():
        try:
            get_model()
        except Exception:
            pass  # error already stored in _model_error; /health will report it
    loop.run_in_executor(concurrent.futures.ThreadPoolExecutor(max_workers=1), _load)


@app.get("/health")
async def health():
    status = "ready" if _model is not None else ("error" if _model_error else "loading")
    return {"status": status, "model": MODEL_NAME, "error": _model_error}


@app.post("/v1/audio/transcriptions")
async def transcribe(
    file: UploadFile = File(...),
    model: str = Form(None),
    language: str = Form("zh"),
    response_format: str = Form("json"),
):
    # Fail fast with a clear message if model is still loading or failed
    if _model is None:
        if _model_error:
            return JSONResponse(
                {"error": f"NeMo model failed to load: {_model_error}"},
                status_code=503,
            )
        return JSONResponse(
            {"error": f"NeMo model '{MODEL_NAME}' is still loading, please retry in a moment"},
            status_code=503,
        )

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
            print(f"[FFmpeg Error] {err}", flush=True)
            return JSONResponse({"error": f"ffmpeg: {err}"}, status_code=500)

        try:
            m = get_model()
            results = m.transcribe([str(out_path)])
            # CTC models return List[str]; RNNT/Transducer models return
            # List[Hypothesis] or a tuple (hypotheses, _) depending on NeMo version
            raw = results
            if isinstance(raw, tuple):          # (hypotheses, extra)
                raw = raw[0]
            item = raw[0] if raw else ""
            if hasattr(item, "text"):           # Hypothesis object
                text = item.text
            elif hasattr(item, "y_sequence"):   # older RNNT Hypothesis
                text = str(item)
            else:
                text = str(item)
            return {"text": text.strip() if isinstance(text, str) else str(text).strip()}
        except Exception as exc:
            traceback.print_exc()  # Print full traceback to logs
            return JSONResponse({"error": str(exc)}, status_code=500)
