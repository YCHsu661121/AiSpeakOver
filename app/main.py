import asyncio
import json
import logging
import logging.handlers
import os
from pathlib import Path

import httpx
import yaml
from fastapi import FastAPI, Form, Request, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

# ── Logging ─────────────────────────────────────────────────────────────────

LOG_DIR = Path("/app/logs")
LOG_DIR.mkdir(exist_ok=True)

_log_handler = logging.handlers.RotatingFileHandler(
    LOG_DIR / "app.log", maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
)
_log_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler(), _log_handler])
logger = logging.getLogger("aispeakover")

# ── Config ──────────────────────────────────────────────────────────────────

def load_config() -> dict:
    for p in [Path("/app/config.yaml"), Path(__file__).parent.parent / "config.yaml", Path("config.yaml")]:
        if p.exists():
            with open(p, encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
    return {}

_cfg = load_config()

OLLAMA_BASE_URL: str = os.environ.get("OLLAMA_BASE_URL") or _cfg.get("ollama_base_url", "http://ai:11434")
DEFAULT_MODEL: str   = _cfg.get("default_model", "qwen2.5:7b")
DEFAULT_SRC:   str   = _cfg.get("default_source_lang", "zh-TW")
DEFAULT_TGT:   str   = _cfg.get("default_target_lang", "en")
WHISPER_BASE_URL: str = os.environ.get("WHISPER_BASE_URL") or _cfg.get("whisper_base_url", "http://whisper:8000")
WHISPER_MODEL:    str = _cfg.get("whisper_model", "Systran/faster-whisper-small")
NEMO_BASE_URL:    str = os.environ.get("NEMO_BASE_URL") or _cfg.get("nemo_base_url", "http://nemo-asr:8001")
NEMO_MODEL:       str = _cfg.get("nemo_model", "stt_zh_conformer_ctc_large")
DEFAULT_STT:      str = os.environ.get("DEFAULT_STT") or _cfg.get("default_stt", "whisper")
DIARIZE_BASE_URL: str = os.environ.get("DIARIZE_BASE_URL") or _cfg.get("diarize_base_url", "http://diarize:8002")

# ── App ──────────────────────────────────────────────────────────────────────

app = FastAPI(title="AiSpeakOver")

@app.get("/api/log")
async def api_log(lines: int = 200):
    """Return the last N lines of app.log as plain text."""
    log_file = LOG_DIR / "app.log"
    if not log_file.exists():
        return JSONResponse({"error": "log file not found"}, status_code=404)
    with open(log_file, encoding="utf-8", errors="replace") as f:
        all_lines = f.readlines()
    return StreamingResponse(
        iter(all_lines[-lines:]),
        media_type="text/plain; charset=utf-8",
    )

@app.post("/api/log")
async def api_log_write(request: Request):
    """Receive a log entry from the frontend and append to app.log."""
    try:
        body = await request.json()
        level = str(body.get("level", "info")).upper()
        msg   = str(body.get("msg", ""))
        logger.log(getattr(logging, level, logging.INFO), "[UI] %s", msg)
    except Exception:
        pass
    return {"ok": True}

# ── API routes (must come before static mount) ───────────────────────────────

@app.get("/api/config")
async def api_config():
    return {
        "ollama_base_url":     OLLAMA_BASE_URL,
        "default_model":       DEFAULT_MODEL,
        "default_source_lang": DEFAULT_SRC,
        "default_target_lang": DEFAULT_TGT,
        "whisper_model":       WHISPER_MODEL,
        "nemo_model":          NEMO_MODEL,
        "default_stt":         DEFAULT_STT,
        "diarize_base_url":    DIARIZE_BASE_URL,
    }


@app.post("/api/transcribe")
async def api_transcribe(
    audio: UploadFile,
    language: str = Form(""),           # empty string = auto-detect
    model: str | None = Form(None),
    backend: str = Form("whisper"),     # "whisper" | "nemo"
    diarize_req: str = Form("false", alias="diarize"),
):
    """Proxy audio to STT backend; optionally run speaker diarization in parallel."""
    if backend == "nemo":
        base_url  = NEMO_BASE_URL
        stt_model = model or NEMO_MODEL
        timeout   = httpx.Timeout(connect=5, read=60, write=10, pool=5)
    else:
        base_url  = WHISPER_BASE_URL
        stt_model = model or WHISPER_MODEL
        timeout   = httpx.Timeout(connect=5, read=30, write=10, pool=5)

    audio_bytes  = await audio.read()
    filename     = audio.filename or "audio.webm"
    content_type = audio.content_type or "audio/webm"

    # Normalize content-type: strip codec suffix (e.g. "audio/webm;codecs=opus" → "audio/webm")
    # and map to a clean MIME + matching filename that faster-whisper-server can decode.
    base_mime = content_type.split(";")[0].strip().lower()
    _MIME_TO_EXT = {
        "audio/webm":  ".webm",
        "audio/ogg":   ".ogg",
        "audio/mp4":   ".mp4",
        "audio/mpeg":  ".mp3",
        "audio/wav":   ".wav",
        "audio/flac":  ".flac",
    }
    ext = _MIME_TO_EXT.get(base_mime, ".webm")
    filename     = "audio" + ext
    content_type = base_mime

    logger.info("transcribe  backend=%s  size=%dB  filename=%s", backend, len(audio_bytes), filename)
    async def stt_call() -> str:
        async with httpx.AsyncClient(timeout=timeout) as client:
            stt_data: dict = {"model": stt_model, "response_format": "json"}
            if language:
                stt_data["language"] = language
            resp = await client.post(
                f"{base_url}/v1/audio/transcriptions",
                files={"file": (filename, audio_bytes)},
                data=stt_data,
            )
            if resp.status_code >= 400:
                body = resp.text
                # ffmpeg errors put the actual message at the END — log tail
                snippet = body[-1000:] if len(body) > 1000 else body
                logger.error("STT %s %d (body_len=%d): ...%s", base_url, resp.status_code, len(body), snippet)
            resp.raise_for_status()
            body = resp.json()
            return (body.get("text") or body.get("transcription") or body.get("transcript") or "").strip()

    async def diarize_call() -> int:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.post(
                f"{DIARIZE_BASE_URL}/assign",
                files={"audio": (filename, audio_bytes)},
            )
            resp.raise_for_status()
            return int(resp.json().get("speaker_id", 0))

    async def stt_call_with_fallback() -> tuple[str, bool]:
        """Try primary backend; if NeMo fails, fall back to Whisper. Returns (text, fell_back)."""
        try:
            return await stt_call(), False
        except Exception as exc:
            if backend == "nemo":
                # NeMo unavailable — retry with Whisper
                import logging
                logging.getLogger(__name__).warning("NeMo failed (%s), falling back to Whisper", exc)
                fallback_url   = WHISPER_BASE_URL
                fallback_model = WHISPER_MODEL
                fallback_timeout = httpx.Timeout(connect=5, read=30, write=10, pool=5)
                async with httpx.AsyncClient(timeout=fallback_timeout) as client:
                    stt_data: dict = {"model": fallback_model, "response_format": "json"}
                    if language:
                        stt_data["language"] = language
                    resp = await client.post(
                        f"{fallback_url}/v1/audio/transcriptions",
                        files={"file": (filename, audio_bytes)},
                        data=stt_data,
                    )
                    if resp.status_code >= 400:
                        body = resp.text
                        snippet = body[-1000:] if len(body) > 1000 else body
                        logger.error("Whisper fallback %d (body_len=%d): ...%s", resp.status_code, len(body), snippet)
                    resp.raise_for_status()
                    return resp.json().get("text", "").strip(), True
            raise

    try:
        if diarize_req.lower() == "true":
            stt_task = asyncio.create_task(stt_call_with_fallback())
            diarize_task = asyncio.create_task(diarize_call())
            results = await asyncio.gather(stt_task, diarize_task, return_exceptions=True)
            if isinstance(results[0], Exception):
                return JSONResponse({"error": str(results[0])}, status_code=500)
            (text, fell_back) = results[0]
            speaker_id = results[1] if not isinstance(results[1], Exception) else None
            resp_body = {"text": text, "speaker_id": speaker_id}
            if fell_back:
                resp_body["stt_fallback"] = "whisper"
            return resp_body
        else:
            text, fell_back = await stt_call_with_fallback()
            resp_body = {"text": text}
            if fell_back:
                resp_body["stt_fallback"] = "whisper"
            return resp_body
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)


@app.post("/api/diarize/reset")
async def api_diarize_reset():
    """Forward reset request to the diarize service."""
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            resp = await client.post(f"{DIARIZE_BASE_URL}/reset")
            return resp.json()
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)


@app.get("/api/health")
async def api_health():
    """Overall health: ollama + whisper + nemo (3 s timeout each)."""
    results: dict = {}
    async with httpx.AsyncClient(timeout=3) as client:
        # Ollama
        try:
            r = await client.get(f"{OLLAMA_BASE_URL}/api/tags")
            models = [m["name"] for m in r.json().get("models", [])]
            results["ollama"] = True
            results["ollama_models"] = models
        except Exception as exc:
            results["ollama"] = False
            results["ollama_error"] = str(exc)
        # STT
        for name, url in [("whisper", WHISPER_BASE_URL), ("nemo", NEMO_BASE_URL)]:
            try:
                r = await client.get(f"{url}/health")
                body = r.json()
                if name == "nemo":
                    results["nemo"] = body.get("status") == "ready"
                    results["nemo_status"] = body.get("status", "unknown")
                    if body.get("error"):
                        results["nemo_error"] = body["error"]
                else:
                    results[name] = r.status_code < 500
            except Exception as exc:
                results[name] = False
                if name == "nemo":
                    results["nemo_status"] = f"unreachable"
    return results


@app.get("/api/stt/health")
async def api_stt_health():
    """Quick reachability check for each STT backend (3 s connect timeout)."""
    results: dict = {}
    async with httpx.AsyncClient(timeout=3) as client:
        for name, url in [("whisper", WHISPER_BASE_URL), ("nemo", NEMO_BASE_URL)]:
            try:
                r = await client.get(f"{url}/health")
                body = r.json()
                # NeMo reports status: "ready" | "loading" | "error"
                # Whisper reports HTTP 200 = healthy
                if name == "nemo":
                    results[name] = body.get("status") == "ready"
                    results["nemo_status"] = body.get("status", "unknown")
                    if body.get("error"):
                        results["nemo_error"] = body["error"]
                else:
                    results[name] = r.status_code < 500
            except Exception as exc:
                results[name] = False
                if name == "nemo":
                    results["nemo_status"] = f"unreachable: {exc}"
    return results


@app.get("/api/models")
async def api_models():
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(f"{OLLAMA_BASE_URL}/api/tags")
            resp.raise_for_status()
            data = resp.json()
            models = [m["name"] for m in data.get("models", [])]
            return {"models": models}
    except Exception as exc:
        return JSONResponse({"models": [], "error": str(exc)}, status_code=200)


@app.get("/api/library")
async def api_library():
    """Return curated list of small translation-capable models."""
    models = [
        {"name": "qwen2.5:1.5b", "size": "1.1 GB", "note": "小巧快速，支援中╱日╱韓↔英"},
        {"name": "qwen2.5:3b",   "size": "2.0 GB", "note": "品質較佳，多語言"},
        {"name": "qwen2.5:7b",   "size": "4.7 GB", "note": "高品質翻譯（原預設）"},
        {"name": "gemma3:1b",    "size": "0.8 GB", "note": "超小，英文為主"},
        {"name": "llama3.2:3b",  "size": "2.0 GB", "note": "平衡速度與品質"},
    ]
    # Mark the configured default
    for m in models:
        m["default"] = (m["name"] == DEFAULT_MODEL)
    return {"models": models}


@app.post("/api/pull")
async def api_pull(request: Request):
    """Stream model pull progress from Ollama (SSE)."""
    body  = await request.json()
    model = body.get("model", DEFAULT_MODEL)

    async def pull_gen():
        try:
            async with httpx.AsyncClient(timeout=None) as client:
                async with client.stream(
                    "POST",
                    f"{OLLAMA_BASE_URL}/api/pull",
                    json={"name": model, "stream": True},
                ) as resp:
                    async for line in resp.aiter_lines():
                        if not line:
                            continue
                        try:
                            data = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        yield f"data: {json.dumps(data)}\n\n"
            yield f"data: {json.dumps({'status': 'done'})}\n\n"
        except Exception as exc:
            yield f"data: {json.dumps({'error': str(exc)})}\n\n"

    return StreamingResponse(
        pull_gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


LANG_NAMES = {
    "zh-TW": "Traditional Chinese (繁體中文)",
    "zh-CN": "Simplified Chinese (简体中文)",
    "en":    "English",
    "ja":    "Japanese (日本語)",
    "ko":    "Korean (한국어)",
}


@app.post("/api/translate")
async def api_translate(request: Request):
    body       = await request.json()
    text       = body.get("text", "").strip()
    source_lang = body.get("source_lang", DEFAULT_SRC)
    target_lang = body.get("target_lang", DEFAULT_TGT)
    model      = body.get("model", DEFAULT_MODEL)

    if not text:
        return JSONResponse({"error": "empty text"}, status_code=400)

    src_name = LANG_NAMES.get(source_lang, source_lang)
    tgt_name = LANG_NAMES.get(target_lang, target_lang)

    system_prompt = (
        f"You are a professional translator. "
        f"Translate the following {src_name} text into {tgt_name}. "
        f"Output ONLY the translated text — no explanations, no notes, no original."
    )

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": text},
        ],
        "stream": True,
        "options": {"temperature": 0.1},
    }

    async def stream_gen():
        try:
            async with httpx.AsyncClient(timeout=60) as client:
                async with client.stream("POST", f"{OLLAMA_BASE_URL}/api/chat", json=payload) as resp:
                    logger.info("translate Ollama HTTP %d  model=%s", resp.status_code, model)
                    token_count = 0
                    async for line in resp.aiter_lines():
                        if not line:
                            continue
                        try:
                            data = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        content = data.get("message", {}).get("content", "")
                        if content:
                            token_count += 1
                            yield f"data: {json.dumps({'content': content})}\n\n"
                        if data.get("done"):
                            logger.info("translate done  tokens=%d", token_count)
                            yield "data: [DONE]\n\n"
        except Exception as exc:
            logger.error("translate stream error: %s", exc)
            yield f"data: {json.dumps({'error': str(exc)})}\n\n"

    return StreamingResponse(
        stream_gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── Static files (last) ───────────────────────────────────────────────────────

_env_static = os.environ.get("STATIC_DIR")
_STATIC_DIR = Path(_env_static) if _env_static else Path(__file__).parent / "static"
app.mount("/", StaticFiles(directory=str(_STATIC_DIR), html=True), name="static")
