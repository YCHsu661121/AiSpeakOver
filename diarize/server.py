"""
Speaker diarization microservice.
Uses resemblyzer (GE2E) for 256-d speaker embeddings and
spectralcluster for periodic offline re-clustering.

Endpoints (OpenAI-inspired):
  POST /assign  — return speaker_id (0 or 1) for an audio chunk
  POST /reset   — clear session state
  GET  /health  — readiness probe
"""

import os
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse

app = FastAPI(title="Speaker Diarizer")

SIM_THRESHOLD = float(os.environ.get("SIM_THRESHOLD", "0.80"))
MAX_SPEAKERS  = int(os.environ.get("MAX_SPEAKERS",   "2"))
RECLUSTER_N   = int(os.environ.get("RECLUSTER_N",   "10"))  # spectral re-cluster every N segments

# ── Model (lazy load at startup) ──────────────────────────────────────────

_encoder = None


def _get_encoder():
    global _encoder
    if _encoder is None:
        from resemblyzer import VoiceEncoder
        print("[Diarize] Loading VoiceEncoder…", flush=True)
        _encoder = VoiceEncoder()
        print("[Diarize] Model ready.", flush=True)
    return _encoder


@app.on_event("startup")
async def startup():
    _get_encoder()


# ── Speaker state ─────────────────────────────────────────────────────────

class SpeakerTracker:
    """Online 2-speaker tracker with spectral re-clustering refinement."""

    def __init__(self):
        self.embeddings: list[np.ndarray] = []   # all embeddings seen
        self.labels:     list[int]        = []   # assigned speaker per embedding
        self.centroids:  list[np.ndarray] = []   # running centroid per speaker

    def reset(self):
        self.embeddings.clear()
        self.labels.clear()
        self.centroids.clear()

    def assign(self, emb: np.ndarray) -> int:
        """Assign embedding to a speaker (0-based). Updates state in-place."""
        # First segment → speaker 0
        if not self.centroids:
            self.centroids.append(emb.copy())
            self.embeddings.append(emb)
            self.labels.append(0)
            return 0

        sims = [_cosine(emb, c) for c in self.centroids]
        best = int(np.argmax(sims))

        if sims[best] >= SIM_THRESHOLD:
            speaker = best
        elif len(self.centroids) < MAX_SPEAKERS:
            # New speaker discovered
            self.centroids.append(emb.copy())
            speaker = len(self.centroids) - 1
        else:
            # Already at max speakers — assign to closest
            speaker = best

        # EMA centroid update (α=0.05 keeps history but adapts gradually)
        alpha = 0.05
        self.centroids[speaker] = (1 - alpha) * self.centroids[speaker] + alpha * emb

        self.embeddings.append(emb)
        self.labels.append(speaker)

        # Periodic spectral re-clustering once both speakers found
        n = len(self.embeddings)
        if len(self.centroids) == MAX_SPEAKERS and n > 0 and n % RECLUSTER_N == 0:
            self._recluster()

        return speaker

    def _recluster(self):
        """SpectralClusterer re-clusters all accumulated embeddings."""
        try:
            from spectralcluster import SpectralClusterer
            arr = np.stack(self.embeddings)
            n_spk = len(self.centroids)
            sc = SpectralClusterer(min_clusters=n_spk, max_clusters=n_spk)
            new_labels = sc.predict(arr)

            # Align: new cluster whose majority overlaps old speaker-0 stays 0
            old = np.array(self.labels)
            overlap = np.sum((old == 0) & (new_labels == 0))
            if overlap < np.sum(old == 0) * 0.5:
                new_labels = 1 - new_labels  # swap 0/1

            # Recompute centroids from re-clustered labels
            for spk in range(n_spk):
                mask = new_labels == spk
                if mask.sum() > 0:
                    self.centroids[spk] = np.mean(arr[mask], axis=0)

            self.labels = list(map(int, new_labels))
        except Exception as exc:
            print(f"[Diarize] Spectral re-cluster failed (non-fatal): {exc}", flush=True)


_tracker = SpeakerTracker()


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    d = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / d) if d > 1e-9 else 0.0


def _to_wav16k(audio_bytes: bytes, suffix: str) -> Optional[Path]:
    tmpdir   = tempfile.mkdtemp()
    in_path  = Path(tmpdir) / f"in{suffix}"
    out_path = Path(tmpdir) / "out.wav"
    in_path.write_bytes(audio_bytes)
    r = subprocess.run(
        ["ffmpeg", "-y", "-i", str(in_path),
         "-ar", "16000", "-ac", "1", "-f", "wav", str(out_path)],
        capture_output=True,
    )
    return out_path if r.returncode == 0 else None


# ── Routes ────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status": "ok",
        "n_speakers": len(_tracker.centroids),
        "n_segments": len(_tracker.embeddings),
    }


@app.post("/assign")
async def assign(audio: UploadFile = File(...)):
    """Return speaker_id (0 or 1) for the submitted audio chunk."""
    mime   = audio.content_type or "audio/webm"
    suffix = ".webm" if "webm" in mime else (".ogg" if "ogg" in mime else ".wav")
    data   = await audio.read()

    wav_path = _to_wav16k(data, suffix)
    if not wav_path:
        return JSONResponse({"error": "ffmpeg conversion failed"}, status_code=500)

    try:
        from resemblyzer import preprocess_wav
        wav = preprocess_wav(str(wav_path))
        if wav is None or len(wav) < 3200:   # < 0.2 s at 16 kHz → too short
            return JSONResponse({"error": "audio too short for diarization"}, status_code=400)

        enc = _get_encoder()
        emb = enc.embed_utterance(wav)
        spk = _tracker.assign(emb)
        return {"speaker_id": spk, "n_speakers": len(_tracker.centroids)}
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)


@app.post("/reset")
def reset():
    _tracker.reset()
    return {"status": "reset", "n_speakers": 0}
