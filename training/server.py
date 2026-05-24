"""FastAPI server wrapping `triage_core.TriageEngine` (Milestone 22).

Single endpoint surface so the browser's local-mode pathway can POST a CXR and get
the SAME calibrated TriageResult the CLI emits. Engine is loaded ONCE at startup
(FastAPI lifespan event) — every request gets a warm Rad-DINO + TXRV + heads + seg
on MPS, sub-second on M4.

CORS: only http://localhost:5173 and http://127.0.0.1:5173 (Vite dev defaults). NOT
permissive — this server holds your trained head and runs on your laptop; do not
listen for cross-origin requests from arbitrary sites. If you change the Vite port,
update `ALLOWED_ORIGINS` here.

    PYTORCH_ENABLE_MPS_FALLBACK=1 training/.venv/bin/python -m uvicorn training.server:app --port 8000

Endpoints:
  - POST /triage  — multipart `file=<image>` OR JSON `{image_b64: "<base64>"}` -> TriageResult
  - GET  /health  — engine readiness + audit pins (model_sha, git_sha, calibration)
"""
from __future__ import annotations

import base64
import binascii
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "training"))

# Default offline (matches the CLI). Override the env vars at the shell to bust the cache.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from triage_core import TriageEngine, get_engine  # noqa: E402


ALLOWED_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Warm the engine on startup so the FIRST /triage request is sub-second.
    If the engine fails to load (HF cache missing, missing weights), FastAPI starts
    anyway and /triage returns a 503 with a human-readable error — the browser shows
    `engine: unreachable` and asks the user to run the load command on the shell."""
    try:
        app.state.engine = get_engine()
        app.state.engine_error = None
    except Exception as e:
        app.state.engine = None
        app.state.engine_error = repr(e)[:400]
    yield


app = FastAPI(title="TB triage — local mode (M22)", version="1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type"],
)


def _engine_or_503(request: Request) -> TriageEngine:
    eng = getattr(request.app.state, "engine", None)
    if eng is None:
        err = getattr(request.app.state, "engine_error", "engine not initialized")
        raise HTTPException(status_code=503, detail={
            "error": "engine_unavailable",
            "reason": err,
            "hint": "Re-run with HF_HUB_OFFLINE=0 once to warm the Rad-DINO cache, then restart the server.",
        })
    return eng


@app.get("/health")
def health(request: Request) -> JSONResponse:
    """Liveness probe + audit pins. The browser SettingsDrawer pings this on toggle-on."""
    eng: TriageEngine | None = getattr(request.app.state, "engine", None)
    if eng is None:
        err = getattr(request.app.state, "engine_error", "engine not initialized")
        return JSONResponse(
            status_code=503,
            content={
                "ok": False,
                "engine": "unreachable",
                "reason": err,
                "hint": "Run `training/.venv/bin/python -m uvicorn training.server:app --port 8000` and check the shell for the stack trace.",
            },
        )
    return JSONResponse(
        content={
            "ok": True,
            "engine": "ready",
            "model_sha": eng.model_sha,
            "git_sha": eng.git_sha,
            "calibration": {
                "T": eng.T,
                "thr_at_95sens": eng.thr_at_95sens,
                "T_sequelae": eng.T_sequelae,
                "version": eng.calibration_version,
            },
        }
    )


@app.post("/triage")
async def triage(request: Request, file: UploadFile | None = File(default=None)) -> JSONResponse:
    """POST a CXR image. Two accepted shapes:

    1. multipart/form-data with field `file=<image>` (the default; matches FormData
       from the browser fetch). Content-Type must be image/* or octet-stream.
    2. application/json with body `{ "image_b64": "<base64 string>" }` (useful for
       non-browser callers and the test suite). The b64 may or may not be a data:url
       prefix; we strip `data:*;base64,` if present.

    Returns the same TriageResult dict the CLI prints under --json.
    """
    eng = _engine_or_503(request)

    image_bytes: bytes | None = None
    if file is not None and file.filename:
        image_bytes = await file.read()
    else:
        # Accept JSON body. Content-Type is checked permissively so a curl with
        # --data-binary still works for quick tests.
        try:
            payload = await request.json()
        except Exception:
            payload = None
        if isinstance(payload, dict) and isinstance(payload.get("image_b64"), str):
            b64 = payload["image_b64"]
            # strip data URL prefix if present
            if "," in b64 and b64.lower().startswith("data:"):
                b64 = b64.split(",", 1)[1]
            try:
                image_bytes = base64.b64decode(b64, validate=True)
            except (binascii.Error, ValueError) as e:
                raise HTTPException(status_code=400, detail={
                    "error": "bad_base64",
                    "reason": str(e),
                }) from e

    if not image_bytes:
        raise HTTPException(status_code=400, detail={
            "error": "no_image",
            "reason": "POST a multipart `file=<image>` or JSON `{image_b64: ...}`.",
        })

    try:
        result = eng.run(image_bytes)
    except ValueError as e:
        # malformed image bytes — surface as 400, not 500
        raise HTTPException(status_code=400, detail={"error": "decode_failed", "reason": str(e)}) from e
    except Exception as e:
        # Real engine failure — surface with the message; never swallow.
        raise HTTPException(status_code=500, detail={"error": "engine_run_failed", "reason": repr(e)[:400]}) from e

    return JSONResponse(content=result.to_dict())


# Module-level handle so `uvicorn training.server:app` discovers the app object.
__all__ = ["app"]
