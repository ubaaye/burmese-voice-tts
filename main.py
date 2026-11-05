import asyncio
import base64
import hmac
import json
import math
import os
import re
import struct
import sys
import tempfile
import time
import uuid

from array import array
from collections import deque
from datetime import datetime, timezone
from typing import Any

import edge_tts
import httpx
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.background import BackgroundTask

try:
    import jwt as _pyjwt
    from jwt.algorithms import ECAlgorithm as _ECAlgorithm
    from jwt.algorithms import OKPAlgorithm as _OKPAlgorithm
    from jwt.algorithms import RSAAlgorithm as _RSAAlgorithm
except Exception:  # PyJWT missing: public routes fail closed (503), never open.
    _pyjwt = None
    _ECAlgorithm = None
    _OKPAlgorithm = None
    _RSAAlgorithm = None

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Frontend static files
if os.path.exists("public"):
    app.mount("/public", StaticFiles(directory="public"), name="public")

HF_SPACE = "https://freococo-f5-myanmar-tts-demo.hf.space"
HF_TOKEN = os.environ.get("HF_TOKEN", "")

# ElevenLabs proxy: the API key lives ONLY in this server's environment.
# It is never sent to the web app or included in any response/log.
EL_API_KEY = os.environ.get("EL_API_KEY", "")
EL_BASE = "https://api.elevenlabs.io"
EL_DEFAULT_VOICE_ID = "RANtSKxEy3vvVXaqppeT"  # "NLO MyVoice v2" clone
EL_DEFAULT_MODEL_ID = "eleven_v4"

# Gemini TTS proxy: the API key lives ONLY in this server's environment.
# It is never sent to the web app or included in any response/log.
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_BASE = "https://generativelanguage.googleapis.com"
GEMINI_TTS_MODEL = os.environ.get("GEMINI_TTS_MODEL", "gemini-3.8-flash-tts")
GEMINI_DEFAULT_VOICE = "Kore"

# --- Public edition (Supabase login + bring-your-own-key) ---
# Every /api/pub/* route requires a valid Supabase access token
# (Authorization: Bearer <jwt>, verified against the project JWKS below).
# The paid/quota routes resolve their provider key per request (phase
# 11, see _resolve_pub_key): a valid Pro code — or the owner's own
# login — runs on the SERVER's env key for the engine, while a visitor
# key header (X-EL-Key / X-Gemini-Key / X-HF-Token) runs on the
# visitor's own key with no code involved. A deployment that leaves the
# env keys UNSET simply cannot serve the code/owner paths (they 503);
# bring-your-own-key calls keep working. Visitor keys are used for that
# request/job only and are never stored or logged.
SUPABASE_URL = os.environ.get("SUPABASE_URL", "").rstrip("/")
SUPABASE_JWKS_URL = (
    f"{SUPABASE_URL}/auth/v1/.well-known/jwks.json" if SUPABASE_URL else ""
)
SUPABASE_ISSUER = f"{SUPABASE_URL}/auth/v1" if SUPABASE_URL else ""

# Service-role key for the pro_codes table (server-side only, like the
# other provider keys: never sent to clients, never logged). Pro routes
# and the admin endpoints fail closed (503) when it is unset.
SUPABASE_SERVICE_KEY = os.environ.get("SUPABASE_SERVICE_KEY", "")
# Shared secret the owner's private app sends as X-Owner-Key on the
# private paid routes (/api/el/*, /api/gemini/*, /api/clone*) and on the
# /api/admin/* endpoints. When unset those routes are locked (503).
OWNER_KEY = os.environ.get("OWNER_KEY", "")

# The owner's Supabase account email. As an alternative to carrying the
# owner key, the /api/admin/* endpoints also accept the owner's
# Supabase login: an "Authorization: Bearer <jwt>" that verifies
# against the project JWKS and whose verified email claim equals this
# address (compared case-insensitively). Unset disables that path, as
# does an unset SUPABASE_URL. The private paid routes are unaffected:
# they keep the owner key only.
OWNER_EMAIL = os.environ.get("OWNER_EMAIL", "").strip()

MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB cap on audio uploads
MAX_TEXT_CHARS = 5000  # blunt backstop on EL/Gemini text (app limits are lower)
PUB_SYNTH_MAX_CHARS = 600  # per-request cap on the free public Edge route
PUB_SYNTH_LIMIT = 120  # requests per window, per user and (secondary) per IP
PUB_SYNTH_WINDOW = 3600  # 1 hour
# Phase 11: a public Pro call may run on the SERVER's key (valid Pro code,
# or the owner). ElevenLabs chars on that path are capped per user per
# UTC day so one code cannot drain the owner's balance; callers using
# their OWN key are never capped.
SERVER_EL_DAILY_CHAR_CAP = 10000
JOB_TTL_SECONDS = 1800  # finished clone jobs (and their audio) expire after 30 min


class TTSRequest(BaseModel):
    text: str
    voice: str = "my-MM-NilarNeural"
    rate: str = "+0%"
    pitch: str = "+0Hz"


@app.get("/")
async def root():
    return FileResponse("public/index.html")


@app.get("/api/health")
async def health():
    return {"ok": True, "service": "burmese-tts", "clone": bool(HF_TOKEN), "el": bool(EL_API_KEY), "gemini": bool(GEMINI_API_KEY), "pub": bool(SUPABASE_URL), "v": 3}


def _silent_remove(path: str):
    try:
        os.remove(path)
    except OSError:
        pass


def _mp3_file_response(path: str) -> FileResponse:
    """FileResponse that deletes the temp MP3 once the response is sent."""
    return FileResponse(
        path, media_type="audio/mpeg",
        background=BackgroundTask(_silent_remove, path),
    )


async def _edge_synth_file(text: str, voice: str, rate: str, pitch: str) -> str:
    """Run Edge TTS into a fresh temp MP3 and return its path.

    On any failure the temp file is removed before the error propagates.
    """
    output_path = tempfile.mktemp(suffix=".mp3")
    try:
        communicate = edge_tts.Communicate(
            text=text, voice=voice, rate=rate, pitch=pitch,
        )
        await communicate.save(output_path)
        return output_path
    except Exception:
        _silent_remove(output_path)
        raise


async def _read_upload_capped(upload: UploadFile) -> bytes:
    """Read an uploaded file, refusing anything over MAX_UPLOAD_BYTES."""
    data = await upload.read()
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail="File is too large: uploads are limited to 10 MB. "
                   "(ဖိုင် အရွယ်အစား 10 MB ထက် မကျော်ရပါ)",
        )
    return data


@app.post("/api/synthesize")
async def synthesize_speech(req: TTSRequest):
    if not req.text or not req.text.strip():
        raise HTTPException(status_code=400, detail="Text is required")

    try:
        output_path = await _edge_synth_file(
            req.text.strip(), req.voice, req.rate, req.pitch,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return _mp3_file_response(output_path)


async def _clone_attempt(text: str, ref_text: str, speed: float,
                         audio_bytes: bytes, filename: str, mime: str,
                         token: str) -> bytes:
    """One clone attempt against the Space; returns WAV bytes."""
    headers = {"Authorization": f"Bearer {token}"}

    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(1500.0)) as client:
            # 1. Upload the reference audio to the Space
            up = await client.post(
                HF_SPACE + "/gradio_api/upload",
                files={"files": (filename, audio_bytes, mime)},
                headers=headers,
            )
            up.raise_for_status()
            up_path = up.json()[0]

            # 2. Call the named endpoint via the v2 route (full FileData,
            #    as gradio_client sends). Unlike queue/join, this route
            #    surfaces the Space's real error messages.
            file_data = {
                "path": up_path,
                "orig_name": filename,
                "mime_type": mime,
                "size": len(audio_bytes),
                "is_stream": False,
                "meta": {"_type": "gradio.FileData"},
            }
            call = await client.post(
                HF_SPACE + "/gradio_api/call/v2/generate_speech",
                json={
                    "text": text,
                    "ref_audio": file_data,
                    "ref_text": ref_text,
                    "speed": speed,
                },
                headers=headers,
            )
            call.raise_for_status()
            event_id = call.json().get("event_id")
            if not event_id:
                raise HTTPException(
                    status_code=502,
                    detail=f"Clone service error: no event_id in call response: {call.text[:200]}",
                )

            # 3. Stream events until completion or error
            out_path = None
            fail_info = ""
            current_event = ""
            async with client.stream(
                "GET",
                HF_SPACE + f"/gradio_api/call/generate_speech/{event_id}",
                headers=headers,
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if line.startswith("event:"):
                        current_event = line[6:].strip()
                        continue
                    if not line.startswith("data:"):
                        continue
                    raw = line[5:].strip()
                    try:
                        payload = json.loads(raw)
                    except Exception:
                        continue
                    print("CLONE EVT:", current_event, raw[:500], flush=True)
                    if current_event == "error" or (
                        isinstance(payload, dict) and payload.get("error")
                    ):
                        fail_info = str(
                            payload.get("error") if isinstance(payload, dict) else payload
                        )[:400]
                        break
                    if isinstance(payload, list) and payload:
                        first = payload[0]
                        if isinstance(first, str):
                            out_path = first
                        elif isinstance(first, dict):
                            out_path = first.get("path") or first.get("url")
                        break

            if not out_path:
                raise HTTPException(
                    status_code=502,
                    detail=f"Clone generation failed on the model server: {fail_info}",
                )

            # 4. Download the generated audio
            dl_url = out_path if out_path.startswith("http") else HF_SPACE + "/gradio_api/file=" + out_path
            got = await client.get(dl_url, headers=headers)
            got.raise_for_status()
            return got.content

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Clone service error: {e}")


async def run_clone(text: str, ref_text: str, speed: float,
                    audio_bytes: bytes, filename: str, mime: str,
                    token: str) -> bytes:
    """Zero-shot Burmese voice cloning via the F5-Myanmar-TTS Space.

    The Space's GPU backend fails intermittently with an empty error,
    so up to 4 attempts are made before giving up. `token` is the HF
    token to use for this call only (the server's HF_TOKEN for private
    routes, a visitor's own token for public routes); it is never the
    fallback for the other kind of route.
    """
    if not token:
        raise HTTPException(status_code=500, detail="Clone is not configured (HF_TOKEN missing)")
    last_err = None
    for attempt in range(1, 5):
        try:
            return await _clone_attempt(text, ref_text, speed, audio_bytes, filename, mime, token)
        except HTTPException as e:
            last_err = e
            print(f"CLONE attempt {attempt} failed: {e.detail}", flush=True)
            if attempt < 4:
                await asyncio.sleep(15)
    raise last_err


def _validate_clone(text: str, ref_text: str):
    if not HF_TOKEN:
        raise HTTPException(status_code=500, detail="Clone is not configured (HF_TOKEN missing)")
    if not text.strip():
        raise HTTPException(status_code=400, detail="Text is required")
    if not ref_text.strip():
        raise HTTPException(status_code=400, detail="Reference transcript is required")


@app.post("/api/clone")
async def clone_speech(
    request: Request,
    text: str = Form(...),
    ref_text: str = Form(...),
    speed: float = Form(1.0),
    ref_audio: UploadFile = File(...),
):
    """Synchronous clone (kept for compatibility)."""
    _require_owner(request)
    _validate_clone(text, ref_text)
    audio_bytes = await _read_upload_capped(ref_audio)
    content = await run_clone(
        text.strip(), ref_text.strip(), speed,
        audio_bytes, ref_audio.filename or "ref.wav",
        ref_audio.content_type or "audio/wav",
        HF_TOKEN,
    )
    return Response(content=content, media_type="audio/wav")


# --- Async clone jobs: create, poll status, fetch audio ---
jobs = {}


async def _job_work(job_id: str, text: str, ref_text: str, speed: float,
                    audio_bytes: bytes, filename: str, mime: str,
                    token: str):
    try:
        jobs[job_id]["status"] = "running"
        content = await run_clone(
            text, ref_text, speed, audio_bytes, filename, mime, token,
        )
        jobs[job_id]["status"] = "done"
        jobs[job_id]["audio"] = content
    except HTTPException as e:
        jobs[job_id]["status"] = "error"
        jobs[job_id]["error"] = str(e.detail)
    except Exception as e:
        jobs[job_id]["status"] = "error"
        jobs[job_id]["error"] = f"Clone service error: {e}"
    finally:
        # Stamp the finish time so the entry (and its in-memory audio)
        # can expire via _sweep_jobs. The HF token is NOT kept in the
        # job dict — it exists only in this task's frame while running.
        job = jobs.get(job_id)
        if job is not None:
            job["finished_at"] = time.time()


def _sweep_jobs():
    """Drop finished clone jobs older than JOB_TTL_SECONDS (with audio)."""
    now = time.time()
    stale = [
        jid for jid, job in jobs.items()
        if job.get("finished_at") and now - job["finished_at"] > JOB_TTL_SECONDS
    ]
    for jid in stale:
        jobs.pop(jid, None)


def _job_status_payload(job_id: str, job: dict) -> dict:
    out = {"job_id": job_id, "status": job["status"]}
    if job["status"] == "error":
        out["error"] = job.get("error", "")
    return out


@app.post("/api/clone/jobs")
async def clone_job_create(
    request: Request,
    text: str = Form(...),
    ref_text: str = Form(...),
    speed: float = Form(1.0),
    ref_audio: UploadFile = File(...),
):
    _require_owner(request)
    _validate_clone(text, ref_text)
    audio_bytes = await _read_upload_capped(ref_audio)
    _sweep_jobs()
    job_id = uuid.uuid4().hex[:12]
    jobs[job_id] = {"status": "queued"}
    asyncio.create_task(_job_work(
        job_id, text.strip(), ref_text.strip(), speed,
        audio_bytes, ref_audio.filename or "ref.wav",
        ref_audio.content_type or "audio/wav",
        HF_TOKEN,
    ))
    return {"job_id": job_id, "status": "queued"}


@app.get("/api/clone/jobs/{job_id}")
async def clone_job_status(job_id: str, request: Request):
    _require_owner(request)
    _sweep_jobs()
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return _job_status_payload(job_id, job)


@app.get("/api/clone/jobs/{job_id}/audio")
async def clone_job_audio(job_id: str, request: Request):
    _require_owner(request)
    _sweep_jobs()
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job["status"] != "done":
        raise HTTPException(status_code=409, detail=f"Job not ready: {job['status']}")
    return Response(content=job["audio"], media_type="audio/wav")


# --- ElevenLabs proxy (key stays server-side in EL_API_KEY) ---

class ELTTSRequest(BaseModel):
    text: str
    voice_id: str = EL_DEFAULT_VOICE_ID
    model_id: str = EL_DEFAULT_MODEL_ID
    speed: float | None = None


def _el_error_detail(resp: httpx.Response) -> str:
    """Extract a short human-readable message from an ElevenLabs error body."""
    try:
        data = resp.json()
    except Exception:
        return resp.text[:200] or "unknown error"
    detail = data.get("detail", data) if isinstance(data, dict) else data
    if isinstance(detail, dict):
        return str(detail.get("message") or detail.get("status") or detail)[:300]
    return str(detail)[:300]


@app.get("/api/el/status")
async def el_status(request: Request):
    """Report whether ElevenLabs is configured, plus usage when available."""
    _require_owner(request)
    if not EL_API_KEY:
        return {"configured": False}
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(30.0)) as client:
            resp = await client.get(
                EL_BASE + "/v1/user/subscription",
                headers={"xi-api-key": EL_API_KEY},
            )
        if resp.status_code != 200:
            # Key exists but usage could not be read; still configured.
            return {"configured": True}
        data = resp.json()
        return {
            "configured": True,
            "used": data.get("character_count"),
            "limit": data.get("character_limit"),
            "resets": data.get("next_character_count_reset_unix"),
        }
    except Exception:
        return {"configured": True}


async def _el_tts_core(req: ELTTSRequest, api_key: str):
    """ElevenLabs TTS using the given key (server key, or a visitor's
    own key on the public route). The key is used for this call only."""
    if not req.text or not req.text.strip():
        raise HTTPException(status_code=400, detail="Text is required")
    if len(req.text) > MAX_TEXT_CHARS:
        raise HTTPException(
            status_code=400,
            detail=f"Text is too long: maximum {MAX_TEXT_CHARS} characters per request",
        )

    body: dict = {"text": req.text.strip(), "model_id": req.model_id}
    if req.speed is not None:
        body["voice_settings"] = {"speed": req.speed}

    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(300.0)) as client:
            resp = await client.post(
                f"{EL_BASE}/v1/text-to-speech/{req.voice_id}",
                json=body,
                headers={
                    "xi-api-key": api_key,
                    "Content-Type": "application/json",
                    "Accept": "audio/mpeg",
                },
            )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"ElevenLabs service error: {e}")

    if resp.status_code != 200:
        raise HTTPException(
            status_code=resp.status_code,
            detail=f"ElevenLabs error ({resp.status_code}): {_el_error_detail(resp)}",
        )
    return Response(content=resp.content, media_type="audio/mpeg")


@app.post("/api/el/tts")
async def el_tts(req: ELTTSRequest, request: Request):
    """Generate speech via ElevenLabs using the server-side key."""
    _require_owner(request)
    if not EL_API_KEY:
        raise HTTPException(
            status_code=503,
            detail="ElevenLabs is not configured on the server (EL_API_KEY missing)",
        )
    return await _el_tts_core(req, EL_API_KEY)


async def _el_voices_core(api_key: str, own_only: bool = False):
    """List ElevenLabs voices available to the given key.

    With own_only=True (the public route), only voices owned by the
    key's account are returned: ElevenLabs marks those as category
    "cloned" or "professional". Premade/generated voices (and items
    with no category at all) are excluded — they are not the visitor's
    own clones and cannot be managed or reliably used through the
    public app.
    """
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(30.0)) as client:
            resp = await client.get(
                EL_BASE + "/v1/voices",
                headers={"xi-api-key": api_key},
            )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"ElevenLabs service error: {e}")

    if resp.status_code != 200:
        raise HTTPException(
            status_code=resp.status_code,
            detail=f"ElevenLabs error ({resp.status_code}): {_el_error_detail(resp)}",
        )
    data = resp.json()
    voices = []
    for v in data.get("voices", []):
        if not isinstance(v, dict):
            continue
        category = v.get("category")
        if own_only and category not in ("cloned", "professional"):
            continue
        voices.append(
            {
                "voice_id": v.get("voice_id"),
                "name": v.get("name"),
                "category": category,
            }
        )
    return {"voices": voices}


@app.get("/api/el/voices")
async def el_voices_list(request: Request):
    """List the ElevenLabs voices available to the server-side key."""
    _require_owner(request)
    if not EL_API_KEY:
        raise HTTPException(
            status_code=503,
            detail="ElevenLabs is not configured on the server (EL_API_KEY missing)",
        )
    return await _el_voices_core(EL_API_KEY)


async def _el_voice_add_core(name: str, sample_bytes: bytes, filename: str,
                             mime: str, api_key: str):
    """Create an ElevenLabs cloned (IVC) voice using the given key
    (server key, or a visitor's own key on the public route). The key
    is used for this call only. Upstream errors surface exactly like
    _el_tts_core's."""
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(300.0)) as client:
            resp = await client.post(
                EL_BASE + "/v1/voices/add",
                data={"name": name},
                files={"files": (filename, sample_bytes, mime)},
                headers={"xi-api-key": api_key},
            )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"ElevenLabs service error: {e}")

    if resp.status_code != 200:
        raise HTTPException(
            status_code=resp.status_code,
            detail=f"ElevenLabs error ({resp.status_code}): {_el_error_detail(resp)}",
        )
    data = resp.json()
    return {"voice_id": data.get("voice_id"), "name": name}


async def _el_voice_lookup_core(voice_id: str, api_key: str):
    """Fetch one ElevenLabs voice using the given key. A missing voice
    surfaces as a clean bilingual 404 (not the raw upstream error)."""
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(30.0)) as client:
            resp = await client.get(
                f"{EL_BASE}/v1/voices/{voice_id}",
                headers={"xi-api-key": api_key},
            )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"ElevenLabs service error: {e}")

    if resp.status_code == 404:
        raise HTTPException(
            status_code=404,
            detail="Voice not found. (အသံ မတွေ့ပါ)",
        )
    if resp.status_code != 200:
        raise HTTPException(
            status_code=resp.status_code,
            detail=f"ElevenLabs error ({resp.status_code}): {_el_error_detail(resp)}",
        )
    try:
        data = resp.json()
    except Exception:
        data = {}
    return data if isinstance(data, dict) else {}


async def _el_voice_delete_core(voice_id: str, api_key: str):
    """Delete an ElevenLabs voice using the given key (server key, or
    a visitor's own key on the public route)."""
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(30.0)) as client:
            resp = await client.delete(
                f"{EL_BASE}/v1/voices/{voice_id}",
                headers={"xi-api-key": api_key},
            )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"ElevenLabs service error: {e}")

    if resp.status_code != 200:
        raise HTTPException(
            status_code=resp.status_code,
            detail=f"ElevenLabs error ({resp.status_code}): {_el_error_detail(resp)}",
        )


@app.post("/api/el/voices")
async def el_voices_add(request: Request, name: str = Form(...), sample: UploadFile = File(...)):
    """Create a new ElevenLabs cloned voice from an uploaded audio sample."""
    _require_owner(request)
    if not EL_API_KEY:
        raise HTTPException(
            status_code=503,
            detail="ElevenLabs is not configured on the server (EL_API_KEY missing)",
        )
    if not name or not name.strip():
        raise HTTPException(status_code=400, detail="Voice name is required")

    sample_bytes = await _read_upload_capped(sample)
    if not sample_bytes:
        raise HTTPException(status_code=400, detail="Sample audio file is empty")

    return await _el_voice_add_core(
        name.strip(), sample_bytes,
        sample.filename or "sample.mp3",
        sample.content_type or "audio/mpeg",
        EL_API_KEY,
    )


@app.delete("/api/el/voices/{voice_id}")
async def el_voices_delete(voice_id: str, request: Request):
    """Delete an ElevenLabs voice using the server-side key."""
    _require_owner(request)
    if voice_id == EL_DEFAULT_VOICE_ID:
        # Server-side guard: the default voice (NLO MyVoice v2) must
        # never be deletable, even if a client asks directly.
        raise HTTPException(
            status_code=403,
            detail="The default voice (NLO MyVoice v2) is protected and "
                   "cannot be deleted. (မူလ အသံကို ဖျက်၍ မရပါ)",
        )
    if not EL_API_KEY:
        raise HTTPException(
            status_code=503,
            detail="ElevenLabs is not configured on the server (EL_API_KEY missing)",
        )
    await _el_voice_delete_core(voice_id, EL_API_KEY)
    return {"deleted": True, "voice_id": voice_id}


# --- Gemini TTS proxy (key stays server-side in GEMINI_API_KEY) ---

class GeminiTTSRequest(BaseModel):
    text: str
    voice: str = GEMINI_DEFAULT_VOICE
    # Optional two-speaker dialogue: a list of exactly 2 objects
    # {"speaker": "<name>", "voice": "<voiceName>"}. Typed as Any and
    # validated manually in the handler so ANY malformed speakers payload
    # returns a clear HTTP 400 instead of a pydantic 422.
    speakers: Any = None
    # Optional tone/style instruction (e.g. "angry", "ဒေါသ").
    # When non-empty it is converted by _gemini_style_tag into a
    # bracketed audio tag ("[angry] ") placed at the very start of the
    # text: live-verified 2026-11-05, the model treats a bracketed tag
    # as markup — never spoken — while still changing the delivery,
    # whereas any "Say in ... tone:" instruction prefix (English or
    # Burmese) gets READ ALOUD as if it were content.
    # Also typed as Any and validated manually (400, not 422).
    style: Any = None


def _gemini_error_detail(resp: httpx.Response) -> str:
    """Extract a short human-readable message from a Google API error body."""
    try:
        data = resp.json()
    except Exception:
        return resp.text[:200] or "unknown error"
    if isinstance(data, dict):
        err = data.get("error")
        if isinstance(err, dict) and err.get("message"):
            return str(err["message"])[:300]
    return str(data)[:300]


def _gemini_speaker_configs(speakers) -> list:
    """Validate the optional two-speaker payload and build Gemini's
    speakerVoiceConfigs list. Raises HTTP 400 on any malformed input."""
    if not isinstance(speakers, list) or len(speakers) != 2:
        raise HTTPException(
            status_code=400,
            detail="speakers must be a list of exactly 2 entries: "
                   '[{"speaker": name, "voice": voiceName}, ...]',
        )
    configs = []
    for entry in speakers:
        if not isinstance(entry, dict):
            raise HTTPException(
                status_code=400,
                detail="Each speakers entry must be an object with "
                       "'speaker' and 'voice'",
            )
        name = entry.get("speaker")
        voice = entry.get("voice")
        if not isinstance(name, str) or not name.strip():
            raise HTTPException(
                status_code=400,
                detail="Each speakers entry needs a non-empty 'speaker' name",
            )
        if not isinstance(voice, str) or not voice.strip():
            raise HTTPException(
                status_code=400,
                detail="Each speakers entry needs a non-empty 'voice' "
                       "(Gemini voice name)",
            )
        configs.append({
            "speaker": name.strip(),
            "voiceConfig": {
                "prebuiltVoiceConfig": {"voiceName": voice.strip()},
            },
        })
    return configs


def _gemini_dialogue_parts(text: str, speaker_names: list) -> list:
    """Split dialogue text into one upstream content part per segment.

    A line starting with "<speakerName>:" (for one of the two declared
    speaker names) opens a new segment for that speaker; any other line
    continues the current speaker's segment, and leading unprefixed text
    belongs to the first speaker. The 3.8 TTS model rejects
    multi-speaker requests unless EVERY text part carries
    speech_metadata naming its speaker ("Multi-speaker generation
    requests must specify speech_metadata.speaker for each text part in
    the contents"), so each returned part is
    {"text": <segment text, "Name:" prefix stripped>,
     "speech_metadata": {"speaker": <speakerName>}}.
    Empty segments are dropped.
    """
    prefix_res = [
        (name, re.compile(r"^\s*" + re.escape(name) + r"\s*:\s?(.*)$"))
        for name in speaker_names
    ]
    segments = []  # list of [speaker, [line, ...]]
    current = None
    for raw_line in text.splitlines():
        matched_name = None
        matched_rest = ""
        for name, pattern in prefix_res:
            m = pattern.match(raw_line)
            if m:
                matched_name = name
                matched_rest = m.group(1)
                break
        if matched_name is not None:
            current = [matched_name, [matched_rest]]
            segments.append(current)
        else:
            if current is None:
                current = [speaker_names[0], []]
                segments.append(current)
            current[1].append(raw_line)
    parts = []
    for speaker, lines in segments:
        seg_text = "\n".join(lines).strip()
        if seg_text:
            parts.append({
                "text": seg_text,
                "speech_metadata": {"speaker": speaker},
            })
    return parts


# Burmese tone words the apps offer as one-tap style chips, mapped to
# concise bracket-tag words. Live-verified 2026-11-05: the model READS
# ALOUD any "Say in ... tone: " director's-note prefix — English or
# Burmese — as if it were content (a styled sample came back longer,
# with the prefix spoken as extra bursts), while a bracketed tag like
# "[angry]" placed before the text is treated as markup, never spoken,
# and still changes the delivery. So every style becomes a tag.
_GEMINI_STYLE_TAGS = [
    ("ဒေါသ", "angry"),
    ("ပျော်", "cheerful"),
    ("ရွှင်လန်း", "cheerful"),
    ("တည်ငြိမ်", "calm"),
    ("နွေးထွေး", "warm"),
    ("ဝမ်းနည်း", "sad"),
    ("တိုးတိုး", "whispering"),
    ("စိတ်လှုပ်ရှား", "excited"),
    ("ပုံပြင်", "storytelling"),
    ("သတင်း", "newsreader"),
    ("နှေး", "slow"),
    ("မြန်", "fast"),
    ("မောပန်း", "out of breath"),
    ("ပျင်းရိ", "bored"),
    ("ကြင်နာ", "gentle"),
    ("ကြောက်", "fearful"),
    ("အံ့ဩ", "surprised"),
    ("ယုံကြည်", "confident"),
    ("လေးနက်", "serious"),
    ("ဟာသ", "humorous"),
    ("ရှက်", "shy"),
]
# Longest keys first, so a short key that happens to be a substring of
# a longer one can never win the match.
_GEMINI_STYLE_TAGS.sort(key=lambda kv: -len(kv[0]))

# Known English tone phrases — the values the apps' preset chips send
# (the private app's TONE_PRESETS and the public app's
# GEM_STYLE_PRESETS), keyed by their normalized form (see
# _gemini_style_tag) — mapped to concise tag words.
_GEMINI_EN_TAG_MAP = {
    "angry": "angry",
    "cheerful": "cheerful",
    "cheerful and friendly": "cheerful",
    "calm": "calm",
    "calm and relaxed": "calm",
    "warm": "warm",
    "warm and enthusiastic": "warm",
    "sad": "sad",
    "sad and heartfelt": "sad",
    "whispered": "whispering",
    "whispered urgently": "whispering",
    "whispering": "whispering",
    "excited": "excited",
    "excited, high-energy": "excited",
    "excited, high energy": "excited",
    "excited high-energy": "excited",
    "excited high energy": "excited",
    "storytelling": "storytelling",
    "warm storytelling": "storytelling",
    "newsreader": "newsreader",
    "clear newsreader": "newsreader",
    "clear newsreader delivery": "newsreader",
    "slow": "slow",
    "speaking slowly": "slow",
    "rapid": "fast",
    "fast": "fast",
    "speaking rapidly": "fast",
    "out of breath": "out of breath",
    "bored": "bored",
    "bored, monotone": "bored",
    "bored and monotone": "bored",
    "gentle": "gentle",
    "gentle and kind": "gentle",
    "fearful": "fearful",
    "surprised": "surprised",
    "confident": "confident",
    "serious": "serious",
    "humorous": "humorous",
    "shy": "shy",
}

# Any character in the Myanmar Unicode block.
_MY_CHARS_RE = re.compile(r"[\u1000-\u109f]")


def _gemini_style_tag(style: str) -> str:
    """Convert a client style into a bracketed audio tag, e.g. "[angry] ".

    Returns "" for an empty style. Resolution order:
    1. A known Burmese tone word (contains-match, longest key first)
       maps to its concise English tag word.
    2. An English style is normalized — lowercase, collapse
       whitespace, drop a leading "say in ", drop a trailing
       " tone of voice" / " tone" / " voice", drop a leading article —
       then mapped to a concise tag when the phrase is known; any
       other cleaned English phrase becomes the tag content itself
       ("very happy" -> "[very happy] ").
    3. Anything else (unmapped Burmese, or mixed text) is bracketed
       as-is: bracketed text sits in markup position and is not read
       aloud, strictly better than a bare instruction string, which
       the model speaks.
    """
    s = (style or "").strip()
    if not s:
        return ""
    for key, tag in _GEMINI_STYLE_TAGS:
        if key in s:
            return f"[{tag}] "
    if not _MY_CHARS_RE.search(s):
        e = re.sub(r"\s+", " ", s.lower()).strip(" :.")
        if e.startswith("say in "):
            e = e[len("say in "):]
        for suffix in (" tone of voice", " tone", " voice"):
            if e.endswith(suffix):
                e = e[: -len(suffix)]
                break
        for article in ("a ", "an ", "the "):
            if e.startswith(article):
                e = e[len(article):]
                break
        e = e.strip(" :.")
        if e:
            return f"[{_GEMINI_EN_TAG_MAP.get(e, e)}] "
    return f"[{s}] "


# A USER TAG is a bracketed group at the very start of a line (after
# optional leading spaces): letters, spaces and hyphens only, e.g.
# "[bored]" or "[cheerful and friendly]". Users type these to style one
# line at a time; the model treats them as markup anywhere in the text.
# Only line-leading brackets count — a bracket mid-line is content.
_USER_TAG_LINE_RE = re.compile(r"^\s*\[([A-Za-z][A-Za-z \-]{0,40})\]")


def _first_user_tag_offset(text: str):
    """Character offset of the start of the first line that opens with
    a user tag, or None when the text has no user tag."""
    offset = 0
    for line in text.splitlines(keepends=True):
        if _USER_TAG_LINE_RE.match(line):
            return offset
        offset += len(line)
    return None


def _starts_with_user_tag(text: str) -> bool:
    return bool(_USER_TAG_LINE_RE.match(text or ""))


def _apply_style_tag_single(text: str, style_tag: str) -> str:
    """Combine the global style tag with a single-voice text body.

    No user tags -> global tag prepended exactly as before. With user
    tags: the global tag is prepended ONLY when real (non-tag) text
    precedes the first user tag — when the text opens with a user tag,
    that tag governs from the start and a prepended global tag would
    fight it, so it is skipped. User tags stay inline verbatim either
    way. An empty style_tag returns the text untouched.
    """
    if not style_tag:
        return text
    tag_at = _first_user_tag_offset(text)
    if tag_at is None:
        return style_tag + text
    if text[:tag_at].strip():
        return style_tag + text
    return text


def _pcm_to_wav(pcm: bytes, sample_rate: int, channels: int = 1,
                bits_per_sample: int = 16) -> bytes:
    """Wrap raw PCM audio in a standard 44-byte WAV header."""
    byte_rate = sample_rate * channels * bits_per_sample // 8
    block_align = channels * bits_per_sample // 8
    header = b"".join([
        b"RIFF", struct.pack("<I", 36 + len(pcm)), b"WAVE",
        b"fmt ",
        struct.pack(
            "<IHHIIHH", 16, 1, channels, sample_rate,
            byte_rate, block_align, bits_per_sample,
        ),
        b"data", struct.pack("<I", len(pcm)),
    ])
    return header + pcm


def _clean_gemini_pcm(pcm: bytes, sample_rate: int) -> bytes:
    """Trim edge junk from Gemini's raw PCM (16-bit little-endian mono).

    Live captures show two artifacts: impulse spikes within the first
    ~10 ms (then silence before the speech starts), and a short loud
    DC-heavy segment that appears AFTER a >=200 ms silence gap
    following the last speech — a detached trailing "blip" that sounds
    like a click/thud at the very end. This trims leading spikes and
    silence up to the first sustained speech, cuts everything after
    the real speech end (+40 ms pad), and applies a 10 ms linear
    fade-in / 15 ms linear fade-out at the cut points so the cuts
    themselves never click.

    Analysis uses 10 ms windows (per-window RMS and mean). The tail
    walk skips windows that are quiet (rms <= 0.02) or DC-heavy
    (|mean| > 0.04 and |mean| >= 0.3 * rms). One refinement over a
    plain backward walk: a loud run detached from the speech by a
    >=150 ms quiet gap whose windows lean DC ON AVERAGE is treated as
    one artifact blip and skipped whole — in the captured fixture the
    blip's individual windows straddle the 0.3 DC ratio, so a purely
    per-window walk stops inside the junk. A detached run that does
    NOT lean DC is kept (it could be a real final word after a pause).

    Safety: if no sustained speech is found, or the trim would remove
    more than 40% of the total duration, the original PCM is returned
    unchanged. Samples in the middle are never modified (the fades
    touch only the first 10 ms / last 15 ms of the kept region).
    """
    if sample_rate <= 0 or len(pcm) < 2:
        return pcm
    samples = array("h")
    samples.frombytes(pcm[: len(pcm) - (len(pcm) % 2)])
    if sys.byteorder == "big":
        samples.byteswap()
    n = len(samples)
    win = max(1, sample_rate // 100)  # 10 ms
    nwin = n // win
    if nwin < 10:
        return pcm

    rms = [0.0] * nwin
    mean = [0.0] * nwin
    for i in range(nwin):
        seg = samples[i * win:(i + 1) * win]
        total = 0
        sq = 0
        for x in seg:
            total += x
            sq += x * x
        mean[i] = (total / win) / 32768.0
        rms[i] = math.sqrt(sq / win) / 32768.0

    LOUD = 0.02

    def is_dc(i):
        return abs(mean[i]) > 0.04 and abs(mean[i]) >= 0.3 * rms[i]

    # Head: first window that is loud AND sustained (at least one of
    # the next 3 windows also loud). A lone loud window — the opening
    # spike — does not qualify. Keep from one window before it.
    start_win = None
    for i in range(nwin):
        if rms[i] > LOUD and any(
            rms[j] > LOUD for j in range(i + 1, min(i + 4, nwin))
        ):
            start_win = i
            break
    if start_win is None:
        return pcm
    start_sample = max(0, start_win - 1) * win

    # Tail: last real speech window, searching backwards from `frm`.
    def last_speech_win(frm):
        i = frm
        while i >= 0:
            if rms[i] <= LOUD or is_dc(i):
                i -= 1
                continue
            # Candidate end i: find the non-quiet block containing it
            # and the quiet gap immediately before that block.
            lo = i
            while lo - 1 >= 0 and rms[lo - 1] > LOUD:
                lo -= 1
            hi = i
            while hi + 1 < nwin and rms[hi + 1] > LOUD:
                hi += 1
            g = lo - 1
            gap = 0
            while g >= 0 and rms[g] <= LOUD:
                gap += 1
                g -= 1
            if gap >= 15 and g >= 0:
                block_mean = sum(mean[lo:hi + 1]) / (hi - lo + 1)
                if abs(block_mean) > 0.04:
                    # Detached DC-leaning blip, not speech: resume the
                    # search at the last window before the gap. `g`
                    # strictly decreases each hop, so this terminates.
                    return last_speech_win(g)
            return i
        return None

    end_win = last_speech_win(nwin - 1)
    if end_win is None:
        return pcm
    pad = sample_rate * 40 // 1000  # 40 ms
    end_sample = min(n, (end_win + 1) * win + pad)
    if end_sample <= start_sample:
        return pcm
    if start_sample + (n - end_sample) > 0.40 * n:
        return pcm
    if start_sample == 0 and end_sample == n:
        return pcm

    kept = samples[start_sample:end_sample]
    fade_in = min(sample_rate // 100, len(kept))  # 10 ms
    for i in range(fade_in):
        kept[i] = int(kept[i] * (i / fade_in))
    fade_out = min(sample_rate * 15 // 1000, len(kept))  # 15 ms
    base = len(kept) - fade_out
    for i in range(fade_out):
        kept[base + i] = int(kept[base + i] * ((fade_out - i) / fade_out))
    if sys.byteorder == "big":
        kept.byteswap()
    return kept.tobytes()


@app.get("/api/gemini/status")
async def gemini_status(request: Request):
    """Report whether Gemini TTS is configured, and which model is in use."""
    _require_owner(request)
    if not GEMINI_API_KEY:
        return {"configured": False}
    return {"configured": True, "model": GEMINI_TTS_MODEL}


@app.post("/api/gemini/tts")
async def gemini_tts(req: GeminiTTSRequest, request: Request):
    """Generate speech via Gemini TTS using the server-side key."""
    _require_owner(request)
    if not GEMINI_API_KEY:
        raise HTTPException(
            status_code=503,
            detail="Gemini is not configured on the server (GEMINI_API_KEY missing)",
        )
    return await _gemini_tts_core(req, GEMINI_API_KEY)


async def _gemini_tts_core(req: GeminiTTSRequest, api_key: str):
    """Gemini TTS using the given key (server key, or a visitor's own
    key on the public route). The key is used for this call only.

    Gemini returns base64-encoded raw PCM (L16); it is wrapped in a WAV
    container before being returned as audio/wav.
    """
    if not req.text or not req.text.strip():
        raise HTTPException(status_code=400, detail="Text is required")
    if len(req.text) > MAX_TEXT_CHARS:
        raise HTTPException(
            status_code=400,
            detail=f"Text is too long: maximum {MAX_TEXT_CHARS} characters per request",
        )

    style = ""
    if req.style is not None:
        if not isinstance(req.style, str):
            raise HTTPException(
                status_code=400,
                detail="style must be a string (a tone/style instruction)",
            )
        style = req.style.strip()

    # Bracketed-tag form of the style ("" when no style): any spoken
    # instruction form would be read aloud by the model as content.
    style_tag = _gemini_style_tag(style)

    text = req.text.strip()

    if req.speakers is not None:
        speaker_configs = _gemini_speaker_configs(req.speakers)
        speech_config = {
            "multiSpeakerVoiceConfig": {
                "speakerVoiceConfigs": speaker_configs,
            },
        }
        # The 3.8 model requires the dialogue split into per-speaker
        # parts, each tagged with speech_metadata (see
        # _gemini_dialogue_parts); one shared text part is rejected.
        parts = _gemini_dialogue_parts(
            text, [c["speaker"] for c in speaker_configs],
        )
        speakers_present = {p["speech_metadata"]["speaker"] for p in parts}
        if len(speakers_present) < 2:
            raise HTTPException(
                status_code=400,
                detail="Dialogue text must give each speaker at least one "
                       "line, prefixed with '<speaker>:' "
                       "(e.g. 'A: ...' and 'B: ...')",
            )
        if style_tag:
            # Put the tag at the start of the FIRST segment's text so
            # it rides on a part that names its speaker: a separate
            # leading part without speech_metadata would break the
            # multi-speaker rule that every part names its speaker.
            # Don't double-tag: when the first segment already opens
            # with the user's own per-line tag, that tag governs and
            # the global tag is skipped for this part.
            if not _starts_with_user_tag(parts[0]["text"]):
                parts[0]["text"] = style_tag + parts[0]["text"]
    else:
        speech_config = {
            "voiceConfig": {
                "prebuiltVoiceConfig": {"voiceName": req.voice},
            },
        }
        # Wire format stays ONE text part; per-line user tags ride
        # inline (see _apply_style_tag_single for the combination rule).
        text = _apply_style_tag_single(text, style_tag)
        parts = [{"text": text}]

    payload = {
        "contents": [{"parts": parts}],
        "generationConfig": {
            "responseModalities": ["AUDIO"],
            "speechConfig": speech_config,
        },
    }

    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(300.0)) as client:
            resp = await client.post(
                f"{GEMINI_BASE}/v1beta/models/{GEMINI_TTS_MODEL}:generateContent",
                json=payload,
                headers={
                    "x-goog-api-key": api_key,
                    "Content-Type": "application/json",
                },
            )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Gemini service error: {e}")

    if resp.status_code != 200:
        raise HTTPException(
            status_code=resp.status_code,
            detail=f"Gemini error ({resp.status_code}): {_gemini_error_detail(resp)}",
        )

    try:
        data = resp.json()
        parts = data["candidates"][0]["content"]["parts"]
        if not isinstance(parts, list):
            parts = []
    except Exception:
        parts = []

    inline = None
    for part in parts:
        if isinstance(part, dict):
            candidate = part.get("inlineData")
            if isinstance(candidate, dict) and candidate.get("data"):
                inline = candidate
                break

    if inline is None:
        raise HTTPException(
            status_code=502,
            detail=f"Gemini returned no audio: {resp.text[:200]}",
        )

    mime = inline.get("mimeType") or ""
    rate_match = re.search(r"rate=(\d+)", mime)
    sample_rate = int(rate_match.group(1)) if rate_match else 24000

    try:
        pcm = base64.b64decode(inline["data"])
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Gemini audio decode failed: {e}")

    # Strip the edge junk Gemini sometimes returns (head spikes, a
    # detached DC-heavy blip after the speech) before wrapping.
    pcm = _clean_gemini_pcm(pcm, sample_rate)

    return Response(content=_pcm_to_wav(pcm, sample_rate), media_type="audio/wav")


# --- Public edition routes (/api/pub/*) ---
# Every route here requires a valid Supabase access token. For the
# paid/quota engines the provider key is resolved per request by
# _resolve_pub_key (phase 11): Pro code (or owner login) -> the
# server's env key; visitor key header -> the visitor's own key.

_jwks_cache: dict = {"keys": None, "fetched_at": 0.0}
JWKS_CACHE_SECONDS = 3600

_PUB_401_MSG = (
    "Invalid or expired login token. Please sign in again. "
    "(အကောင့် ပြန်ဝင်ရန် လိုအပ်ပါတယ်)"
)


async def _fetch_jwks(force: bool = False) -> list:
    """Fetch (and cache for an hour) the Supabase project's JWKS.

    Infrastructure calls (JWKS + the pro-code store) use
    trust_env=False: they must go directly to Supabase, never through
    whatever proxy variables the host environment happens to carry.
    """
    now = time.time()
    if (
        not force
        and _jwks_cache["keys"] is not None
        and now - _jwks_cache["fetched_at"] < JWKS_CACHE_SECONDS
    ):
        return _jwks_cache["keys"]
    async with httpx.AsyncClient(timeout=httpx.Timeout(15.0), trust_env=False) as client:
        resp = await client.get(SUPABASE_JWKS_URL)
        resp.raise_for_status()
        keys = resp.json().get("keys", [])
    if not isinstance(keys, list):
        keys = []
    _jwks_cache["keys"] = keys
    _jwks_cache["fetched_at"] = now
    return keys


def _signing_key_for(header: dict, keys: list):
    """Build a verification key from the JWKS entry matching the token."""
    kid = header.get("kid")
    for jwk in keys:
        if not isinstance(jwk, dict):
            continue
        if kid is not None and jwk.get("kid") != kid:
            continue
        kty = jwk.get("kty")
        algo_cls = {"RSA": _RSAAlgorithm, "EC": _ECAlgorithm,
                    "OKP": _OKPAlgorithm}.get(kty)
        if algo_cls is None:
            continue
        try:
            # PyJWT >= 2.10 renamed from_jwk_data() to from_jwk().
            from_jwk = getattr(algo_cls, "from_jwk", None) or getattr(
                algo_cls, "from_jwk_data")
            return from_jwk(json.dumps(jwk))
        except Exception:
            continue
    return None


async def _verify_supabase_token(token: str) -> dict:
    """Verify a Supabase access token against the project JWKS.

    Checks the signature, expiry (exp) and issuer (iss). Every failure
    raises the same generic 401 — token material is never echoed in
    error messages or logs.
    """
    try:
        header = _pyjwt.get_unverified_header(token)
    except Exception:
        raise HTTPException(status_code=401, detail=_PUB_401_MSG)
    try:
        keys = await _fetch_jwks()
    except Exception:
        keys = []
    key = _signing_key_for(header, keys)
    if key is None:
        # The signing key may have rotated since the cache was filled:
        # refetch once before giving up.
        try:
            keys = await _fetch_jwks(force=True)
        except Exception:
            keys = []
        key = _signing_key_for(header, keys)
    if key is None:
        raise HTTPException(status_code=401, detail=_PUB_401_MSG)
    try:
        return _pyjwt.decode(
            token,
            key,
            algorithms=["RS256", "ES256", "EdDSA"],
            # Supabase user access tokens carry aud="authenticated";
            # PyJWT refuses tokens with an aud claim unless the
            # expected audience is given (anon/service tokens differ).
            audience="authenticated",
            issuer=SUPABASE_ISSUER,
            options={"require": ["exp", "iss", "sub"]},
        )
    except Exception:
        raise HTTPException(status_code=401, detail=_PUB_401_MSG)


async def _require_pub_identity(request: Request):
    """Gate for every /api/pub/* route; returns (user_id, email).

    The email is the JWT's verified email claim, lowercased, or None
    when the token carries none. Fails CLOSED: when SUPABASE_URL is
    not configured (or the JWT library is missing) the public routes
    return 503 — never open.
    """
    if not SUPABASE_URL or _pyjwt is None:
        raise HTTPException(
            status_code=503,
            detail="Public access is not configured on this server. "
                   "(Public အသုံးပြုမှု မဖွင့်ထားပါ)",
        )
    auth = request.headers.get("Authorization", "")
    token = auth[7:].strip() if auth.lower().startswith("bearer ") else ""
    if not token:
        raise HTTPException(
            status_code=401,
            detail="Login required: send your Supabase access token as "
                   "'Authorization: Bearer <token>'. "
                   "(အကောင့် ဝင်ထားရန် လိုအပ်ပါတယ်)",
        )
    payload = await _verify_supabase_token(token)
    sub = payload.get("sub")
    if not sub:
        raise HTTPException(status_code=401, detail=_PUB_401_MSG)
    email = payload.get("email")
    email = str(email).strip().lower() if email else None
    return str(sub), email


async def _require_pub_user(request: Request) -> str:
    """Gate for the free public route; returns the user's id (sub)."""
    user_id, _email = await _require_pub_identity(request)
    return user_id


def _require_byok_key(request: Request, header: str, service: str) -> str:
    """Return the visitor's own provider key from a header, or 401.

    The public routes have no fallback to the server's keys: without
    the visitor's key the request stops here.
    """
    key = (request.headers.get(header) or "").strip()
    if not key:
        raise HTTPException(
            status_code=401,
            detail=f"Your own {service} API key is required: send it in "
                   f"the {header} header. The public server never uses "
                   f"the owner's key. (သင့်ကိုယ်ပိုင် {service} API key "
                   f"ကို {header} header ဖြင့် ပေးပို့ပါ)",
        )
    return key


# --- Phase 11: key/code resolution for the public Pro routes ---
# A Pro call runs on exactly one key, resolved in this order:
#   1. X-Pro-Code header present -> validate the code (existing rules,
#      owner-email bypass included); valid -> the SERVER's env key.
#      A presented code wins over a visitor key: an invalid code is a
#      403 even when a visitor key is also sent.
#   2. Caller is the owner (verified JWT email == OWNER_EMAIL) -> the
#      server's env key, no code involved.
#   3. Visitor key header present -> the visitor's own key; no code is
#      read, validated or bound.
#   4. Neither -> 401.
# On the server-key paths (1-2) a missing env key is a 503 — the route
# never silently falls back to anything else.

_KEY_OR_CODE_401_MSG = (
    "An API key or a Pro code is required. "
    "(ကိုယ့် API Key ထည့်ပါ — ဒါမှမဟုတ် Pro Code သုံးပါ)"
)

_PRO_NOT_CONFIGURED_MSG = (
    "Pro service is not configured on the server. "
    "(Pro service ကို server မှာ မထည့်ထားပါ)"
)

_VOICE_OWNER_ONLY_MSG = (
    "Adding or deleting voices needs your own ElevenLabs key. "
    "(အသံ ထည့်/ဖျက် ချင်ရင် ကိုယ့် ElevenLabs key သုံးပါ)"
)

_EL_CAP_429_MSG = (
    "Daily limit for Pro-code ElevenLabs use reached. "
    "(Pro Code နဲ့ ElevenLabs သုံးတဲ့ နေ့စဉ် limit ပြည့်သွားပါပြီ)"
)


def _is_owner_email(user_email) -> bool:
    """True when the verified JWT email is the owner's (OWNER_EMAIL,
    case-insensitive). Disabled when OWNER_EMAIL is unset."""
    return bool(
        OWNER_EMAIL and user_email
        and str(user_email).strip().lower() == OWNER_EMAIL.lower()
    )


async def _resolve_pub_key(request: Request, user_id: str,
                           user_email: str, header_name: str,
                           env_name: str):
    """Resolve the provider key for a public Pro call (phase 11).

    Returns (api_key, code, needs_bind, used_server_key); `code` and
    `needs_bind` are the _require_pro_code results to hand to
    _pro_code_touch after a successful call (None/False when no code
    was involved). Raises 403 for an invalid presented code, 503 when
    a server-key path has no env key configured, and 401 when the
    caller brought neither a code nor a key.
    """
    if (request.headers.get("X-Pro-Code") or "").strip():
        code, needs_bind = await _require_pro_code(
            request, user_id, user_email)
        env_key = (globals().get(env_name) or "").strip()
        if not env_key:
            raise HTTPException(status_code=503,
                                detail=_PRO_NOT_CONFIGURED_MSG)
        return env_key, code, needs_bind, True
    if _is_owner_email(user_email):
        env_key = (globals().get(env_name) or "").strip()
        if not env_key:
            raise HTTPException(status_code=503,
                                detail=_PRO_NOT_CONFIGURED_MSG)
        return env_key, None, False, True
    visitor_key = (request.headers.get(header_name) or "").strip()
    if visitor_key:
        return visitor_key, None, False, False
    raise HTTPException(status_code=401, detail=_KEY_OR_CODE_401_MSG)


async def _pub_optional_code(request: Request, user_id: str,
                             user_email: str):
    """Code handling for the clone-job poll routes: they carry no
    upstream key, so there is nothing to resolve — a presented
    X-Pro-Code is validated (and later stamped) exactly as before,
    but a caller without one (e.g. bring-your-own-key) may still poll
    their own jobs; job ownership is the real gate there."""
    if (request.headers.get("X-Pro-Code") or "").strip():
        return await _require_pro_code(request, user_id, user_email)
    return None, False


# ElevenLabs daily char cap for server-key callers (phase 11):
# user_id -> (UTC date ISO string, chars used that day). In-memory by
# design — a restart resets the day count, never the other way.
_el_server_daily: dict = {}


def _el_server_cap_check(user_id: str, n_chars: int):
    """429 when this server-key call would push the user past today's
    ElevenLabs char cap (checked BEFORE the upstream call)."""
    today = datetime.now(timezone.utc).date().isoformat()
    day, used = _el_server_daily.get(user_id, (today, 0))
    if day != today:
        used = 0
    if used + n_chars > SERVER_EL_DAILY_CHAR_CAP:
        raise HTTPException(status_code=429, detail=_EL_CAP_429_MSG)


def _el_server_cap_add(user_id: str, n_chars: int):
    """Record a successful server-key ElevenLabs call's chars."""
    today = datetime.now(timezone.utc).date().isoformat()
    day, used = _el_server_daily.get(user_id, (today, 0))
    if day != today:
        used = 0
    _el_server_daily[user_id] = (today, used + n_chars)


_pub_synth_hits_user: dict = {}
_pub_synth_hits_ip: dict = {}


def _client_ip(request: Request) -> str:
    fwd = request.headers.get("x-forwarded-for", "")
    if fwd:
        return fwd.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def _pub_synth_rate_check(user_id: str, ip: str):
    """Sliding-window limit for the free public Edge route: at most
    PUB_SYNTH_LIMIT requests per PUB_SYNTH_WINDOW per user, with the
    same limit per IP as a secondary guard."""
    now = time.time()
    cutoff = now - PUB_SYNTH_WINDOW
    buckets = ((_pub_synth_hits_user, user_id), (_pub_synth_hits_ip, ip))
    for bucket, key in buckets:
        dq = bucket.get(key)
        if dq:
            while dq and dq[0] <= cutoff:
                dq.popleft()
            if len(dq) >= PUB_SYNTH_LIMIT:
                raise HTTPException(
                    status_code=429,
                    detail=f"Rate limit reached: the free public voice "
                           f"allows {PUB_SYNTH_LIMIT} requests per hour. "
                           "Please try again later. "
                           "(တစ်နာရီ အကြိမ်ရေ ပြည့်သွားပါပြီ — "
                           "ခဏနေ မှ ပြန် စမ်းပါ)",
                )
    for bucket, key in buckets:
        bucket.setdefault(key, deque()).append(now)


@app.post("/api/pub/synthesize")
async def pub_synthesize(req: TTSRequest, request: Request):
    """Free public Edge TTS: login required, rate-limited, short texts."""
    user_id = await _require_pub_user(request)
    await _check_mode_enabled(user_id, "edge")
    if not req.text or not req.text.strip():
        raise HTTPException(status_code=400, detail="Text is required")
    if len(req.text) > PUB_SYNTH_MAX_CHARS:
        raise HTTPException(
            status_code=400,
            detail=f"Text is too long: the free public route allows at "
                   f"most {PUB_SYNTH_MAX_CHARS} characters per request. "
                   f"(အများဆုံး စာလုံး {PUB_SYNTH_MAX_CHARS} လုံးသာ ရပါတယ်)",
        )
    _pub_synth_rate_check(user_id, _client_ip(request))
    try:
        output_path = await _edge_synth_file(
            req.text.strip(), req.voice, req.rate, req.pitch,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return _mp3_file_response(output_path)


@app.post("/api/pub/el/tts")
async def pub_el_tts(req: ELTTSRequest, request: Request):
    """Public ElevenLabs TTS. Key resolution (phase 11): a valid Pro
    code or the owner runs on the server's key (daily char cap
    applies); otherwise the visitor's own X-EL-Key, no code needed."""
    user_id, user_email = await _require_pub_identity(request)
    await _check_mode_enabled(user_id, "el")
    api_key, code, needs_bind, server_key = await _resolve_pub_key(
        request, user_id, user_email, "X-EL-Key", "EL_API_KEY")
    if server_key:
        _el_server_cap_check(user_id, len(req.text))
    resp = await _el_tts_core(req, api_key)
    if server_key:
        _el_server_cap_add(user_id, len(req.text))
    await _pro_code_touch(code, user_id, needs_bind)
    return resp


@app.get("/api/pub/el/voices")
async def pub_el_voices(request: Request):
    """List cloned/professional voices on the resolved key's account:
    the visitor's own (X-EL-Key), or the server's for a Pro-code /
    owner caller — code users need to see the owner's clones."""
    user_id, user_email = await _require_pub_identity(request)
    await _check_mode_enabled(user_id, "el")
    api_key, code, needs_bind, server_key = await _resolve_pub_key(
        request, user_id, user_email, "X-EL-Key", "EL_API_KEY")
    resp = await _el_voices_core(api_key, own_only=True)
    await _pro_code_touch(code, user_id, needs_bind)
    return resp


@app.post("/api/pub/el/voices")
async def pub_el_voice_add(request: Request, name: str = Form(...), sample: UploadFile = File(...)):
    """Create a cloned voice with the resolved key (phase 11). On the
    server-key path only the OWNER may create voices — a Pro-code
    caller would otherwise add voices to the owner's account — so a
    non-owner code user gets a 403 telling them to use their own key;
    bring-your-own-key callers create on their own account as before."""
    user_id, user_email = await _require_pub_identity(request)
    await _check_mode_enabled(user_id, "el")
    api_key, code, needs_bind, server_key = await _resolve_pub_key(
        request, user_id, user_email, "X-EL-Key", "EL_API_KEY")
    if server_key and not _is_owner_email(user_email):
        raise HTTPException(status_code=403, detail=_VOICE_OWNER_ONLY_MSG)
    if not name or not name.strip():
        raise HTTPException(status_code=400, detail="Voice name is required")
    sample_bytes = await _read_upload_capped(sample)
    if not sample_bytes:
        raise HTTPException(status_code=400, detail="Sample audio file is empty")
    resp = await _el_voice_add_core(
        name.strip(), sample_bytes,
        sample.filename or "sample.mp3",
        sample.content_type or "audio/mpeg",
        api_key,
    )
    await _pro_code_touch(code, user_id, needs_bind)
    return resp


@app.delete("/api/pub/el/voices/{voice_id}")
async def pub_el_voice_delete(voice_id: str, request: Request):
    """Delete a voice with the resolved key (phase 11). Only
    cloned/professional voices can
    be deleted: the voice's category is checked with the resolved key
    BEFORE the upstream delete is called, so premade voices get a
    clean 400 here instead of an upstream delete error. On the
    server-key path only the OWNER may delete (phase 11)."""
    user_id, user_email = await _require_pub_identity(request)
    await _check_mode_enabled(user_id, "el")
    api_key, code, needs_bind, server_key = await _resolve_pub_key(
        request, user_id, user_email, "X-EL-Key", "EL_API_KEY")
    if server_key and not _is_owner_email(user_email):
        raise HTTPException(status_code=403, detail=_VOICE_OWNER_ONLY_MSG)
    voice = await _el_voice_lookup_core(voice_id, api_key)
    if voice.get("category") not in ("cloned", "professional"):
        raise HTTPException(
            status_code=400,
            detail="This voice cannot be deleted. (ဒီ အသံ ကို ဖျက်လို့ မရပါ)",
        )
    await _el_voice_delete_core(voice_id, api_key)
    await _pro_code_touch(code, user_id, needs_bind)
    return {"ok": True}


@app.post("/api/pub/gemini/tts")
async def pub_gemini_tts(req: GeminiTTSRequest, request: Request):
    """Public Gemini TTS (3.8 contract). Key resolution (phase 11):
    Pro code / owner -> the server's key; otherwise the visitor's own
    X-Gemini-Key, no code needed."""
    user_id, user_email = await _require_pub_identity(request)
    await _check_mode_enabled(user_id, "gemini")
    api_key, code, needs_bind, server_key = await _resolve_pub_key(
        request, user_id, user_email, "X-Gemini-Key", "GEMINI_API_KEY")
    resp = await _gemini_tts_core(req, api_key)
    await _pro_code_touch(code, user_id, needs_bind)
    return resp


@app.post("/api/pub/clone/jobs")
async def pub_clone_job_create(
    request: Request,
    text: str = Form(...),
    ref_text: str = Form(...),
    speed: float = Form(1.0),
    ref_audio: UploadFile = File(...),
):
    """Public clone job (phase 11 key resolution): a valid Pro code
    or the owner runs the job on the server's HF_TOKEN; otherwise the
    visitor's own X-HF-Token, no code needed.

    The resolved token is handed to the background job for its
    upstream calls only; it is not written into the job entry or
    anywhere else.
    """
    user_id, user_email = await _require_pub_identity(request)
    await _check_mode_enabled(user_id, "f5")
    token, code, needs_bind, server_key = await _resolve_pub_key(
        request, user_id, user_email, "X-HF-Token", "HF_TOKEN")
    if not text.strip():
        raise HTTPException(status_code=400, detail="Text is required")
    if not ref_text.strip():
        raise HTTPException(status_code=400, detail="Reference transcript is required")
    audio_bytes = await _read_upload_capped(ref_audio)
    _sweep_jobs()
    job_id = uuid.uuid4().hex[:12]
    jobs[job_id] = {"status": "queued", "owner": user_id}
    asyncio.create_task(_job_work(
        job_id, text.strip(), ref_text.strip(), speed,
        audio_bytes, ref_audio.filename or "ref.wav",
        ref_audio.content_type or "audio/wav",
        token,
    ))
    await _pro_code_touch(code, user_id, needs_bind)
    return {"job_id": job_id, "status": "queued"}


def _pub_job_or_404(job_id: str, user_id: str) -> dict:
    _sweep_jobs()
    job = jobs.get(job_id)
    if not job or job.get("owner") != user_id:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@app.get("/api/pub/clone/jobs/{job_id}")
async def pub_clone_job_status(job_id: str, request: Request):
    user_id, user_email = await _require_pub_identity(request)
    code, needs_bind = await _pub_optional_code(request, user_id, user_email)
    job = _pub_job_or_404(job_id, user_id)
    await _pro_code_touch(code, user_id, needs_bind)
    return _job_status_payload(job_id, job)


@app.get("/api/pub/clone/jobs/{job_id}/audio")
async def pub_clone_job_audio(job_id: str, request: Request):
    user_id, user_email = await _require_pub_identity(request)
    code, needs_bind = await _pub_optional_code(request, user_id, user_email)
    job = _pub_job_or_404(job_id, user_id)
    if job["status"] != "done":
        raise HTTPException(status_code=409, detail=f"Job not ready: {job['status']}")
    await _pro_code_touch(code, user_id, needs_bind)
    return Response(content=job["audio"], media_type="audio/wav")


# --- Owner lock + Pro codes (phase 4) ---
# The private paid routes (/api/el/*, /api/gemini/*, /api/clone*)
# require the X-Owner-Key header to match OWNER_KEY. The
# /api/admin/* endpoints accept that key too, or alternatively the
# owner's Supabase login (see _require_admin). When no admin path is
# configured those routes return 503 (locked) rather than open. The
# public Pro routes use the codes below via _resolve_pub_key (phase
# 11): a valid code is one way to run on the server's keys.

def _require_owner(request: Request):
    """Gate for the owner's private paid routes (key only)."""
    if not OWNER_KEY:
        raise HTTPException(
            status_code=503,
            detail="Owner access is locked on this server "
                   "(OWNER_KEY not configured).",
        )
    supplied = (request.headers.get("X-Owner-Key") or "").strip()
    if not supplied or not hmac.compare_digest(supplied, OWNER_KEY):
        raise HTTPException(
            status_code=401,
            detail="Owner key required or invalid. "
                   "(ပိုင်ရှင် key လိုအပ်ပါတယ်)",
        )


_OWNER_401_MSG = (
    "Owner key required or invalid. "
    "(ပိုင်ရှင် key လိုအပ်ပါတယ်)"
)

_OWNER_503_MSG = (
    "Owner access is locked on this server "
    "(OWNER_KEY not configured)."
)


async def _require_admin(request: Request):
    """Gate for the /api/admin/* endpoints: owner key OR owner login.

    Passes when EITHER
    (a) the X-Owner-Key header matches OWNER_KEY (the original
        behaviour, unchanged), or
    (b) an "Authorization: Bearer <jwt>" is present, the token verifies
        against the Supabase project JWKS (same helper as the public
        routes), and its verified email claim equals OWNER_EMAIL,
        compared case-insensitively.
    Path (b) is disabled when OWNER_EMAIL or SUPABASE_URL is unset (or
    the JWT library is unavailable). The owner key is checked first so
    a valid key never depends on the JWKS fetch. When neither path is
    configured the endpoints stay locked (503); any other failure is
    the same generic 401 as the owner-key gate — no auth material (key
    or token) is echoed in errors or logs.
    """
    if OWNER_KEY:
        supplied = (request.headers.get("X-Owner-Key") or "").strip()
        if supplied and hmac.compare_digest(supplied, OWNER_KEY):
            return
    jwt_path = bool(OWNER_EMAIL and SUPABASE_URL and _pyjwt is not None)
    if jwt_path:
        auth = request.headers.get("Authorization", "")
        token = auth[7:].strip() if auth.lower().startswith("bearer ") else ""
        if token:
            try:
                payload = await _verify_supabase_token(token)
            except HTTPException:
                payload = None
            email = (payload or {}).get("email")
            if (
                isinstance(email, str)
                and email.strip().lower() == OWNER_EMAIL.lower()
            ):
                return
    if not OWNER_KEY and not jwt_path:
        raise HTTPException(status_code=503, detail=_OWNER_503_MSG)
    raise HTTPException(status_code=401, detail=_OWNER_401_MSG)


_PRO_403_MSG = (
    "Pro code required or invalid: the Pro features need a valid Pro "
    "code issued by the owner. (Pro code လိုအပ်ပါတယ် — ပိုင်ရှင် ပေးတဲ့ "
    "code မှန်မှ သုံးလို့ရပါတယ်)"
)

_PRO_SELECT = "code,label,bound_user_id,bound_email,active,created_at,last_used_at"


def _sb_headers() -> dict:
    return {
        "apikey": SUPABASE_SERVICE_KEY,
        "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
        "Content-Type": "application/json",
    }


def _pro_store_configured() -> bool:
    return bool(SUPABASE_URL and SUPABASE_SERVICE_KEY)


async def _sb_codes_get(code: str):
    """Fetch one pro_codes row by code, or None. 502 if the store is
    unreachable (fails closed — a store outage never opens Pro)."""
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(15.0), trust_env=False) as client:
            resp = await client.get(
                f"{SUPABASE_URL}/rest/v1/pro_codes",
                params={"code": f"eq.{code}", "select": _PRO_SELECT},
                headers=_sb_headers(),
            )
    except Exception:
        raise HTTPException(status_code=502, detail="Pro code service unavailable")
    if resp.status_code != 200:
        raise HTTPException(status_code=502, detail="Pro code service unavailable")
    rows = resp.json()
    if isinstance(rows, list) and rows and isinstance(rows[0], dict):
        return rows[0]
    return None


async def _sb_codes_patch(code: str, body: dict) -> list:
    """PATCH one pro_codes row; returns the updated rows (may be [])."""
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(15.0), trust_env=False) as client:
            resp = await client.patch(
                f"{SUPABASE_URL}/rest/v1/pro_codes",
                params={"code": f"eq.{code}"},
                json=body,
                headers={**_sb_headers(), "Prefer": "return=representation"},
            )
    except Exception:
        raise HTTPException(status_code=502, detail="Pro code service unavailable")
    if resp.status_code != 200:
        raise HTTPException(status_code=502, detail="Pro code service unavailable")
    rows = resp.json()
    return rows if isinstance(rows, list) else []


async def _sb_codes_insert(code: str, label: str, bound_email=None) -> dict:
    body: dict = {"code": code, "label": label}
    if bound_email:
        body["bound_email"] = bound_email
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(15.0), trust_env=False) as client:
            resp = await client.post(
                f"{SUPABASE_URL}/rest/v1/pro_codes",
                json=body,
                headers={**_sb_headers(), "Prefer": "return=representation"},
            )
    except Exception:
        raise HTTPException(status_code=502, detail="Pro code service unavailable")
    if resp.status_code not in (200, 201):
        raise HTTPException(status_code=502, detail="Pro code service unavailable")
    rows = resp.json()
    if isinstance(rows, list) and rows and isinstance(rows[0], dict):
        return rows[0]
    return {"code": code, "label": label, "created_at": None,
            "bound_email": bound_email}


async def _sb_codes_list() -> list:
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(15.0), trust_env=False) as client:
            resp = await client.get(
                f"{SUPABASE_URL}/rest/v1/pro_codes",
                params={"select": _PRO_SELECT, "order": "created_at.desc"},
                headers=_sb_headers(),
            )
    except Exception:
        raise HTTPException(status_code=502, detail="Pro code service unavailable")
    if resp.status_code != 200:
        raise HTTPException(status_code=502, detail="Pro code service unavailable")
    rows = resp.json()
    return rows if isinstance(rows, list) else []


async def _require_pro_code(request: Request, user_id: str,
                            user_email: str = None):
    """Validate the caller's Pro code (X-Pro-Code header).

    Returns (code, needs_bind). Raises 503 when the pro store is not
    configured, 403 when the code is missing, malformed, unknown,
    revoked, bound to a different user id, or issued for a different
    login email (bound_email, compared case-insensitively) than the
    caller's verified JWT email.

    Owner bypass: when the caller's verified JWT email equals
    OWNER_EMAIL (case-insensitive), the check passes immediately with
    (None, False) — no code is required, a presented code is ignored,
    and no code row is touched. Disabled when OWNER_EMAIL is unset.
    """
    if OWNER_EMAIL and user_email \
            and user_email.strip().lower() == OWNER_EMAIL.lower():
        return None, False
    if not _pro_store_configured():
        raise HTTPException(
            status_code=503,
            detail="Pro access is not configured on this server. "
                   "(Pro မဖွင့်ထားပါ)",
        )
    raw = (request.headers.get("X-Pro-Code") or "").strip()
    if not raw:
        raise HTTPException(status_code=403, detail=_PRO_403_MSG)
    try:
        code = str(uuid.UUID(raw))
    except Exception:
        raise HTTPException(status_code=403, detail=_PRO_403_MSG)
    row = await _sb_codes_get(code)
    if row is None or not row.get("active"):
        raise HTTPException(status_code=403, detail=_PRO_403_MSG)
    bound = row.get("bound_user_id")
    if bound and str(bound) != user_id:
        raise HTTPException(status_code=403, detail=_PRO_403_MSG)
    bound_email = row.get("bound_email")
    if bound_email and str(bound_email).strip().lower() != (user_email or ""):
        raise HTTPException(status_code=403, detail=_PRO_403_MSG)
    return code, bound is None


async def _pro_code_touch(code: str, user_id: str, bind: bool):
    """Best-effort bookkeeping after a successful Pro call: stamp
    last_used_at, and bind the code to this user on first use. A stamp
    failure never fails the request, and the code value is not logged.
    A None code (owner bypass) touches no row at all.
    """
    if code is None:
        return
    body: dict = {"last_used_at": datetime.now(timezone.utc).isoformat()}
    if bind:
        body["bound_user_id"] = user_id
    try:
        await _sb_codes_patch(code, body)
    except Exception:
        print("WARN: pro code stamp failed", flush=True)


# --- Per-user usage view + TTS kill-switch (phase 9) ---
# The admin app lists every registered user with their TTS usage
# (aggregated from usage_log) and can disable individual engines per
# user (user_controls.disabled_modes). The public routes consult the
# same table as a SOFT control: any store problem means "enabled".

TTS_MODES = ("edge", "el", "gemini", "f5", "vox")

_MODE_DISABLED_MSG = (
    "This TTS engine is disabled by the admin. "
    "(ဒီ TTS ကို admin က ပိတ်ထားပါတယ်)"
)

_USER_SVC_502_MSG = "User service unavailable"


async def _sb_get_json(path: str, params: dict):
    """GET a Supabase URL (REST or Auth Admin) with the service key;
    returns the parsed JSON body. Any failure is a generic 502 — no
    upstream detail (and never any key material) is echoed."""
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(15.0), trust_env=False) as client:
            resp = await client.get(
                f"{SUPABASE_URL}{path}", params=params, headers=_sb_headers(),
            )
    except Exception:
        raise HTTPException(status_code=502, detail=_USER_SVC_502_MSG)
    if resp.status_code != 200:
        raise HTTPException(status_code=502, detail=_USER_SVC_502_MSG)
    try:
        return resp.json()
    except Exception:
        raise HTTPException(status_code=502, detail=_USER_SVC_502_MSG)


async def _sb_auth_users() -> list:
    """All registered users from the Supabase Auth Admin API."""
    data = await _sb_get_json("/auth/v1/admin/users", {"per_page": 1000})
    users = data.get("users") if isinstance(data, dict) else None
    if not isinstance(users, list):
        raise HTTPException(status_code=502, detail=_USER_SVC_502_MSG)
    return [u for u in users if isinstance(u, dict)]


async def _sb_usage_rows() -> list:
    rows = await _sb_get_json(
        "/rest/v1/usage_log",
        {"select": "user_id,mode,chars,duration_seconds,created_at",
         "limit": 20000},
    )
    if not isinstance(rows, list):
        raise HTTPException(status_code=502, detail=_USER_SVC_502_MSG)
    return [r for r in rows if isinstance(r, dict)]


async def _sb_controls_rows() -> list:
    rows = await _sb_get_json(
        "/rest/v1/user_controls", {"select": "user_id,disabled_modes"},
    )
    if not isinstance(rows, list):
        raise HTTPException(status_code=502, detail=_USER_SVC_502_MSG)
    return [r for r in rows if isinstance(r, dict)]


def _valid_modes(value) -> list:
    """The canonical subset of a stored/requested disabled_modes list."""
    if not isinstance(value, list):
        return []
    return [m for m in value if m in TTS_MODES]


async def _check_mode_enabled(user_id: str, mode: str):
    """Raise 403 when the admin disabled `mode` for this user.

    FAILS OPEN by design: an unconfigured store, a fetch error, a
    non-200 response, or a missing/malformed row all mean "enabled" —
    this soft control must never take the public routes down.
    """
    if not _pro_store_configured():
        return
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(10.0), trust_env=False) as client:
            resp = await client.get(
                f"{SUPABASE_URL}/rest/v1/user_controls",
                params={"user_id": f"eq.{user_id}", "select": "disabled_modes"},
                headers=_sb_headers(),
            )
    except Exception:
        return
    if resp.status_code != 200:
        return
    try:
        rows = resp.json()
    except Exception:
        return
    if isinstance(rows, list) and rows and isinstance(rows[0], dict):
        if mode in _valid_modes(rows[0].get("disabled_modes")):
            raise HTTPException(status_code=403, detail=_MODE_DISABLED_MSG)


class AdminCodeCreate(BaseModel):
    label: str = ""
    email: str = ""


def _require_pro_store():
    if not _pro_store_configured():
        raise HTTPException(
            status_code=503,
            detail="Pro codes are not configured on this server "
                   "(SUPABASE_URL / SUPABASE_SERVICE_KEY missing).",
        )


def _public_code_row(row: dict) -> dict:
    return {
        "code": row.get("code"),
        "label": row.get("label"),
        "bound_user_id": row.get("bound_user_id"),
        "bound_email": row.get("bound_email"),
        "active": bool(row.get("active")),
        "created_at": row.get("created_at"),
        "last_used_at": row.get("last_used_at"),
    }


@app.post("/api/admin/codes")
async def admin_codes_create(req: AdminCodeCreate, request: Request):
    """Issue a new Pro code (owner only).

    An optional email issues the code FOR that user's login gmail:
    it is stored (trimmed, lowercased) as bound_email, and becomes
    the label when no label was given. Without an email the code is
    unbound and binds to its first user as before.
    """
    await _require_admin(request)
    _require_pro_store()
    code = str(uuid.uuid4())
    email = (req.email or "").strip().lower()
    label = (req.label or "").strip()
    if email and not label:
        label = email
    row = await _sb_codes_insert(code, label, email or None)
    return {
        "code": row.get("code", code),
        "label": row.get("label", label),
        "bound_email": row.get("bound_email", email or None),
        "created_at": row.get("created_at"),
    }


@app.get("/api/admin/codes")
async def admin_codes_list(request: Request):
    """List all Pro codes with their binding/status (owner only)."""
    await _require_admin(request)
    _require_pro_store()
    rows = await _sb_codes_list()
    return {"codes": [_public_code_row(r) for r in rows if isinstance(r, dict)]}


@app.post("/api/admin/codes/{code}/revoke")
async def admin_codes_revoke(code: str, request: Request):
    """Revoke a Pro code (owner only)."""
    await _require_admin(request)
    _require_pro_store()
    rows = await _sb_codes_patch(code, {"active": False})
    if not rows:
        raise HTTPException(status_code=404, detail="Code not found")
    return {"code": code, "active": False}


@app.post("/api/admin/codes/{code}/unbind")
async def admin_codes_unbind(code: str, request: Request):
    """Clear a Pro code's bound user so it can be reassigned (owner only)."""
    await _require_admin(request)
    _require_pro_store()
    rows = await _sb_codes_patch(code, {"bound_user_id": None})
    if not rows:
        raise HTTPException(status_code=404, detail="Code not found")
    return {"code": code, "bound_user_id": None}


def _blank_usage() -> dict:
    return {
        "totals": {"calls": 0, "chars": 0, "seconds": 0.0},
        "by_mode": {m: {"calls": 0, "chars": 0, "seconds": 0.0}
                    for m in TTS_MODES},
        "last_used_at": None,
    }


@app.get("/api/admin/users")
async def admin_users_list(request: Request):
    """Every registered user with per-engine TTS usage totals and
    their disabled engines (owner only). Aggregation happens here:
    usage_log rows are summed per user per mode in Python."""
    await _require_admin(request)
    _require_pro_store()
    auth_users = await _sb_auth_users()
    usage_rows = await _sb_usage_rows()
    controls_rows = await _sb_controls_rows()

    disabled_by_user = {}
    for row in controls_rows:
        uid = row.get("user_id")
        if uid:
            disabled_by_user[str(uid)] = _valid_modes(row.get("disabled_modes"))

    usage_by_user: dict = {}
    for row in usage_rows:
        uid = row.get("user_id")
        if not uid:
            continue
        uid = str(uid)
        entry = usage_by_user.setdefault(uid, _blank_usage())
        mode = row.get("mode")
        bucket = entry["by_mode"].get(mode) if isinstance(mode, str) else None
        try:
            chars = int(row.get("chars") or 0)
        except Exception:
            chars = 0
        try:
            seconds = float(row.get("duration_seconds") or 0)
        except Exception:
            seconds = 0.0
        entry["totals"]["calls"] += 1
        entry["totals"]["chars"] += chars
        entry["totals"]["seconds"] += seconds
        if bucket is not None:
            bucket["calls"] += 1
            bucket["chars"] += chars
            bucket["seconds"] += seconds
        created = row.get("created_at")
        if isinstance(created, str) and (
            entry["last_used_at"] is None or created > entry["last_used_at"]
        ):
            entry["last_used_at"] = created

    users = []
    seen = set()
    for u in auth_users:
        uid = u.get("id")
        if not uid:
            continue
        uid = str(uid)
        seen.add(uid)
        entry = usage_by_user.get(uid) or _blank_usage()
        users.append({
            "user_id": uid,
            "email": u.get("email"),
            "created_at": u.get("created_at"),
            "last_sign_in_at": u.get("last_sign_in_at"),
            "last_used_at": entry["last_used_at"],
            "totals": entry["totals"],
            "by_mode": entry["by_mode"],
            "disabled_modes": disabled_by_user.get(uid, []),
        })
    # Usage recorded for an id that is no longer (or not yet) in the
    # auth list still shows up, with no email on file.
    for uid in sorted(set(usage_by_user) - seen):
        entry = usage_by_user[uid]
        users.append({
            "user_id": uid,
            "email": None,
            "created_at": None,
            "last_sign_in_at": None,
            "last_used_at": entry["last_used_at"],
            "totals": entry["totals"],
            "by_mode": entry["by_mode"],
            "disabled_modes": disabled_by_user.get(uid, []),
        })
    return {"users": users}


@app.post("/api/admin/users/{user_id}/controls")
async def admin_users_set_controls(user_id: str, request: Request):
    """Set one user's disabled TTS engines (owner only).

    Body: {"disabled_modes": [...]} — every item must be one of the
    canonical mode strings, else 400. Upserts the user_controls row.
    """
    await _require_admin(request)
    _require_pro_store()
    try:
        body = await request.json()
    except Exception:
        body = None
    modes = body.get("disabled_modes") if isinstance(body, dict) else None
    if not isinstance(modes, list) or any(m not in TTS_MODES for m in modes):
        raise HTTPException(
            status_code=400,
            detail="disabled_modes must be a list containing only: "
                   + ", ".join(TTS_MODES),
        )
    modes = list(dict.fromkeys(modes))

    email = None
    try:
        for u in await _sb_auth_users():
            if str(u.get("id")) == user_id:
                email = u.get("email")
                break
    except Exception:
        email = None  # the email stamp is best-effort; the upsert matters

    row = {
        "user_id": user_id,
        "email": email,
        "disabled_modes": modes,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(15.0), trust_env=False) as client:
            resp = await client.post(
                f"{SUPABASE_URL}/rest/v1/user_controls",
                json=row,
                headers={**_sb_headers(),
                         "Prefer": "resolution=merge-duplicates,return=representation"},
            )
    except Exception:
        raise HTTPException(status_code=502, detail=_USER_SVC_502_MSG)
    if resp.status_code not in (200, 201):
        raise HTTPException(status_code=502, detail=_USER_SVC_502_MSG)
    return {"ok": True, "user_id": user_id, "disabled_modes": modes}


# --- Feature settings + VoxCPM public mode (phase 12) ---
# One app_settings row (key='features', jsonb value) holds the
# admin-controlled feature flags. Today that is only the VoxCPM
# session mode: vox_enabled (master switch) and vox_url (the live
# Kaggle tunnel base URL — it changes on every session restart, so
# the admin pastes the current one here instead of redeploying).
# Reads FAIL CLOSED: any store problem means "off, no session", so a
# Supabase outage can never open the mode by accident.
#
# The public VoxCPM routes proxy to the Kaggle notebook backend
# (voxcpm2_kaggle_backend_2GPU.ipynb): POST {vox_url}/tts with a
# multipart form — fields text (required), ref_text (transcript of
# the reference clip, "" for reference-only cloning) and ref_audio
# (the reference audio file) — and it answers audio/wav. The owner's
# own login (OWNER_EMAIL) may use the mode even while the master
# switch is off; a per-user admin disable ("vox" in user_controls)
# still blocks any non-owner it is set on.

_FEATURES_DEFAULT = {"vox_enabled": False, "vox_url": ""}

_VOX_OFF_MSG = (
    "VoxCPM is turned off by the admin. "
    "(VoxCPM ကို admin က ပိတ်ထားပါတယ်)"
)

_VOX_NO_SESSION_MSG = (
    "VoxCPM session is not available right now. "
    "(VoxCPM session မရှိသေးပါ)"
)

_VOX_UPSTREAM_MSG = (
    "VoxCPM session error — try again later. "
    "(VoxCPM session အမှား ဖြစ်ပါတယ် — နောက်မှ ပြန် စမ်းပါ)"
)

_FEATURES_SVC_502_MSG = "Settings service unavailable"


def _features_from_value(value) -> dict:
    """Normalize a stored app_settings 'features' value."""
    if not isinstance(value, dict):
        return dict(_FEATURES_DEFAULT)
    url = value.get("vox_url")
    return {
        "vox_enabled": bool(value.get("vox_enabled")),
        "vox_url": url if isinstance(url, str) else "",
    }


async def _get_features() -> dict:
    """Current feature settings. Fails CLOSED to all-off on any
    error (unconfigured store, fetch failure, non-200, malformed
    body, missing row)."""
    if not _pro_store_configured():
        return dict(_FEATURES_DEFAULT)
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(10.0), trust_env=False) as client:
            resp = await client.get(
                f"{SUPABASE_URL}/rest/v1/app_settings",
                params={"key": "eq.features", "select": "value"},
                headers=_sb_headers(),
            )
    except Exception:
        return dict(_FEATURES_DEFAULT)
    if resp.status_code != 200:
        return dict(_FEATURES_DEFAULT)
    try:
        rows = resp.json()
    except Exception:
        return dict(_FEATURES_DEFAULT)
    if isinstance(rows, list) and rows and isinstance(rows[0], dict):
        return _features_from_value(rows[0].get("value"))
    return dict(_FEATURES_DEFAULT)


async def _set_features(patch: dict) -> dict:
    """Merge `patch` into the stored features row (upsert) and
    return the stored dict."""
    merged = _features_from_value({**await _get_features(), **patch})
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(15.0), trust_env=False) as client:
            resp = await client.post(
                f"{SUPABASE_URL}/rest/v1/app_settings",
                json={"key": "features", "value": merged},
                headers={**_sb_headers(),
                         "Prefer": "resolution=merge-duplicates,return=representation"},
            )
    except Exception:
        raise HTTPException(status_code=502, detail=_FEATURES_SVC_502_MSG)
    if resp.status_code not in (200, 201):
        raise HTTPException(status_code=502, detail=_FEATURES_SVC_502_MSG)
    try:
        rows = resp.json()
    except Exception:
        rows = None
    if isinstance(rows, list) and rows and isinstance(rows[0], dict):
        return _features_from_value(rows[0].get("value"))
    return merged


@app.get("/api/admin/settings")
async def admin_settings_get(request: Request):
    """Current feature settings (owner only)."""
    await _require_admin(request)
    _require_pro_store()
    return await _get_features()


@app.post("/api/admin/settings")
async def admin_settings_post(request: Request):
    """Patch feature settings (owner only). Body may include
    vox_enabled (bool) and/or vox_url (string: "" or an https://
    URL; trimmed before storing). Returns the stored dict."""
    await _require_admin(request)
    _require_pro_store()
    try:
        body = await request.json()
    except Exception:
        body = None
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="JSON body required")
    patch: dict = {}
    if "vox_enabled" in body:
        if not isinstance(body["vox_enabled"], bool):
            raise HTTPException(
                status_code=400, detail="vox_enabled must be a boolean")
        patch["vox_enabled"] = body["vox_enabled"]
    if "vox_url" in body:
        url = body["vox_url"]
        if not isinstance(url, str):
            raise HTTPException(
                status_code=400, detail="vox_url must be a string")
        url = url.strip()
        if url and not url.startswith("https://"):
            raise HTTPException(
                status_code=400,
                detail="vox_url must be empty or start with https://")
        patch["vox_url"] = url
    return await _set_features(patch)


@app.get("/api/pub/vox/status")
async def pub_vox_status(request: Request):
    """Whether the VoxCPM mode is on for this caller (master switch
    or the owner's login) and whether a session URL is set."""
    user_id, user_email = await _require_pub_identity(request)
    features = await _get_features()
    return {
        "enabled": bool(features["vox_enabled"]
                        or _is_owner_email(user_email)),
        "available": bool(features["vox_url"]),
    }


@app.post("/api/pub/vox/tts")
async def pub_vox_tts(
    request: Request,
    text: str = Form(...),
    ref_text: str = Form(""),
    ref_audio: UploadFile = File(...),
):
    """VoxCPM voice-clone TTS: forwards the multipart form to the
    Kaggle backend's POST /tts (fields text / ref_text / ref_audio,
    exactly the notebook server's contract) and streams its WAV
    back. Gated by the per-user disable list, the admin master
    switch (owner's login excepted), and a live session URL."""
    user_id, user_email = await _require_pub_identity(request)
    await _check_mode_enabled(user_id, "vox")
    features = await _get_features()
    if not (features["vox_enabled"] or _is_owner_email(user_email)):
        raise HTTPException(status_code=403, detail=_VOX_OFF_MSG)
    base = features["vox_url"]
    if not base:
        raise HTTPException(status_code=503, detail=_VOX_NO_SESSION_MSG)
    if not text or not text.strip():
        raise HTTPException(status_code=400, detail="Text is required")
    audio_bytes = await _read_upload_capped(ref_audio)
    if not audio_bytes:
        raise HTTPException(
            status_code=400, detail="Reference audio file is empty")
    files = {
        "ref_audio": (
            ref_audio.filename or "ref.wav",
            audio_bytes,
            ref_audio.content_type or "audio/wav",
        ),
    }
    data = {"text": text.strip(), "ref_text": ref_text or ""}
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(600.0), trust_env=False) as client:
            resp = await client.post(
                base.rstrip("/") + "/tts", data=data, files=files)
    except Exception:
        raise HTTPException(status_code=502, detail=_VOX_UPSTREAM_MSG)
    if resp.status_code != 200:
        raise HTTPException(status_code=502, detail=_VOX_UPSTREAM_MSG)
    return Response(
        content=resp.content,
        media_type=resp.headers.get("content-type") or "audio/wav",
    )
