import asyncio
import base64
import hmac
import json
import os
import re
import struct
import tempfile
import time
import uuid

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
# A public deployment sets SUPABASE_URL and deliberately leaves HF_TOKEN /
# EL_API_KEY / GEMINI_API_KEY UNSET, so the owner's keys can never serve
# strangers. Every /api/pub/* route requires a valid Supabase access token
# (Authorization: Bearer <jwt>, verified against the project JWKS below),
# and the paid/quota routes additionally require the VISITOR's own provider
# key per request (X-EL-Key / X-Gemini-Key / X-HF-Token headers). Visitor
# keys are used for that request/job only and are never stored or logged.
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
    # Optional tone/style instruction (e.g. "Say cheerfully and slowly").
    # When non-empty it is prepended to the text sent upstream as
    # "<style>: <text>", the model's documented style-prompt pattern.
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
        if style:
            # Prefix the direction onto the FIRST segment's text so the
            # direction itself is spoken by a speaker: a separate
            # leading part without speech_metadata would break the
            # multi-speaker rule that every part names its speaker.
            parts[0]["text"] = f"{style}: {parts[0]['text']}"
    else:
        speech_config = {
            "voiceConfig": {
                "prebuiltVoiceConfig": {"voiceName": req.voice},
            },
        }
        if style:
            text = f"{style}: {text}"
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

    return Response(content=_pcm_to_wav(pcm, sample_rate), media_type="audio/wav")


# --- Public edition routes (/api/pub/*) ---
# Every route here requires (1) a valid Supabase access token and
# (2) for the paid/quota engines, the visitor's OWN provider key in a
# header. The server's env keys (HF_TOKEN / EL_API_KEY / GEMINI_API_KEY)
# are never used by these routes and there is no fallback to them.

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
    """Public ElevenLabs TTS with the VISITOR's own key (X-EL-Key)."""
    user_id, user_email = await _require_pub_identity(request)
    await _check_mode_enabled(user_id, "el")
    api_key = _require_byok_key(request, "X-EL-Key", "ElevenLabs")
    code, needs_bind = await _require_pro_code(request, user_id, user_email)
    resp = await _el_tts_core(req, api_key)
    await _pro_code_touch(code, user_id, needs_bind)
    return resp


@app.get("/api/pub/el/voices")
async def pub_el_voices(request: Request):
    """List the voices on the VISITOR's own ElevenLabs account."""
    user_id, user_email = await _require_pub_identity(request)
    await _check_mode_enabled(user_id, "el")
    api_key = _require_byok_key(request, "X-EL-Key", "ElevenLabs")
    code, needs_bind = await _require_pro_code(request, user_id, user_email)
    resp = await _el_voices_core(api_key, own_only=True)
    await _pro_code_touch(code, user_id, needs_bind)
    return resp


@app.post("/api/pub/el/voices")
async def pub_el_voice_add(request: Request, name: str = Form(...), sample: UploadFile = File(...)):
    """Create a cloned voice on the VISITOR's own ElevenLabs account
    (X-EL-Key): the same upstream call as the private route, with the
    visitor's key. The server's EL_API_KEY is never used here."""
    user_id, user_email = await _require_pub_identity(request)
    await _check_mode_enabled(user_id, "el")
    api_key = _require_byok_key(request, "X-EL-Key", "ElevenLabs")
    code, needs_bind = await _require_pro_code(request, user_id, user_email)
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
    """Delete a voice from the VISITOR's own ElevenLabs account
    (X-EL-Key). Only the visitor's own cloned/professional voices can
    be deleted: the voice's category is checked with the visitor's key
    BEFORE the upstream delete is called, so premade voices get a
    clean 400 here instead of an upstream delete error."""
    user_id, user_email = await _require_pub_identity(request)
    await _check_mode_enabled(user_id, "el")
    api_key = _require_byok_key(request, "X-EL-Key", "ElevenLabs")
    code, needs_bind = await _require_pro_code(request, user_id, user_email)
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
    """Public Gemini TTS (3.8 contract) with the VISITOR's own key."""
    user_id, user_email = await _require_pub_identity(request)
    await _check_mode_enabled(user_id, "gemini")
    api_key = _require_byok_key(request, "X-Gemini-Key", "Gemini")
    code, needs_bind = await _require_pro_code(request, user_id, user_email)
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
    """Public clone job using the VISITOR's own HF token (X-HF-Token).

    The token is handed to the background job for its upstream calls
    only; it is not written into the job entry or anywhere else.
    """
    user_id, user_email = await _require_pub_identity(request)
    await _check_mode_enabled(user_id, "f5")
    token = _require_byok_key(request, "X-HF-Token", "Hugging Face")
    code, needs_bind = await _require_pro_code(request, user_id, user_email)
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
    code, needs_bind = await _require_pro_code(request, user_id, user_email)
    job = _pub_job_or_404(job_id, user_id)
    await _pro_code_touch(code, user_id, needs_bind)
    return _job_status_payload(job_id, job)


@app.get("/api/pub/clone/jobs/{job_id}/audio")
async def pub_clone_job_audio(job_id: str, request: Request):
    user_id, user_email = await _require_pub_identity(request)
    code, needs_bind = await _require_pro_code(request, user_id, user_email)
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
# configured those routes return 503 (locked) rather than open. Pro
# routes additionally require a valid Pro code (below).

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

TTS_MODES = ("edge", "el", "gemini", "f5")

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
