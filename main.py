import asyncio
import base64
import json
import os
import re
import struct
import tempfile
import uuid

from typing import Any

import edge_tts
import httpx
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

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
    return {"ok": True, "service": "burmese-tts", "clone": bool(HF_TOKEN), "el": bool(EL_API_KEY), "gemini": bool(GEMINI_API_KEY), "v": 3}


@app.post("/api/synthesize")
async def synthesize_speech(req: TTSRequest):
    if not req.text or not req.text.strip():
        raise HTTPException(status_code=400, detail="Text is required")

    output_path = tempfile.mktemp(suffix=".mp3")

    try:
        communicate = edge_tts.Communicate(
            text=req.text.strip(),
            voice=req.voice,
            rate=req.rate,
            pitch=req.pitch
        )
        await communicate.save(output_path)
        return FileResponse(output_path, media_type="audio/mpeg")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


async def _clone_attempt(text: str, ref_text: str, speed: float,
                         audio_bytes: bytes, filename: str, mime: str) -> bytes:
    """One clone attempt against the Space; returns WAV bytes."""
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}

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
                    audio_bytes: bytes, filename: str, mime: str) -> bytes:
    """Zero-shot Burmese voice cloning via the F5-Myanmar-TTS Space.

    The Space's GPU backend fails intermittently with an empty error,
    so up to 4 attempts are made before giving up.
    """
    if not HF_TOKEN:
        raise HTTPException(status_code=500, detail="Clone is not configured (HF_TOKEN missing)")
    last_err = None
    for attempt in range(1, 5):
        try:
            return await _clone_attempt(text, ref_text, speed, audio_bytes, filename, mime)
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
    text: str = Form(...),
    ref_text: str = Form(...),
    speed: float = Form(1.0),
    ref_audio: UploadFile = File(...),
):
    """Synchronous clone (kept for compatibility)."""
    _validate_clone(text, ref_text)
    audio_bytes = await ref_audio.read()
    content = await run_clone(
        text.strip(), ref_text.strip(), speed,
        audio_bytes, ref_audio.filename or "ref.wav",
        ref_audio.content_type or "audio/wav",
    )
    return Response(content=content, media_type="audio/wav")


# --- Async clone jobs: create, poll status, fetch audio ---
jobs = {}


async def _job_work(job_id: str, text: str, ref_text: str, speed: float,
                    audio_bytes: bytes, filename: str, mime: str):
    try:
        jobs[job_id]["status"] = "running"
        content = await run_clone(text, ref_text, speed, audio_bytes, filename, mime)
        jobs[job_id]["status"] = "done"
        jobs[job_id]["audio"] = content
    except HTTPException as e:
        jobs[job_id]["status"] = "error"
        jobs[job_id]["error"] = str(e.detail)
    except Exception as e:
        jobs[job_id]["status"] = "error"
        jobs[job_id]["error"] = f"Clone service error: {e}"


@app.post("/api/clone/jobs")
async def clone_job_create(
    text: str = Form(...),
    ref_text: str = Form(...),
    speed: float = Form(1.0),
    ref_audio: UploadFile = File(...),
):
    _validate_clone(text, ref_text)
    audio_bytes = await ref_audio.read()
    job_id = uuid.uuid4().hex[:12]
    jobs[job_id] = {"status": "queued"}
    asyncio.create_task(_job_work(
        job_id, text.strip(), ref_text.strip(), speed,
        audio_bytes, ref_audio.filename or "ref.wav",
        ref_audio.content_type or "audio/wav",
    ))
    return {"job_id": job_id, "status": "queued"}


@app.get("/api/clone/jobs/{job_id}")
async def clone_job_status(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    out = {"job_id": job_id, "status": job["status"]}
    if job["status"] == "error":
        out["error"] = job.get("error", "")
    return out


@app.get("/api/clone/jobs/{job_id}/audio")
async def clone_job_audio(job_id: str):
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
async def el_status():
    """Report whether ElevenLabs is configured, plus usage when available."""
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


@app.post("/api/el/tts")
async def el_tts(req: ELTTSRequest):
    """Generate speech via ElevenLabs using the server-side key."""
    if not EL_API_KEY:
        raise HTTPException(
            status_code=503,
            detail="ElevenLabs is not configured on the server (EL_API_KEY missing)",
        )
    if not req.text or not req.text.strip():
        raise HTTPException(status_code=400, detail="Text is required")

    body: dict = {"text": req.text.strip(), "model_id": req.model_id}
    if req.speed is not None:
        body["voice_settings"] = {"speed": req.speed}

    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(300.0)) as client:
            resp = await client.post(
                f"{EL_BASE}/v1/text-to-speech/{req.voice_id}",
                json=body,
                headers={
                    "xi-api-key": EL_API_KEY,
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


@app.get("/api/el/voices")
async def el_voices_list():
    """List the ElevenLabs voices available to the server-side key."""
    if not EL_API_KEY:
        raise HTTPException(
            status_code=503,
            detail="ElevenLabs is not configured on the server (EL_API_KEY missing)",
        )
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(30.0)) as client:
            resp = await client.get(
                EL_BASE + "/v1/voices",
                headers={"xi-api-key": EL_API_KEY},
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
    voices = [
        {
            "voice_id": v.get("voice_id"),
            "name": v.get("name"),
            "category": v.get("category"),
        }
        for v in data.get("voices", [])
        if isinstance(v, dict)
    ]
    return {"voices": voices}


@app.post("/api/el/voices")
async def el_voices_add(name: str = Form(...), sample: UploadFile = File(...)):
    """Create a new ElevenLabs cloned voice from an uploaded audio sample."""
    if not EL_API_KEY:
        raise HTTPException(
            status_code=503,
            detail="ElevenLabs is not configured on the server (EL_API_KEY missing)",
        )
    if not name or not name.strip():
        raise HTTPException(status_code=400, detail="Voice name is required")

    sample_bytes = await sample.read()
    if not sample_bytes:
        raise HTTPException(status_code=400, detail="Sample audio file is empty")

    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(300.0)) as client:
            resp = await client.post(
                EL_BASE + "/v1/voices/add",
                data={"name": name.strip()},
                files={
                    "files": (
                        sample.filename or "sample.mp3",
                        sample_bytes,
                        sample.content_type or "audio/mpeg",
                    )
                },
                headers={"xi-api-key": EL_API_KEY},
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
    return {"voice_id": data.get("voice_id"), "name": name.strip()}


@app.delete("/api/el/voices/{voice_id}")
async def el_voices_delete(voice_id: str):
    """Delete an ElevenLabs voice using the server-side key."""
    if not EL_API_KEY:
        raise HTTPException(
            status_code=503,
            detail="ElevenLabs is not configured on the server (EL_API_KEY missing)",
        )
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(30.0)) as client:
            resp = await client.delete(
                f"{EL_BASE}/v1/voices/{voice_id}",
                headers={"xi-api-key": EL_API_KEY},
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
async def gemini_status():
    """Report whether Gemini TTS is configured, and which model is in use."""
    if not GEMINI_API_KEY:
        return {"configured": False}
    return {"configured": True, "model": GEMINI_TTS_MODEL}


@app.post("/api/gemini/tts")
async def gemini_tts(req: GeminiTTSRequest):
    """Generate speech via Gemini TTS using the server-side key.

    Gemini returns base64-encoded raw PCM (L16); it is wrapped in a WAV
    container before being returned as audio/wav.
    """
    if not GEMINI_API_KEY:
        raise HTTPException(
            status_code=503,
            detail="Gemini is not configured on the server (GEMINI_API_KEY missing)",
        )
    if not req.text or not req.text.strip():
        raise HTTPException(status_code=400, detail="Text is required")

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
                    "x-goog-api-key": GEMINI_API_KEY,
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
