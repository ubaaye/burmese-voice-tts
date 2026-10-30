import asyncio
import json
import os
import tempfile
import uuid

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
    return {"ok": True, "service": "burmese-tts", "clone": bool(HF_TOKEN)}


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


async def run_clone(text: str, ref_text: str, speed: float,
                    audio_bytes: bytes, filename: str, mime: str) -> bytes:
    """Zero-shot Burmese voice cloning via the F5-Myanmar-TTS Space.

    Returns the generated WAV bytes; raises HTTPException on failure.
    """
    if not HF_TOKEN:
        raise HTTPException(status_code=500, detail="Clone is not configured (HF_TOKEN missing)")

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

            # 2. Join the generation queue (full FileData, as gradio_client sends)
            file_data = {
                "path": up_path,
                "orig_name": filename,
                "mime_type": mime,
                "size": len(audio_bytes),
                "is_stream": False,
                "meta": {"_type": "gradio.FileData"},
            }
            session = uuid.uuid4().hex[:12]
            join = await client.post(
                HF_SPACE + "/gradio_api/queue/join",
                json={
                    "data": [text, file_data, ref_text, speed],
                    "session_hash": session,
                    "fn_index": 0,
                },
                headers=headers,
            )
            join.raise_for_status()

            # 3. Stream events until completion
            out_path = None
            fail_info = ""
            async with client.stream(
                "GET",
                HF_SPACE + f"/gradio_api/queue/data?session_hash={session}",
                headers=headers,
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    try:
                        evt = json.loads(line[5:].strip())
                    except Exception:
                        continue
                    print("CLONE EVT:", json.dumps(evt)[:600], flush=True)
                    if evt.get("msg") == "process_completed":
                        if evt.get("success"):
                            data = evt.get("output", {}).get("data", [])
                            if data:
                                first = data[0]
                                if isinstance(first, str):
                                    out_path = first
                                elif isinstance(first, dict):
                                    out_path = first.get("path") or first.get("url")
                        else:
                            fail_info = str(
                                evt.get("error") or evt.get("output") or evt
                            )[:400]
                        break
                    if evt.get("msg") == "process_failed":
                        fail_info = str(
                            evt.get("error") or evt.get("output") or evt
                        )[:400]
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
