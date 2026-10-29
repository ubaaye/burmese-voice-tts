import os
import json
import tempfile
import uuid

import edge_tts
import httpx
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
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


@app.post("/api/clone")
async def clone_speech(
    text: str = Form(...),
    ref_text: str = Form(...),
    speed: float = Form(1.0),
    ref_audio: UploadFile = File(...),
):
    """Zero-shot Burmese voice cloning via the F5-Myanmar-TTS Space."""
    if not HF_TOKEN:
        raise HTTPException(status_code=500, detail="Clone is not configured (HF_TOKEN missing)")
    if not text.strip():
        raise HTTPException(status_code=400, detail="Text is required")
    if not ref_text.strip():
        raise HTTPException(status_code=400, detail="Reference transcript is required")

    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    audio_bytes = await ref_audio.read()
    filename = ref_audio.filename or "ref.wav"
    mime = ref_audio.content_type or "audio/wav"

    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(660.0)) as client:
            # 1. Upload the reference audio to the Space
            up = await client.post(
                HF_SPACE + "/gradio_api/upload",
                files={"files": (filename, audio_bytes, mime)},
                headers=headers,
            )
            up.raise_for_status()
            up_path = up.json()[0]

            # 2. Join the generation queue
            session = uuid.uuid4().hex[:12]
            join = await client.post(
                HF_SPACE + "/gradio_api/queue/join",
                json={
                    "data": [text.strip(), {"path": up_path}, ref_text.strip(), speed],
                    "session_hash": session,
                    "fn_index": 0,
                },
                headers=headers,
            )
            join.raise_for_status()

            # 3. Stream events until completion
            out_path = None
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
                    if evt.get("msg") == "process_completed":
                        if evt.get("success"):
                            data = evt.get("output", {}).get("data", [])
                            if data:
                                out_path = data[0].get("path") or data[0].get("url")
                        break
                    if evt.get("msg") == "process_failed":
                        break

            if not out_path:
                raise HTTPException(status_code=502, detail="Clone generation failed on the model server")

            # 4. Download the generated audio
            dl_url = out_path if out_path.startswith("http") else HF_SPACE + "/gradio_api/file=" + out_path
            got = await client.get(dl_url, headers=headers)
            got.raise_for_status()
            return Response(content=got.content, media_type="audio/wav")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Clone service error: {e}")
