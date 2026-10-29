import os
import tempfile
import edge_tts
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
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
    return {"ok": True, "service": "burmese-tts"}


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
