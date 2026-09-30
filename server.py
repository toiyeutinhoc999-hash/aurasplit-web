from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
import subprocess
from pathlib import Path
import librosa
import numpy as np

app = FastAPI(title="AuraSplit AI Web API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
OUTPUT_DIR = "separated"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Giao diện trang chủ khi truy cập link web
@app.get("/", response_class=HTMLResponse)
async def read_index():
    if os.path.exists("index.html"):
        with open("index.html", "r", encoding="utf-8") as f:
            return f.read()
    return "<h3>AuraSplit Web Server đang hoạt động! (Không tìm thấy file index.html)</h3>"

@app.post("/api/split")
async def split_audio(file: UploadFile = File(...), mode: str = Form("vocal_beat")):
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    try:
        y, sr = librosa.load(file_path, duration=60)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        bpm = float(np.atleast_1d(tempo)[0])

        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        key_idx = np.argmax(np.mean(chroma, axis=1))
        key_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        musical_key = key_names[key_idx]
    except Exception as e:
        bpm = 0.0
        musical_key = "Unknown"

    cmd = ["demucs", "-n", "htdemucs_ft"]
    if mode == "vocal_beat":
        cmd.append("--two-stems=vocals")
        
    cmd.append(file_path)
    
    process = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    if process.returncode != 0:
        return JSONResponse(status_code=500, content={"error": "Quá trình xử lý AI thất bại"})
    
    song_name = Path(file.filename).stem
    result_folder = os.path.join(OUTPUT_DIR, "htdemucs_ft", song_name)
    
    return {
        "status": "success",
        "bpm": round(bpm, 1),
        "key": musical_key,
        "folder_path": result_folder
    }
