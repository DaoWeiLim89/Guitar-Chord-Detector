from fastapi import FastAPI, File, UploadFile, Form, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import tempfile
import shutil
import os
from typing import Optional
import uuid
import main
from pydantic import BaseModel

app = FastAPI()

# Enable CORS for frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change this to your frontend domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class SuccessResponse(BaseModel):
    status: str
    type: str
    output: str
    song_name: str
    artist_name: str

class ErrorResponse(BaseModel):
    status: str
    message: str

# Set up rate limiting
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(
        status_code=429,
        content={
            "status": "error", 
            "message": "Rate limit exceeded. Please try again later."
        }
    )

@app.post("/api/analyzeChords", response_model=SuccessResponse | ErrorResponse)
@limiter.limit("3/minute")
async def analyze_Chords(
    request: Request, # instead of websockets (doesn't need open connection for back and forth conversation)
    file: UploadFile = File(...),  # Max file size ~50MB
    song_name: str = Form(...),
    artist_name: str = Form(...)
):
    """
    Main endpoint: Accept MP3 file, song name, and artist name
    Returns: JSON with detected chords and lyrics
    """
    # Check file size
    if file.size and file.size > 50_000_000:
        return ErrorResponse(status="error", message="File size exceeds the 50MB limit")
    # Check file safety
    if not file.filename:
        return ErrorResponse(status="error", message="No filename provided")
    if not file.filename.endswith(('.mp3', '.wav', '.m4a')):    
        return ErrorResponse(status="error", message="Invalid file type")
    
    # Verify actual file content
    #Read first few bytes to check file signature/magic numbers
    await file.seek(0)
    header = await file.read(12)
    await file.seek(0)  # Reset for later use

    # MP3 files start with ID3 or have MPEG sync bytes
    is_mp3 = header.startswith(b'ID3') or (len(header) > 1 and header[0] == 0xFF and (header[1] & 0xE0) == 0xE0)
    # WAV files start with 'RIFF' and contain 'WAVE'
    is_wav = header.startswith(b'RIFF') and b'WAVE' in header
    # M4A files start with specific atoms
    is_m4a = b'ftyp' in header[:12]

    if not (is_mp3 or is_wav or is_m4a):
        return ErrorResponse(status="error", message="File content is not a valid audio file")
    
    # Process the audio file
    temp_path = None
    try:
        # Save uploaded file temporarily
        temp_path = os.path.join(tempfile.gettempdir(), f"audio_{uuid.uuid4()}_{file.filename}")
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Get the formatted output
        type, output = main.process_audio_file(temp_path, song_name, artist_name)
        
        # Clean up temp file
        os.remove(temp_path)
         
        if output is None or output.strip() == "":
            return ErrorResponse(status="error", message="Could not process the audio file or retrieve lyrics. output is None.")
        else:
            return SuccessResponse(
                status="success",
                type=type, # "Not Set", "synced", "unsynced", "chords_only"
                output=output,
                song_name=song_name,
                artist_name=artist_name
            )
        
    except Exception as e:
        print(f"Processing error: {str(e)}")
        return ErrorResponse(status="error", message="An error occurred during processing")
    
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

@app.get("/health")
def health_check():
    """Health check endpoint for Render.com"""
    return {"status": "healthy"}
