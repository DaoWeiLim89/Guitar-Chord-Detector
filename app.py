from fastapi import FastAPI, File, UploadFile, Form, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import tempfile
import os
import uuid
import main
from pydantic import BaseModel
import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
import re
import traceback

# Create a pool of threads to handle heavy lifting
executor = ThreadPoolExecutor(max_workers=4)  # Limit workers

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    yield
    # Shutdown - clean up executor
    executor.shutdown(wait=True)

app = FastAPI(lifespan=lifespan)

# Allowed origins
ALLOWED_ORIGINS = ["https://chord-detector-website.vercel.app"]

# Middleware to restrict requests to allowed origins
@app.middleware("http")
async def check_origin(request: Request, call_next):
    if request.url.path == "/health":
        return await call_next(request)
        
    # Standard CORS check handles the browser security. 
    # We only want to block obvious non-browser requests or bad actors.
    
    # 1. OPTIONAL: Block requests with NO User-Agent (often scripts)
    user_agent = request.headers.get("user-agent", "")
    if not user_agent:
        return JSONResponse(status_code=403, content={"message": "Missing User-Agent"})

    # 2. Relaxed Origin Check
    # If Origin is present, it MUST match. If missing, we let it slide (to avoid blocking privacy users),
    # trusting the CORS middleware to handle browser contexts.
    origin = request.headers.get("origin")
    if origin and origin not in ALLOWED_ORIGINS:
         return JSONResponse(status_code=403, content={"message": "Invalid Origin"})

    return await call_next(request)

# Enable CORS for frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://chord-detector-website.vercel.app"],  # Frontend Domain
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

def clean_filename(filename: str) -> str:
    # Remove any character that isn't alphanumeric, dot, dash, or underscore
    return re.sub(r'[^a-zA-Z0-9._-]', '', filename)

@app.post("/api/analyzeChords", response_model=SuccessResponse | ErrorResponse)
@limiter.limit("2/minute")
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
        safe_name = clean_filename(file.filename)
        temp_path = os.path.join(tempfile.gettempdir(), f"audio_{uuid.uuid4()}_{safe_name}")
        
        # More robust file size check -> checking while writing
        MAX_SIZE = 50 * 1024 * 1024  # 50MB
        size_written = 0
        # Write to file (Streaming is safer for RAM)
        with open(temp_path, "wb") as buffer:
            while content := await file.read(1024 * 1024):  # Read 1MB chunks
                size_written += len(content)
                if size_written > MAX_SIZE:
                    # Delete file and abort immediately
                    buffer.close()
                    os.remove(temp_path)
                    return ErrorResponse(status="error", message="File too large")
                buffer.write(content)

        loop = asyncio.get_running_loop()
        
        # Get the formatted output
        type, output = await loop.run_in_executor(
            executor, 
            lambda: main.process_audio_file(temp_path, song_name, artist_name)
        )
         
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
        print(f"Error processing file: {e}")
        traceback.print_exc()
        return ErrorResponse(status="error", message="An error occurred during processing")
    
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path) # clean up temp file

@app.get("/health")
def health_check():
    """Health check endpoint for Render.com"""
    return {"status": "healthy"}
