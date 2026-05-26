import time
import logging
import asyncio
from pathlib import Path
from typing import Optional, Dict, Any, List
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# Setup clean logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("StudyCopilotServer")

# ==============================================================================
# SECTION 1: IN-PROJECT IMPORTS
# ==============================================================================
from backend.config import settings
from backend.screen_capture import ScreenCaptureService
from backend.ocr_engine import OCRExtractor
from backend.ai_engine import AIEngine, MicrophoneTranscriptionService

# ==============================================================================
# SECTION 2: APP INITIALIZATION & CORS CONFIG
# ==============================================================================
app = FastAPI(
    title="AI Study Copilot API",
    description="Low-latency API orchestrator powered by NVIDIA NIM & FastAPI",
    version="1.0.0"
)

# Enable CORS (Cross-Origin Resource Sharing) so your Frontend (e.g. React/Vite) 
# can securely talk to this backend server even if they are hosted on different ports!
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In development, allow all origins. Set specific URLs for production!
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Instantiate core services
capture_service = ScreenCaptureService()
ocr_extractor = OCRExtractor()
ai_engine = AIEngine()
audio_service = MicrophoneTranscriptionService()

# In-memory session state storage
session_state = {
    "is_auto_capturing": False,
    "last_captured_image_path": None,
    "last_extracted_text": "",
    "study_history": []
}

# ==============================================================================
# SECTION 3: REQUEST AND RESPONSE MODELS
# ==============================================================================
class StudyQueryRequest(BaseModel):
    custom_question: Optional[str] = None

class TargetWindowRequest(BaseModel):
    window_title: str

# ==============================================================================
# SECTION 4: API ENDPOINTS
# ==============================================================================

@app.get("/")
async def root():
    """Health check endpoint to ensure server is running smoothly."""
    return {
        "status": "healthy",
        "service": "AI Study Copilot Backend",
        "message": "Study assistant is ready for your learning sessions!"
    }


@app.get("/windows")
async def get_active_windows():
    """Lists all active windows. Useful for selecting a specific slideshow or book reader."""
    windows = capture_service.list_active_windows()
    return {"windows": windows}


@app.post("/capture")
async def capture_and_analyze(request: Optional[StudyQueryRequest] = None):
    """
    Captures the screen, runs OCR to extract text, and executes the NVIDIA NIM 
    helper to explain the concepts. Automatically saves results to memory!
    """
    try:
        # 1. Capture screenshot (low latency)
        img = capture_service.capture_fullscreen()
        if not img:
            raise HTTPException(status_code=500, detail="Failed to capture screenshot.")
        
        # Save image and register path
        image_path = capture_service.save_capture(img)
        session_state["last_captured_image_path"] = str(image_path)
        
        # 2. Extract text (low-latency OCR with pre-processing)
        extracted_text = ocr_extractor.extract_text(img)
        session_state["last_extracted_text"] = extracted_text
        
        # 3. Request assistance from NVIDIA NIM
        custom_question = request.custom_question if request else None
        ai_response = ai_engine.generate_study_help(extracted_text, custom_question)
        
        # Save analysis to history/memory folder
        timestamp = int(time.time())
        note_content = (
            f"=== STUDY SESSION NOTE ({timestamp}) ===\n"
            f"Screenshot File: {image_path.name}\n"
            f"Extracted Text:\n{extracted_text}\n\n"
            f"AI Copilot Assistant Explanation:\n{ai_response}\n"
        )
        
        note_file = settings.MEMORY_DIR / f"study_note_{timestamp}.txt"
        with open(note_file, "w", encoding="utf-8") as f:
            f.write(note_content)
            
        # Update in-memory state
        record = {
            "timestamp": timestamp,
            "image": image_path.name,
            "text": extracted_text,
            "explanation": ai_response
        }
        session_state["study_history"].append(record)
        
        return record
        
    except Exception as e:
        logger.error(f"Error during capture and analyze sequence: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query/stream")
async def query_copilot_stream(request: StudyQueryRequest):
    """
    Query current screen context with custom questions using a streaming response.
    Perceived latency drops to almost ZERO since content is displayed instantly chunk-by-chunk!
    """
    extracted_text = session_state["last_extracted_text"]
    if not extracted_text:
        extracted_text = "No screen captured yet. Please trigger /capture first."
        
    # Streaming response headers tell the browser to keep reading the chunk streams
    return StreamingResponse(
        ai_engine.generate_study_help_stream(extracted_text, request.custom_question),
        media_type="text/event-stream"
    )


@app.get("/history")
async def get_study_history():
    """Retrieves all saved notes from the memory folder."""
    saved_notes = []
    try:
        note_files = sorted(
            settings.MEMORY_DIR.glob("study_note_*.txt"),
            key=lambda x: x.stat().st_mtime,
            reverse=True
        )
        for note_file in note_files:
            # Parse simple timestamp from name
            parts = note_file.stem.split("_")
            timestamp = int(parts[-1]) if len(parts) > 1 else int(time.time())
            
            with open(note_file, "r", encoding="utf-8") as f:
                content = f.read()
                
            saved_notes.append({
                "timestamp": timestamp,
                "file_name": note_file.name,
                "snippet": content[:300] + "..." if len(content) > 300 else content
            })
    except Exception as e:
        logger.error(f"Failed to read study memory folder: {e}")
        
    return {"history": saved_notes}


# ==============================================================================
# SECTION 5: MICROPHONE API ENDPOINTS (FUTURE INTEGRATION READY)
# ==============================================================================

@app.post("/audio/start")
async def start_audio_recording():
    """Starts recording microphoned lecture. Hooks directly into transcription services!"""
    success = audio_service.start_recording()
    if not success:
        return {"status": "error", "message": "Recording session already in progress."}
    return {"status": "success", "message": "Microphone recording started!"}


@app.post("/audio/stop")
async def stop_audio_recording():
    """Stops audio capture, triggers transcription, and returns transcribed lecture notes."""
    transcription = audio_service.stop_recording()
    return {
        "status": "success",
        "transcription": transcription,
        "advice": "Use this text as extra context inside custom AI queries!"
    }


# ==============================================================================
# SECTION 6: BACKGROUND TASKS & UTILITIES
# ==============================================================================
async def continuous_capture_loop(interval: float):
    """Loop background task to continuously capture screen for study updates."""
    logger.info("Continuous auto-capture background daemon started.")
    while session_state["is_auto_capturing"]:
        try:
            img = capture_service.capture_fullscreen()
            if img:
                raw_text = ocr_extractor.extract_text(img)
                session_state["last_extracted_text"] = raw_text
                logger.info(f"Auto-capture complete: {len(raw_text)} chars extracted.")
        except Exception as e:
            logger.error(f"Error in background capture thread: {e}")
            
        await asyncio.sleep(interval)


@app.post("/auto-capture/start")
async def start_auto_capture(background_tasks: BackgroundTasks):
    """Enables automatic periodic background screenshots for seamless hands-free studying!"""
    if session_state["is_auto_capturing"]:
        return {"message": "Auto-capture daemon already active."}
        
    session_state["is_auto_capturing"] = True
    background_tasks.add_task(continuous_capture_loop, settings.CAPTURE_INTERVAL_SECONDS)
    return {"status": "success", "message": "Automatic screenshot capture started."}


@app.post("/auto-capture/stop")
async def stop_auto_capture():
    """Disables background screenshots."""
    session_state["is_auto_capturing"] = False
    return {"status": "success", "message": "Automatic screenshot capture stopped."}


# ==============================================================================
# SECTION 7: MAIN RUNNER BLOCK
# ==============================================================================
if __name__ == "__main__":
    import uvicorn
    # Start the server on configured address/port
    print(f"🚀 Launching AI Study Copilot on http://{settings.HOST}:{settings.PORT}")
    uvicorn.run("backend.main:app", host=settings.HOST, port=settings.PORT, reload=settings.DEBUG)
