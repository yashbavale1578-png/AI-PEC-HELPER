import os
from pathlib import Path
from dotenv import load_dotenv

# ==============================================================================
# SECTION 1: ENVIRONMENT SETUP
# ==============================================================================
# load_dotenv reads keys and values from a '.env' file in the root directory and 
# sets them as environment variables. This keeps API keys and local settings safe 
# and separated from our source code!
load_dotenv()

# ==============================================================================
# SECTION 2: BASE DIRECTORY CONFIGURATION
# ==============================================================================
# We define our folder path variables so the application knows exactly where to
# read and write data (e.g. screenshots, AI notes).
# Path(__file__) gives the absolute path to this config.py file. 
# .parent gives 'backend/', and .parent.parent gives the root folder 'AI STUDY COPILOT'.
BASE_DIR = Path(__file__).resolve().parent.parent

# Resolve specific absolute folder paths
MEMORY_DIR = BASE_DIR / "memory"
ASSETS_DIR = BASE_DIR / "assets"
DOCS_DIR = BASE_DIR / "docs"

# Automatically create these folders on startup if they don't exist yet!
# exist_ok=True ensures we don't crash if they already exist.
MEMORY_DIR.mkdir(parents=True, exist_ok=True)
ASSETS_DIR.mkdir(parents=True, exist_ok=True)
DOCS_DIR.mkdir(parents=True, exist_ok=True)

# ==============================================================================
# SECTION 3: SYSTEM CONFIGURATION CLASS
# ==============================================================================
class Config:
    """
    App-wide Configuration Settings.
    This class reads environment variables with safe defaults to make the app
    easy to customize, run, and scale.
    """
    
    # ------------------ NVIDIA NIM API Config ------------------
    # NVIDIA NIM provides high-speed, highly optimized inference endpoints.
    # The SDK interface is 100% compatible with standard OpenAI SDK parameters!
    # Obtain your key starting with 'nvapi-' from build.nvidia.com
    NVIDIA_API_KEY: str = os.getenv("NVIDIA_API_KEY", "")
    
    # Default model hosted on NVIDIA NIM. 
    # Standard high-performance student helper model: Meta Llama 3.1 70B Instruct.
    # Check current catalog at build.nvidia.com/meta/llama-3.1-70b-instruct
    NIM_MODEL: str = os.getenv("NIM_MODEL", "meta/llama-3.1-70b-instruct")
    
    # The gateway endpoint for NVIDIA's cloud-hosted NIM endpoints
    NVIDIA_BASE_URL: str = "https://integrate.api.nvidia.com/v1"

    # ------------------ Screen Capture Settings ------------------
    # How often should the backend capture the screen in automatic mode? (in seconds)
    CAPTURE_INTERVAL_SECONDS: float = float(os.getenv("CAPTURE_INTERVAL_SECONDS", "3.0"))
    
    # Quality of captured screen images (1 to 100). Higher means clearer, lower means faster.
    SCREENSHOT_QUALITY: int = int(os.getenv("SCREENSHOT_QUALITY", "80"))
    
    # Maximum number of screenshots we keep in the 'assets' folder to prevent running out of storage.
    MAX_SCREENSHOT_HISTORY: int = int(os.getenv("MAX_SCREENSHOT_HISTORY", "30"))

    # ------------------ OCR Settings ------------------
    # Language code for text recognition (e.g. 'eng' for English)
    OCR_LANGUAGE: str = os.getenv("OCR_LANGUAGE", "eng")
    
    # Dual-engine selector: 'easyocr' (deep learning) or 'tesseract' (lightweight)
    OCR_ENGINE_TYPE: str = os.getenv("OCR_ENGINE_TYPE", "tesseract")

    # ------------------ Server / Network Settings ------------------
    # Local address and port on which our FastAPI server will listen
    HOST: str = os.getenv("HOST", "127.0.0.1")
    PORT: int = int(os.getenv("PORT", "8000"))
    DEBUG: bool = os.getenv("DEBUG", "True").lower() in ("true", "1", "yes")

# Instantiate a single global settings object that other files can import
settings = Config()
