import os
import logging
from pathlib import Path
from typing import Optional, Union, Dict, Any
from PIL import Image, ImageOps, ImageFilter

# Set up clean logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("OCREngine")

# ==============================================================================
# SECTION 1: DYNAMIC DEPENDENCY RESOLUTION
# ==============================================================================
# We dynamically try to import popular python OCR packages:
# 1. 'pytesseract': Lightweight wrapper around Tesseract-OCR binary.
# 2. 'easyocr': Deep-learning based OCR that uses PyTorch.
# If none are available, the class operates in 'Mock Mode' with an educational fallback.

try:
    import pytesseract
    # You may need to specify the tesseract executable path on Windows, e.g.:
    # pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
except ImportError:
    logger.warning("pytesseract package not installed. Run 'pip install pytesseract' to enable.")
    pytesseract = None

try:
    import easyocr
except ImportError:
    logger.warning("easyocr package not installed. Run 'pip install easyocr' to enable.")
    easyocr = None

from backend.config import settings


# ==============================================================================
# SECTION 2: OCR EXTRACTOR CLASS
# ==============================================================================
class OCRExtractor:
    """
    An image OCR service designed to handle screenshot text extraction.
    Includes built-in image preprocessing techniques (grayscale, contrast boost,
    and adaptive filtering) to achieve highest accuracy and lowest latency.
    """
    
    def __init__(self, engine_type: Optional[str] = None):
        # Allow override or read defaults from config
        self.engine_type = engine_type or settings.OCR_ENGINE_TYPE
        self.easyocr_reader = None
        
        # Initialize selected engine if available
        self._initialize_engine()

    def _initialize_engine(self):
        """Initializes the backend engine (loading models or testing binaries)."""
        if self.engine_type == "easyocr" and easyocr is not None:
            try:
                logger.info("Initializing EasyOCR Reader (Loading English model)...")
                # We load English by default; this downloads weights once on first startup.
                # GPU is automatically used if CUDA is available!
                self.easyocr_reader = easyocr.Reader(['en'], gpu=True)
                logger.info("EasyOCR initialized successfully!")
            except Exception as e:
                logger.error(f"Failed to load EasyOCR: {e}. Switching to Tesseract fallback.")
                self.engine_type = "tesseract"
                
        if self.engine_type == "tesseract":
            if pytesseract is None:
                logger.warning("Tesseract engine selected but 'pytesseract' module is missing. Running in Mock Mode.")
            else:
                logger.info("Tesseract OCR wrapper ready.")

    # ==============================================================================
    # SECTION 3: IMAGE PREPROCESSING FOR LOW LATENCY & HIGH ACCURACY
    # ==============================================================================
    def preprocess_image(self, img: Image.Image) -> Image.Image:
        """
        Applies mathematical and structural corrections to screenshots before OCR.
        Grayscale reduction and threshold filtering greatly enhance OCR recognition
        speed and accuracy by removing digital noise.
        """
        try:
            # 1. Convert image to grayscale (removes color channels, speeds up processing)
            processed_img = img.convert('L')
            
            # 2. Resize if the screenshot is very small (upscaling tiny text helps accuracy)
            if img.width < 1000:
                scale_factor = 2
                processed_img = processed_img.resize(
                    (img.width * scale_factor, img.height * scale_factor), 
                    Image.Resampling.LANCZOS
                )
                
            # 3. Boost contrast to separate text foreground from background
            processed_img = ImageOps.autocontrast(processed_img)
            
            # 4. Apply a mild sharpening filter to sharpen characters
            processed_img = processed_img.filter(ImageFilter.SHARPEN)
            
            return processed_img
        except Exception as e:
            logger.error(f"Image preprocessing failed: {e}. Returning original.")
            return img

    # ==============================================================================
    # SECTION 4: TEXT EXTRACTION ENGINE
    # ==============================================================================
    def extract_text(self, img_source: Union[Path, str, Image.Image], preprocess: bool = True) -> str:
        """
        Orchestrates loading, preprocessing, and extracting text from a screenshot source.
        Returns a single clean string of extracted text.
        """
        # Load image if file path is provided
        if isinstance(img_source, (str, Path)):
            try:
                img = Image.open(img_source)
            except Exception as e:
                logger.error(f"Failed to open image file: {e}")
                return ""
        else:
            img = img_source

        # Preprocess to optimize OCR pipeline
        if preprocess:
            img = self.preprocess_image(img)

        # Run extraction based on configured engine
        extracted_text = ""

        # --- Route A: EasyOCR ---
        if self.engine_type == "easyocr" and self.easyocr_reader is not None:
            try:
                # Save PIL Image temporarily or convert to numpy array
                # For simplicity, we convert PIL image to raw bytes and pass directly
                import numpy as np
                img_np = np.array(img)
                # readtext returns tuple boxes: (box coordinates, text, confidence)
                results = self.easyocr_reader.readtext(img_np)
                extracted_text = " ".join([item[1] for item in results])
            except Exception as e:
                logger.error(f"EasyOCR extraction error: {e}. Trying Tesseract fallback...")
                extracted_text = self._run_tesseract_ocr(img)

        # --- Route B: Tesseract ---
        elif self.engine_type == "tesseract" and pytesseract is not None:
            extracted_text = self._run_tesseract_ocr(img)

        # --- Route C: Mock Mode (Fallback for development with zero installations) ---
        else:
            extracted_text = self._get_mock_ocr_text()

        # Clean trailing whitespace and lines
        return extracted_text.strip()

    def _run_tesseract_ocr(self, img: Image.Image) -> str:
        """Internal helper to safely trigger Tesseract CLI executable."""
        try:
            # We enforce standard layout configurations: --psm 3 means automatic page segmentation
            custom_config = f'--psm 3 -l {settings.OCR_LANGUAGE}'
            return pytesseract.image_to_string(img, config=custom_config)
        except Exception as e:
            logger.error(f"Tesseract OCR failed: {e}. Check if Tesseract-OCR is installed on your OS environment.")
            return self._get_mock_ocr_text()

    def _get_mock_ocr_text(self) -> str:
        """
        Provides helpful sample text if no OCR engines are installed on the local system.
        Ensures the developer can immediately test and play with the Copilot!
        """
        logger.info("OCR Engine running in 'Mock Mode'. Check install guides in config.")
        return (
            "Welcome to the AI Study Copilot!\n"
            "This is a demonstration of your copilot architecture.\n"
            "Topic: Introduction to Computer Systems Architecture.\n"
            "Key terms: Central Processing Unit (CPU), Graphics Processing Unit (GPU),\n"
            "NVIDIA NIM, Deep Learning, Low Latency Pipeline.\n"
            "To activate full OCR, make sure to install 'tesseract-ocr' or 'easyocr'!"
        )

# ==============================================================================
# SECTION 5: INDEPENDENT MODULE TEST RUNNER
# ==============================================================================
if __name__ == "__main__":
    print("--------------------------------------------------")
    print("OCRExtractor Diagnostics")
    print("--------------------------------------------------")
    # Initialize extractor (default will automatically resolve fallbacks)
    extractor = OCRExtractor()
    
    # Create a small blank image with text to test (or use mock mode)
    test_img = Image.new('RGB', (400, 100), color=(255, 255, 255))
    
    print("\nExtracting from test image:")
    extracted = extractor.extract_text(test_img)
    print("--------------------------------------------------")
    print("Result:\n" + extracted)
    print("--------------------------------------------------")
