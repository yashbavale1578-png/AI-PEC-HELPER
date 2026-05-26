import logging
from typing import Generator, Optional, List, Dict
import openai
from openai import OpenAI

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("AIEngine")

from backend.config import settings

# ==============================================================================
# SECTION 1: NVIDIA NIM ENGINE CLASS
# ==============================================================================
class AIEngine:
    """
    Manages study interactions with the NVIDIA NIM microservice.
    NVIDIA NIM offers high throughput, ultra-low latency AI endpoints.
    The SDK is fully compatible with standard OpenAI chat structure.
    """
    
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        # Retrieve key from settings or parameter
        self.api_key = api_key or settings.NVIDIA_API_KEY
        self.model_name = model or settings.NIM_MODEL
        
        self.client = None
        self._setup_client()

    def _setup_client(self):
        """Initializes the OpenAI wrapper client directed at NVIDIA's endpoints."""
        if not self.api_key:
            logger.warning(
                "NVIDIA_API_KEY is missing! Call API will run in Mock Demo mode.\n"
                "To fix: Add your key to .env file as NVIDIA_API_KEY=nvapi-xxxx"
            )
            return

        try:
            # Point the standard OpenAI SDK to the NVIDIA NIM REST endpoint
            self.client = OpenAI(
                base_url=settings.NVIDIA_BASE_URL,
                api_key=self.api_key
            )
            logger.info(f"NVIDIA NIM client successfully initialized with model: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to create NVIDIA NIM client: {e}")

    # ==============================================================================
    # SECTION 2: CORE INFERENCE METHODS (LOW LATENCY STREAMING)
    # ==============================================================================
    def generate_study_help(self, screen_text: str, custom_question: Optional[str] = None) -> str:
        """
        Queries the NVIDIA model with the extracted text from the student's screen.
        Non-streaming blocking method. Excellent for structured outputs.
        """
        if not self.client:
            return self._get_mock_ai_response(screen_text, custom_question)

        # Build clean instruction prompting to get the best educational responses
        system_prompt = (
            "You are an expert, encouraging AI Study Copilot.\n"
            "Your goal is to explain the educational content captured on the user's screen in simple, "
            "beginner-friendly terms. Highlight key concepts, define vocabulary, and answer queries concisely."
        )

        user_content = f"CONTEXT FROM MY STUDY SCREEN:\n\"\"\"\n{screen_text}\n\"\"\"\n\n"
        if custom_question:
            user_content += f"STUDENT QUESTION: {custom_question}"
        else:
            user_content += "Please provide a concise overview of what is on this screen, defining any major terms."

        try:
            # Standard chat completion structure
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content}
                ],
                temperature=0.2, # Low temperature ensures high factual accuracy and consistency
                max_tokens=1024
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"NVIDIA NIM API request failed: {e}")
            return f"Error talking to NVIDIA NIM: {e}. Check API key validity."

    def generate_study_help_stream(self, screen_text: str, custom_question: Optional[str] = None) -> Generator[str, None, None]:
        """
        A streaming generator designed to maximize responsiveness.
        Streams text chunks from NVIDIA NIM the split-second they are generated.
        This provides a premium 'low-latency' UI experience!
        """
        if not self.client:
            # Fallback mock streaming generator
            yield "Mock Demo Mode Streaming Chunk 1...\n"
            yield "Mock Demo Mode Streaming Chunk 2: " + (custom_question or "Overview explanation.")
            return

        system_prompt = (
            "You are an expert AI Study Copilot.\n"
            "Explain concepts clearly, write definitions, and answer student questions directly and helpfully."
        )

        user_content = f"CONTEXT FROM MY STUDY SCREEN:\n\"\"\"\n{screen_text}\n\"\"\"\n\n"
        if custom_question:
            user_content += f"STUDENT QUESTION: {custom_question}"
        else:
            user_content += "Please summarize the core ideas on the screen into study notes."

        try:
            # Enable stream=True in the OpenAI client request
            stream = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content}
                ],
                temperature=0.3,
                max_tokens=1024,
                stream=True
            )
            
            # Yield text fragments as they arrive
            for chunk in stream:
                content = chunk.choices[0].delta.content
                if content is not None:
                    yield content
        except Exception as e:
            logger.error(f"NVIDIA NIM Streaming failed: {e}")
            yield f"\n[Streaming Error: {e}]"

    def _get_mock_ai_response(self, text: str, question: Optional[str]) -> str:
        """Fallback mock system to facilitate testing when API keys are not provided."""
        return (
            "💡 [AI COPILOT DEMO MODE]\n"
            f"I analyzed your screen containing the text: '{text[:80]}...'\n\n"
            f"Here is your study summary:\n"
            "1. NVIDIA NIM refers to NVIDIA Inference Microservices, which allow low-latency model runs.\n"
            "2. Modular programming makes code easy to test and debug.\n"
            "3. Good notes are critical to retention!\n\n"
            f"You asked: '{question or 'None specified'}' - This is a great area to study further! "
            "To connect this directly to Gemini or Llama 3 models on NVIDIA, add your api key in the backend/.env."
        )


# ==============================================================================
# SECTION 3: FUTURE MICROPHONE TRANSCRIPTION SKELETON
# ==============================================================================
class MicrophoneTranscriptionService:
    """
    A plug-and-play structural hook designed to host future audio features.
    You can easily install a local speech engine like Whisper or Google Speech,
    and reference this class to transcribe mic audio in real time during lectures!
    """
    
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self.is_recording = False
        logger.info("Microphone Transcription Service instantiated (ready for Whisper integration!).")

    def start_recording(self) -> bool:
        """Starts recording audio from default input device."""
        if self.is_recording:
            logger.warning("Already recording!")
            return False
            
        self.is_recording = True
        logger.info("🎤 Microphone recording started (mock).")
        # TODO: Implement speech-to-text audio buffer capture using PyAudio or sounddevice
        return True

    def stop_recording(self) -> str:
        """Stops recording audio and initiates speech transcription models."""
        if not self.is_recording:
            logger.warning("No active recording session found.")
            return ""
            
        self.is_recording = False
        logger.info("🎤 Microphone recording stopped.")
        
        # TODO: Transcribe buffer using Whisper e.g. self.whisper_model.transcribe(audio_path)
        mock_transcription = "[MOCK TRANSCRIPTION: 'We are learning about multi-core parallel architecture today. Make sure to note GPU memory hierarchies.']"
        return mock_transcription

    def transcribe_audio_file(self, file_path: str) -> str:
        """Transcribes an existing audio file (e.g. mp3/wav format)."""
        logger.info(f"Processing audio file: {file_path}")
        return "[Mock Transcription for processed file]"


# ==============================================================================
# SECTION 4: INDEPENDENT MODULE TEST RUNNER
# ==============================================================================
if __name__ == "__main__":
    print("--------------------------------------------------")
    print("AIEngine & Audio Service Diagnostics")
    print("--------------------------------------------------")
    
    # Initialize Engine (Mock since we have no API keys)
    engine = AIEngine()
    
    print("\n[1] Testing AI Explanation:")
    ans = engine.generate_study_help("Topic: Artificial Intelligence, 3 basic steps: Input, Compute, Output.")
    print(ans)
    
    print("\n[2] Testing Microphone Transcription Hooks:")
    audio_service = MicrophoneTranscriptionService()
    audio_service.start_recording()
    time.sleep(1)
    transcription = audio_service.stop_recording()
    print(f"Result: {transcription}")
    print("--------------------------------------------------")
