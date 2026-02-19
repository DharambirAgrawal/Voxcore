"""
╔══════════════════════════════════════════════════════════════════════════════════╗
║                           VOXCORE — input/stt.py                                ║
║         GROQ WHISPER-TURBO STT — SPEECH-TO-TEXT WITH HF FALLBACK               ║
╚══════════════════════════════════════════════════════════════════════════════════╝

PURPOSE:
    Subscribes to SPEECH_END events from VAD. Takes the buffered audio, creates
    a temporary WAV file, sends it to Groq's whisper-large-v3-turbo API for
    transcription, and publishes TRANSCRIPT_READY with the recognized text.

    On Groq rate limit (HTTP 429), automatically falls back to a self-hosted
    HuggingFace faster-whisper-small endpoint for zero-cost unlimited fallback.

═══════════════════════════════════════════════════════════════════════════════════
IMPORTS REQUIRED:
═══════════════════════════════════════════════════════════════════════════════════

import asyncio                          # Async operations
import logging                          # Module logger
import tempfile                         # Create temporary WAV files
import os                               # File operations, env vars
import io                               # BytesIO for in-memory WAV

import numpy as np                      # Audio array manipulation
from scipy.io import wavfile            # Write WAV files for Groq API
from groq import AsyncGroq              # Groq SDK async client
import aiohttp                          # Async HTTP for HuggingFace fallback

from core.event_bus import EventBus, EventType

═══════════════════════════════════════════════════════════════════════════════════
CLASSES:
═══════════════════════════════════════════════════════════════════════════════════

──────────────────────────────────────────────────────────────────────────────────
CLASS: STTClient
──────────────────────────────────────────────────────────────────────────────────
    Speech-to-Text client with Groq primary and HuggingFace fallback.

    CONSTRUCTOR: __init__(self, event_bus: EventBus, config: dict)
    ─────────────────────────────────────────────────────────────
        INPUTS:
            - event_bus: EventBus — For subscribing to SPEECH_END and publishing TRANSCRIPT_READY
            - config: dict — The "models" section from config.yaml:
                - stt_primary: str ("whisper-large-v3-turbo")
                - stt_fallback: str (HuggingFace endpoint URL)
              AND the "audio" section:
                - sample_rate: int (16000)
        
        INITIALIZES:
            self.event_bus: EventBus             = event_bus
            self.primary_model: str              = config["models"].get("stt_primary", "whisper-large-v3-turbo")
            self.fallback_url: str               = config["models"].get("stt_fallback", "")
            self.sample_rate: int                = config["audio"].get("sample_rate", 16000)
            
            # Groq async client
            self._groq_client: AsyncGroq         = AsyncGroq(api_key=os.environ.get("GROQ_API_KEY"))
            
            # HuggingFace API key for fallback
            self._hf_api_key: str                = os.environ.get("HF_API_KEY", "")
            
            # Event subscription
            self._speech_end_queue: asyncio.Queue = event_bus.subscribe(EventType.SPEECH_END)
            
            # Rate limit tracking
            self._using_fallback: bool           = False
            self._fallback_until: float          = 0.0   # timestamp when we can retry Groq
            
            self._logger: logging.Logger         = logging.getLogger("STT")

    METHODS:
    ─────────────────────────────────────────────────────────────

    async def run(self) -> None
        INPUTS: None
        OUTPUT: None (runs forever)
        WHAT IT DOES:
            Infinite loop:
            1. event = await self._speech_end_queue.get()
            2. audio_buffer = event.data["audio_buffer"]   # raw PCM bytes (int16)
            3. duration = event.data["duration_s"]
            4. If duration < 0.3: skip (too short to be meaningful speech)
            5. wav_bytes = self._pcm_to_wav(audio_buffer)
            6. Try:
               transcript = await self._transcribe_groq(wav_bytes)
               If blocked (rate limited):
                   transcript = await self._transcribe_fallback(wav_bytes)
            7. If transcript is empty or whitespace: skip
            8. Publish TRANSCRIPT_READY event with data:
               {"text": transcript, "confidence": 1.0, "language": "en"}
            9. Log: "Transcript: '{transcript}'"

    def _pcm_to_wav(self, pcm_bytes: bytes) -> bytes
        INPUTS:
            - pcm_bytes: bytes — Raw PCM audio data, int16, 16kHz mono
        OUTPUT:
            - bytes — Complete WAV file in memory as bytes
        WHAT IT DOES:
            1. Converts pcm_bytes to numpy array: audio = np.frombuffer(pcm_bytes, dtype=np.int16)
            2. Creates a BytesIO buffer: buf = io.BytesIO()
            3. Writes WAV: wavfile.write(buf, self.sample_rate, audio)
            4. Returns buf.getvalue()
        
        NOTE: Groq's STT API expects a WAV file upload, not raw PCM.

    async def _transcribe_groq(self, wav_bytes: bytes) -> str
        INPUTS:
            - wav_bytes: bytes — WAV file content
        OUTPUT:
            - str — Transcribed text
        RAISES:
            - RateLimitError (or similar) if Groq returns 429
        WHAT IT DOES:
            1. Check if currently rate-limited:
               If time.time() < self._fallback_until, raise rate limit error
            2. Create a temporary file or use BytesIO:
               file_tuple = ("audio.wav", wav_bytes, "audio/wav")
            3. Call Groq API:
               transcription = await self._groq_client.audio.transcriptions.create(
                   file=file_tuple,
                   model=self.primary_model,
                   language="en",
                   response_format="text"
               )
            4. Return transcription.text.strip()
        
        ERROR HANDLING:
            On groq.RateLimitError or HTTP 429:
                - Set self._using_fallback = True
                - Set self._fallback_until = time.time() + 60  (retry in 60s)
                - Log warning: "Groq STT rate limited, falling back to HF"
                - Re-raise so run() catches and calls fallback

    async def _transcribe_fallback(self, wav_bytes: bytes) -> str
        INPUTS:
            - wav_bytes: bytes — WAV file content
        OUTPUT:
            - str — Transcribed text from HuggingFace endpoint
        WHAT IT DOES:
            1. If self.fallback_url is empty:
               Log error "No fallback STT endpoint configured"
               Return ""
            2. Create aiohttp session:
               async with aiohttp.ClientSession() as session:
                   headers = {"Authorization": f"Bearer {self._hf_api_key}"}
                   data = aiohttp.FormData()
                   data.add_field('file', wav_bytes, filename='audio.wav', content_type='audio/wav')
                   async with session.post(self.fallback_url, data=data, headers=headers) as resp:
                       result = await resp.json()
                       return result.get("text", "").strip()
        
        ERROR HANDLING:
            On any aiohttp exception:
                - Log error with traceback
                - Return "" (empty transcript — this turn is lost)
        
        NOTE: The HuggingFace endpoint should be a faster-whisper-small deployment
              that accepts WAV files and returns {"text": "transcription..."}.
              Deploy using HuggingFace Spaces with Gradio or a simple Flask API.

    async def check_groq_availability(self) -> bool
        INPUTS: None
        OUTPUT: bool — Whether Groq STT is currently available (not rate limited)
        WHAT IT DOES:
            1. If time.time() > self._fallback_until:
               Set self._using_fallback = False
               Return True
            2. Return False

═══════════════════════════════════════════════════════════════════════════════════
EXPORTS:
    - STTClient    (class)
═══════════════════════════════════════════════════════════════════════════════════

LATENCY EXPECTATIONS:
    - Groq whisper-large-v3-turbo: ~200ms for a 3-second audio clip
    - HuggingFace faster-whisper-small (fallback): ~500-1000ms depending on server
    - WAV conversion overhead: <5ms
    - Total STT step: 200-250ms (Groq) or 500-1000ms (fallback)

RATE LIMITS (Groq Free Tier):
    - whisper-large-v3-turbo: 20 requests/minute, 7,200 audio seconds/hour
    - whisper-large-v3: 20 requests/minute, 7,200 audio seconds/hour
    - At 3-5 second speech segments, 20 RPM gives ~60-100 seconds of speech/minute
      which is more than enough for natural conversation (humans speak ~30-40 seconds/min)
"""
