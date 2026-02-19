"""
╔══════════════════════════════════════════════════════════════════════════════════╗
║                       VOXCORE — output/tts_client.py                            ║
║       GROQ ORPHEUS STREAMING TTS — CONVERTS SENTENCES TO AUDIO IN REAL-TIME    ║
╚══════════════════════════════════════════════════════════════════════════════════╝

PURPOSE:
    Subscribes to LLM_SPEECH_TOKEN events (one sentence at a time). For each
    sentence, calls Groq Orpheus TTS API to synthesize audio. Uses response
    streaming to push audio chunks to the AudioPlayer BEFORE the full sentence
    is synthesized.

    Target: < 200ms time-to-first-audio-byte (TTFAB).

    Orpheus supports inline vocal direction tags ([cheerful], [sad], [excited], etc.)
    which are passed through as-is. The emotion tag in each sentence controls
    the prosody of that specific sentence.

═══════════════════════════════════════════════════════════════════════════════════
IMPORTS REQUIRED:
═══════════════════════════════════════════════════════════════════════════════════

import asyncio                          # Async operations
import logging                          # Module logger
import os                               # Environment variables

from groq import AsyncGroq              # Groq async SDK for TTS
from core.event_bus import EventBus, EventType
from output.voice_profile import VoiceProfile

═══════════════════════════════════════════════════════════════════════════════════
CLASSES:
═══════════════════════════════════════════════════════════════════════════════════

──────────────────────────────────────────────────────────────────────────────────
CLASS: TTSClient
──────────────────────────────────────────────────────────────────────────────────
    Groq Orpheus TTS client that converts sentences to streaming audio.

    CONSTRUCTOR: __init__(self, event_bus: EventBus, voice_profile: VoiceProfile, config: dict)
    ─────────────────────────────────────────────────────────────
        INPUTS:
            - event_bus: EventBus — Subscribe to LLM_SPEECH_TOKEN, publish TTS_CHUNK_READY
            - voice_profile: VoiceProfile — Voice name, sample rate, format settings
            - config: dict — The "models" section from config.yaml:
                - tts: str ("canopylabs/orpheus-v1-english")
        
        INITIALIZES:
            self.event_bus: EventBus             = event_bus
            self.voice_profile: VoiceProfile     = voice_profile
            self.model: str                      = config.get("tts", "canopylabs/orpheus-v1-english")
            self._groq_client: AsyncGroq         = AsyncGroq(api_key=os.environ.get("GROQ_API_KEY"))
            
            self._speech_token_queue: asyncio.Queue = event_bus.subscribe(EventType.LLM_SPEECH_TOKEN)
            self._is_cancelled: bool             = False
            
            self._logger: logging.Logger         = logging.getLogger("TTSClient")

    METHODS:
    ─────────────────────────────────────────────────────────────

    async def run(self) -> None
        INPUTS: None
        OUTPUT: None (runs forever)
        WHAT IT DOES:
            1. Registers callback on INTERRUPT_DETECTED to set self._is_cancelled = True
            2. Enters infinite loop:
               a. event = await self._speech_token_queue.get()
               b. If self._is_cancelled:
                  Drain the queue (clear any pending sentences)
                  self._is_cancelled = False
                  Continue
               c. sentence = event.data["text"]
               d. sentence_index = event.data["sentence_index"]
               e. emotion = event.data.get("emotion", "")
               f. await self._synthesize_sentence(sentence, sentence_index)

    async def _synthesize_sentence(self, sentence: str, sentence_index: int) -> None
        INPUTS:
            - sentence: str — The sentence to synthesize (may include emotion tags)
            - sentence_index: int — Ordering index for the audio player queue
        OUTPUT: None (publishes TTS_CHUNK_READY events)
        WHAT IT DOES:
            1. Log: "TTS: synthesizing sentence {sentence_index}: '{sentence[:60]}...'"
            2. Call Groq Orpheus TTS with streaming:
               response = await self._groq_client.audio.speech.create(
                   model=self.model,
                   input=sentence,
                   voice=self.voice_profile.voice_name,
                   response_format=self.voice_profile.response_format,  # "wav"
               )
            3. Read the response:
               OPTION A — If Groq SDK supports streaming response:
                   async for chunk in response.iter_bytes(chunk_size=4096):
                       If self._is_cancelled: break
                       Publish TTS_CHUNK_READY event with data:
                           {"audio": chunk, "sentence_index": sentence_index}
               
               OPTION B — If response is complete bytes:
                   audio_bytes = response.content
                   # Split into chunks for incremental playback
                   chunk_size = 4096
                   for i in range(0, len(audio_bytes), chunk_size):
                       If self._is_cancelled: break
                       chunk = audio_bytes[i:i + chunk_size]
                       Publish TTS_CHUNK_READY event with data:
                           {"audio": chunk, "sentence_index": sentence_index}
            
            4. Publish TTS_SENTENCE_DONE event with data:
               {"sentence_index": sentence_index}
            5. Log: "TTS: sentence {sentence_index} synthesized"
        
        ERROR HANDLING:
            On any Groq error:
                - Log error: "TTS synthesis failed: {error}"
                - Skip this sentence (don't block the pipeline)
            On rate limit:
                - Log warning
                - Wait and retry once

    async def cancel(self) -> None
        INPUTS: None
        OUTPUT: None
        WHAT IT DOES:
            1. Sets self._is_cancelled = True
            2. Drains self._speech_token_queue (empties pending sentences)
            3. Log: "TTS cancelled — clearing queue"
        
        Called by TurnManager on INTERRUPT_DETECTED.

    async def synthesize_single(self, text: str) -> bytes
        INPUTS:
            - text: str — Text to synthesize (for one-off synthesis, e.g. safety refusal)
        OUTPUT:
            - bytes — Complete WAV audio bytes
        WHAT IT DOES:
            1. Calls Groq Orpheus TTS (non-streaming):
               response = await self._groq_client.audio.speech.create(
                   model=self.model,
                   input=text,
                   voice=self.voice_profile.voice_name,
                   response_format="wav"
               )
            2. Returns response.content
        
        Used for one-off synthesis (safety refusal messages, etc.)

═══════════════════════════════════════════════════════════════════════════════════
EXPORTS:
    - TTSClient   (class)
═══════════════════════════════════════════════════════════════════════════════════

LATENCY NOTE:
    Groq Orpheus TTS achieves ~150-200ms time-to-first-byte (TTFB).
    Combined with the fast brain's ~100-200ms TTFT, the first audio word
    plays ~350-450ms after the LLM starts generating.

ORPHEUS VOICES:
    English: tara, leah, jess, leo, dan, mia, zac, zoe
    Arabic (Saudi): fahad, sultan, lulwa, noura

ORPHEUS EMOTION TAGS (passed inline in text):
    [cheerful] [calm] [concerned] [excited] [empathetic]
    [curious] [surprised] [sad] [angry] [whisper] [laugh]
    
    Example: "[excited] I found it! [calm] Let me explain what happened."
"""
