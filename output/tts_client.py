"""
╔══════════════════════════════════════════════════════════════════════════════════╗
║                       VOXCORE — output/tts_client.py                            ║
║     KOKORO-ONNX LOCAL TTS — CONVERTS SENTENCES TO AUDIO IN REAL-TIME           ║
╚══════════════════════════════════════════════════════════════════════════════════╝

PURPOSE:
    Subscribes to LLM_SPEECH_TOKEN events (one sentence at a time). For each
    sentence, runs Kokoro-ONNX locally to synthesize audio. The model runs
    entirely on CPU — no API calls, no rate limits, zero cost.

    Kokoro-ONNX generates 24kHz WAV audio. The LLM emotion tags (e.g.
    [cheerful], [calm]) are STRIPPED before synthesis since Kokoro does not
    use bracket-style emotion tags — it relies on punctuation and the voice
    style itself for prosody.

    Target: < 100ms synthesis for short sentences (local model, no network).

═══════════════════════════════════════════════════════════════════════════════════
IMPORTS REQUIRED:
═══════════════════════════════════════════════════════════════════════════════════

import asyncio                          # Async operations
import io                              # BytesIO for WAV conversion
import logging                          # Module logger
import re                              # Strip emotion tags

import numpy as np                      # Audio array operations
import soundfile as sf                  # WAV encoding
from kokoro_onnx import Kokoro          # Local ONNX TTS engine

from core.event_bus import EventBus, EventType
from output.voice_profile import VoiceProfile

═══════════════════════════════════════════════════════════════════════════════════
CONSTANTS:
═══════════════════════════════════════════════════════════════════════════════════

EMOTION_TAG_PATTERN = re.compile(r'\[\w+\]\s*')   # Strip [cheerful], [calm], etc.
KOKORO_SAMPLE_RATE = 24000                          # Kokoro outputs 24kHz audio

═══════════════════════════════════════════════════════════════════════════════════
CLASSES:
═══════════════════════════════════════════════════════════════════════════════════

──────────────────────────────────────────────────────────────────────────────────
CLASS: TTSClient
──────────────────────────────────────────────────────────────────────────────────
    Kokoro-ONNX TTS client that converts sentences to audio locally.

    CONSTRUCTOR: __init__(self, event_bus: EventBus, voice_profile: VoiceProfile, config: dict)
    ─────────────────────────────────────────────────────────────
        INPUTS:
            - event_bus: EventBus — Subscribe to LLM_SPEECH_TOKEN, publish TTS_CHUNK_READY
            - voice_profile: VoiceProfile — Voice ID, sample rate, settings
            - config: dict — The "models" section from config.yaml:
                - tts_model: str ("models/kokoro-v1.0.onnx")
                - tts_voices: str ("models/voices-v1.0.bin")

        INITIALIZES:
            self.event_bus: EventBus             = event_bus
            self.voice_profile: VoiceProfile     = voice_profile
            model_path = config.get("tts_model", "models/kokoro-v1.0.onnx")
            voices_path = config.get("tts_voices", "models/voices-v1.0.bin")
            self._kokoro: Kokoro                 = Kokoro(model_path, voices_path)

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
               e. Strip emotion tags: sentence = EMOTION_TAG_PATTERN.sub("", sentence).strip()
               f. If sentence: await self._synthesize_sentence(sentence, sentence_index)

    async def _synthesize_sentence(self, sentence: str, sentence_index: int) -> None
        INPUTS:
            - sentence: str — Clean sentence (emotion tags already stripped)
            - sentence_index: int — Ordering index for the audio player queue
        OUTPUT: None (publishes TTS_CHUNK_READY events)
        WHAT IT DOES:
            1. Log: "TTS: synthesizing sentence {sentence_index}: '{sentence[:60]}...'"
            2. Run Kokoro in executor (it's sync/CPU-bound):
               loop = asyncio.get_event_loop()
               samples, sample_rate = await loop.run_in_executor(
                   None, self._kokoro.create, sentence,
                   self.voice_profile.voice_name, 1.0, "en-us"
               )
            3. Convert numpy array to WAV bytes:
               buf = io.BytesIO()
               sf.write(buf, samples, sample_rate, format="WAV")
               wav_bytes = buf.getvalue()
            4. Split WAV bytes into chunks for incremental playback:
               chunk_size = 4096
               for i in range(0, len(wav_bytes), chunk_size):
                   If self._is_cancelled: break
                   chunk = wav_bytes[i:i + chunk_size]
                   is_last = (i + chunk_size) >= len(wav_bytes)
                   Publish TTS_CHUNK_READY event with data:
                       {"audio": chunk, "sentence_index": sentence_index,
                        "sentence_done": is_last}
            5. Publish TTS_SENTENCE_DONE event
            6. Log: "TTS: sentence {sentence_index} synthesized"

        ERROR HANDLING:
            On any error:
                - Log error: "TTS synthesis failed: {error}"
                - Skip this sentence (don't block the pipeline)

    async def cancel(self) -> None
        INPUTS: None
        OUTPUT: None
        WHAT IT DOES:
            1. Sets self._is_cancelled = True
            2. Drains self._speech_token_queue (empties pending sentences)
            3. Log: "TTS cancelled — clearing queue"

    async def synthesize_single(self, text: str) -> bytes
        INPUTS:
            - text: str — Text to synthesize (for one-off synthesis, e.g. safety refusal)
        OUTPUT:
            - bytes — Complete WAV audio bytes
        WHAT IT DOES:
            1. Strip emotion tags: text = EMOTION_TAG_PATTERN.sub("", text).strip()
            2. Run Kokoro synthesis in executor
            3. Convert to WAV bytes and return

═══════════════════════════════════════════════════════════════════════════════════
EXPORTS:
    - TTSClient   (class)
═══════════════════════════════════════════════════════════════════════════════════

LATENCY NOTE:
    Kokoro-ONNX runs locally on CPU. For short sentences (< 20 words),
    synthesis takes ~50-150ms. No network latency. No rate limits.
    Combined with the fast brain's ~100-200ms TTFT, the first audio word
    plays ~250-450ms after the LLM starts generating.

KOKORO VOICES (partial list):
    Female: af_heart, af_bella, af_nicole, af_sarah, af_sky, af_alloy,
            af_aoede, af_jessica, af_kore, af_nova, af_river
    Male:   am_adam, am_michael, am_echo, am_eric, am_fenrir, am_liam,
            am_onyx, am_puck
    British Female: bf_alice, bf_emma, bf_isabella, bf_lily
    British Male:   bm_daniel, bm_fable, bm_george, bm_lewis

EMOTION HANDLING:
    Kokoro does NOT use bracket-style emotion tags like [cheerful].
    The LLM still generates them (for context), but TTSClient strips
    them before passing text to Kokoro. Kokoro infers prosody from
    punctuation (!, ?, ...) and the voice's natural style.

    Download models from:
    https://github.com/thewh1teagle/kokoro-onnx/releases/tag/model-files
"""


import asyncio
import io
import logging
import re
from typing import Optional

import numpy as np
import soundfile as sf
from kokoro_onnx import Kokoro

from core.event_bus import EventBus, EventType
from output.voice_profile import VoiceProfile


# Strip LLM emotion tags before sending to Kokoro
EMOTION_TAG_PATTERN = re.compile(r'\[\w+\]\s*')
KOKORO_SAMPLE_RATE = 24000


class TTSClient:
    """Kokoro-ONNX local TTS client that converts sentences to audio."""

    def __init__(self, event_bus: EventBus, voice_profile: VoiceProfile, config: dict) -> None:
        self.event_bus: EventBus = event_bus
        self.voice_profile: VoiceProfile = voice_profile

        model_path = config.get("tts_model", "models/kokoro-v1.0.onnx")
        voices_path = config.get("tts_voices", "models/voices-v1.0.bin")
        self._kokoro: Kokoro = Kokoro(model_path, voices_path)

        self._speech_token_queue: asyncio.Queue = event_bus.subscribe(EventType.LLM_SPEECH_TOKEN)
        self._is_cancelled: bool = False

        self._logger: logging.Logger = logging.getLogger("TTSClient")
        self._logger.info("Kokoro-ONNX TTS loaded (voice=%s)", voice_profile.voice_name)

    async def run(self) -> None:
        """Main loop — consume LLM_SPEECH_TOKEN events and synthesize audio."""
        # Register interrupt handler
        interrupt_queue: asyncio.Queue = self.event_bus.subscribe(EventType.INTERRUPT_DETECTED)

        async def _handle_interrupts() -> None:
            while True:
                await interrupt_queue.get()
                self._is_cancelled = True

        asyncio.create_task(_handle_interrupts())

        while True:
            event = await self._speech_token_queue.get()

            if self._is_cancelled:
                # Drain pending sentences
                while not self._speech_token_queue.empty():
                    try:
                        self._speech_token_queue.get_nowait()
                    except asyncio.QueueEmpty:
                        break
                self._is_cancelled = False
                continue

            sentence: str = event.data["text"]
            sentence_index: int = event.data["sentence_index"]

            # Strip LLM emotion tags — Kokoro doesn't use them
            sentence = EMOTION_TAG_PATTERN.sub("", sentence).strip()
            if not sentence:
                continue

            await self._synthesize_sentence(sentence, sentence_index)

    async def _synthesize_sentence(self, sentence: str, sentence_index: int) -> None:
        """Synthesize a single sentence via local Kokoro-ONNX and publish audio chunks."""
        self._logger.info(
            "TTS: synthesizing sentence %d: '%s'",
            sentence_index,
            sentence[:60] + ("..." if len(sentence) > 60 else ""),
        )

        try:
            # Run Kokoro in executor (CPU-bound, sync call)
            loop = asyncio.get_event_loop()
            samples, sample_rate = await loop.run_in_executor(
                None,
                lambda: self._kokoro.create(
                    sentence,
                    voice=self.voice_profile.voice_name,
                    speed=1.0,
                    lang="en-us",
                ),
            )

            # Convert numpy array → WAV bytes
            buf = io.BytesIO()
            sf.write(buf, samples, sample_rate, format="WAV")
            wav_bytes = buf.getvalue()

            # Split into chunks for incremental playback
            chunk_size = 4096
            for i in range(0, len(wav_bytes), chunk_size):
                if self._is_cancelled:
                    break
                chunk = wav_bytes[i : i + chunk_size]
                is_last = (i + chunk_size) >= len(wav_bytes)
                await self.event_bus.publish(
                    EventType.TTS_CHUNK_READY,
                    {
                        "audio": chunk,
                        "sentence_index": sentence_index,
                        "sentence_done": is_last,
                    },
                    source="TTSClient",
                )

            # Signal sentence completion
            await self.event_bus.publish(
                EventType.TTS_SENTENCE_DONE,
                {"sentence_index": sentence_index},
                source="TTSClient",
            )
            self._logger.info("TTS: sentence %d synthesized", sentence_index)

        except Exception as e:
            self._logger.error("TTS synthesis failed: %s", e)

    async def cancel(self) -> None:
        """Cancel current synthesis and drain pending sentences."""
        self._is_cancelled = True
        while not self._speech_token_queue.empty():
            try:
                self._speech_token_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        self._logger.info("TTS cancelled — clearing queue")

    async def synthesize_single(self, text: str) -> bytes:
        """One-off synthesis for safety refusals, etc. Returns complete WAV bytes."""
        # Strip emotion tags
        clean = EMOTION_TAG_PATTERN.sub("", text).strip()

        loop = asyncio.get_event_loop()
        samples, sample_rate = await loop.run_in_executor(
            None,
            lambda: self._kokoro.create(
                clean,
                voice=self.voice_profile.voice_name,
                speed=1.0,
                lang="en-us",
            ),
        )

        buf = io.BytesIO()
        sf.write(buf, samples, sample_rate, format="WAV")
        return buf.getvalue()