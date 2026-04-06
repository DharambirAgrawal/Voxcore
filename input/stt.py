"""
╔══════════════════════════════════════════════════════════════════════════════════╗
║                           VOXCORE — input/stt.py                                ║
║         GROQ WHISPER-TURBO STT — SPEECH-TO-TEXT WITH HF FALLBACK               ║
╚══════════════════════════════════════════════════════════════════════════════════╝
"""

import asyncio
import io
import logging
import os
import time

import aiohttp
import numpy as np
from groq import AsyncGroq, RateLimitError
from scipy.io import wavfile

from core.event_bus import EventBus, EventType


class STTClient:
    """Speech-to-Text client with Groq primary and HuggingFace fallback."""

    def __init__(self, event_bus: EventBus, config: dict) -> None:
        self.event_bus: EventBus = event_bus
        self.primary_model: str = config.get("models", {}).get(
            "stt_primary", "whisper-large-v3-turbo"
        )
        self.fallback_url: str = config.get("models", {}).get("stt_fallback", "")
        self.sample_rate: int = config.get("audio", {}).get("sample_rate", 16000)

        # Groq async client
        self._groq_client: AsyncGroq = AsyncGroq(
            api_key=os.environ.get("GROQ_API_KEY")
        )

        # HuggingFace API key for fallback
        self._hf_api_key: str = os.environ.get("HF_API_KEY", "")

        # Event subscription
        self._speech_end_queue: asyncio.Queue = event_bus.subscribe(
            EventType.SPEECH_END
        )

        # Rate limit tracking
        self._using_fallback: bool = False
        self._fallback_until: float = 0.0

        self._logger: logging.Logger = logging.getLogger("STT")

    async def run(self) -> None:
        """Process SPEECH_END events and produce transcripts forever."""
        while True:
            event = await self._speech_end_queue.get()
            audio_buffer: bytes = event.data["audio_buffer"]
            duration: float = event.data["duration_s"]

            # Skip very short utterances
            if duration < 0.3:
                self._logger.debug(
                    "Skipping short audio segment (%.2fs)", duration
                )
                continue

            wav_bytes = self._pcm_to_wav(audio_buffer)

            transcript = ""
            try:
                if not self._using_fallback or await self.check_groq_availability():
                    transcript = await self._transcribe_groq(wav_bytes)
                else:
                    transcript = await self._transcribe_fallback(wav_bytes)
            except RateLimitError:
                self._logger.warning(
                    "Groq STT rate limited, falling back to HF"
                )
                transcript = await self._transcribe_fallback(wav_bytes)
            except Exception:
                self._logger.exception("STT transcription error")
                transcript = await self._transcribe_fallback(wav_bytes)

            if not transcript or not transcript.strip():
                continue

            await self.event_bus.publish(
                EventType.TRANSCRIPT_READY,
                {
                    "text": transcript,
                    "confidence": 1.0,
                    "language": "en",
                },
            )
            self._logger.info("Transcript: '%s'", transcript)

    def _pcm_to_wav(self, pcm_bytes: bytes) -> bytes:
        """Convert raw PCM int16 bytes to a complete in-memory WAV file."""
        audio = np.frombuffer(pcm_bytes, dtype=np.int16)
        buf = io.BytesIO()
        wavfile.write(buf, self.sample_rate, audio)
        return buf.getvalue()

    async def _transcribe_groq(self, wav_bytes: bytes) -> str:
        """Transcribe audio via Groq whisper-large-v3-turbo API."""
        # Check if we're still in the rate-limit cooldown window
        if time.time() < self._fallback_until:
            raise RateLimitError(
                message="Still in rate-limit cooldown",
                response=None,  # type: ignore[arg-type]
                body=None,
            )

        try:
            file_tuple = ("audio.wav", wav_bytes, "audio/wav")
            transcription = await self._groq_client.audio.transcriptions.create(
                file=file_tuple,
                model=self.primary_model,
                language="en",
                response_format="text",
            )
            # response_format="text" returns a plain string
            if isinstance(transcription, str):
                return transcription.strip()
            return transcription.text.strip()

        except RateLimitError:
            self._using_fallback = True
            self._fallback_until = time.time() + 60
            self._logger.warning(
                "Groq STT rate limited, falling back to HF for 60s"
            )
            raise

    async def _transcribe_fallback(self, wav_bytes: bytes) -> str:
        """Transcribe audio via a HuggingFace hosted whisper endpoint."""
        if not self.fallback_url:
            self._logger.error("No fallback STT endpoint configured")
            return ""

        try:
            async with aiohttp.ClientSession() as session:
                headers = {"Authorization": f"Bearer {self._hf_api_key}"}
                data = aiohttp.FormData()
                data.add_field(
                    "file",
                    wav_bytes,
                    filename="audio.wav",
                    content_type="audio/wav",
                )
                async with session.post(
                    self.fallback_url, data=data, headers=headers
                ) as resp:
                    result = await resp.json()
                    return result.get("text", "").strip()
        except Exception:
            self._logger.exception("Fallback STT (HuggingFace) failed")
            return ""

    async def check_groq_availability(self) -> bool:
        """Check whether the Groq rate-limit cooldown has expired."""
        if time.time() > self._fallback_until:
            self._using_fallback = False
            return True
        return False