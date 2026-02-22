"""
╔══════════════════════════════════════════════════════════════════════════════════╗
║                   VOXCORE — input/speaking_monitor.py                            ║
║       GATE 3 — LLM-POWERED SEMANTIC INTERRUPT GATE (allam-2-7b)                ║
╚══════════════════════════════════════════════════════════════════════════════════╝

PURPOSE:
    The Speaking Monitor is the brain of the three-gate interrupt system.
    It runs ONLY during the SPEAKING state and classifies user speech into
    one of three outcomes:

        IGNORE    — Pure filler/reaction, AI continues uninterrupted
        INJECT    — Meaningful addition, silently inject context, AI continues
        INTERRUPT — Clear intent to stop/redirect, kill TTS, go to THINKING

    Gate 1 (duration + energy) is handled by InterruptionDetector which
    publishes GATE1_PASSED.  This module receives that event, runs Gate 2
    (filler word check via FillerDetector) and Gate 3 (whisper transcription
    via HF /transcribe/monitor + allam-2-7b classification via Groq).

    Uses HF_AUDIO_BASE_URL + /transcribe/monitor for fast short-clip STT.
    Uses Groq allam-2-7b for semantic classification (~150ms).

    Rate limit aware: debounce of 3s between Gate 3 calls.
    allam-2-7b: 30 RPM on free tier → ~5-10 calls per conversation is typical.
"""

import asyncio
import io
import logging
import os
import time
from collections import deque
from typing import Optional

import aiohttp
import numpy as np
from groq import AsyncGroq
from scipy.io import wavfile

from core.session import Session, TurnState
from core.event_bus import EventBus, EventType
from input.filler_detector import FillerDetector
from input.mic_stream import MicStream


# ═══════════════════════════════════════════════════════════════════════════════
# CLASSIFIER PROMPT
# ═══════════════════════════════════════════════════════════════════════════════

CLASSIFIER_SYSTEM_PROMPT = """\
You are a real-time speech intent classifier for a voice AI system.
The AI assistant is currently speaking. The user has said something.
Classify the user's intent as exactly one of three words:

IGNORE — The user is reacting naturally without wanting to take over.
  Examples: "yeah", "right", "uh-huh", "wow", "haha", "oh interesting",
            laughter, short affirmations, single word reactions.
  IMPORTANT: If the user's transcript contains words or phrases the AI is
  currently saying (acoustic echo/feedback from speakers), classify as IGNORE.
  Echo signs: transcript closely matches or is a fragment of the AI's words.

INJECT — The user added something meaningful but doesn't need the AI to stop.
  Examples: "oh and also...", "by the way...", "actually I forgot to mention",
            adding context, minor corrections that don't change the topic.

INTERRUPT — The user clearly wants the AI to stop and respond to them.
  Examples: "wait", "stop", "hold on", "no", "actually...", "I have a question",
            asking a new question, contradicting, expressing urgency.

Reply with ONLY one word: IGNORE, INJECT, or INTERRUPT"""


class SpeakingMonitor:
    """Gate 3 — LLM-powered semantic interrupt classifier.

    Activated during SPEAKING state only. Receives GATE1_PASSED events
    from InterruptionDetector, runs Gate 2 (filler check) and Gate 3
    (allam-2-7b classification). Fires INTERRUPT_DETECTED, publishes
    SOFT_INJECT context, or does nothing (IGNORE).

    Uses HF_AUDIO_BASE_URL/transcribe/monitor for fast short-clip STT
    and Groq allam-2-7b for semantic classification.
    """

    def __init__(
        self,
        session: Session,
        event_bus: EventBus,
        mic_stream: MicStream,
        filler_detector: FillerDetector,
        config: dict,
    ) -> None:
        self.session = session
        self.event_bus = event_bus
        self.mic_stream = mic_stream
        self.filler_detector = filler_detector

        # Config
        monitor_cfg = config.get("speaking_monitor", {})
        self._enabled: bool = monitor_cfg.get("enabled", True)
        self._model: str = monitor_cfg.get("model", "qwen/qwen3-32b")
        self._debounce_s: float = monitor_cfg.get("gate3_debounce_s", 3.0)
        self._audio_window_ms: int = monitor_cfg.get("gate3_audio_window_ms", 600)
        self._context_words: int = monitor_cfg.get("context_words", 30)

        # HF Audio endpoint for /transcribe/monitor
        self._hf_audio_base_url: str = os.environ.get("HF_AUDIO_BASE_URL", "")
        self._hf_audio_api_key: str = os.environ.get("HF_AUDIO_API_KEY", "")

        # Groq client for allam classification
        self._groq_client = AsyncGroq(api_key=os.environ.get("GROQ_API_KEY"))

        # Audio buffering for rolling window
        self._sample_rate: int = config.get("audio", {}).get("sample_rate", 16000)
        self._audio_buffer: deque = deque(maxlen=500)  # ~15s of 30ms chunks

        # State tracking
        self._is_active: bool = False
        self._last_classify_time: float = 0.0
        self._last_ai_speech: str = ""  # Rolling buffer of what the AI is saying
        self._audio_queue: Optional[asyncio.Queue] = None

        self._logger = logging.getLogger("SpeakingMonitor")

    async def run(self) -> None:
        """Main loop — listen for GATE1_PASSED events and run Gate 2 + 3."""
        if not self._enabled:
            self._logger.info("SpeakingMonitor disabled in config")
            return

        if not self._hf_audio_base_url:
            self._logger.warning(
                "HF_AUDIO_BASE_URL not set — SpeakingMonitor will use Groq whisper fallback"
            )

        # Subscribe to events
        gate1_queue = self.event_bus.subscribe(EventType.GATE1_PASSED)

        # Track AI speech for context — subscribe to LLM_SPEECH_TOKEN
        self.event_bus.subscribe(EventType.LLM_SPEECH_TOKEN, self._on_ai_speech)

        # Track state changes for activation/deactivation
        self.event_bus.subscribe(EventType.STATE_CHANGED, self._on_state_changed)

        # Get audio consumer for building rolling buffer
        self._audio_queue = self.mic_stream.add_consumer()
        asyncio.create_task(self._buffer_audio())

        self._logger.info(
            "SpeakingMonitor started (model=%s, debounce=%.1fs, window=%dms)",
            self._model, self._debounce_s, self._audio_window_ms,
        )

        while True:
            try:
                event = await gate1_queue.get()

                # Only process during SPEAKING
                if self.session.state != TurnState.SPEAKING:
                    continue

                if not self._is_active:
                    continue

                # Debounce: don't fire Gate 3 more than once per debounce period
                now = time.time()
                if now - self._last_classify_time < self._debounce_s:
                    self._logger.debug(
                        "Gate 3 debounced (%.1fs since last)",
                        now - self._last_classify_time,
                    )
                    continue

                # Run Gate 2 + 3 in background task
                asyncio.create_task(self._classify_interrupt(event.data))

            except asyncio.CancelledError:
                self._logger.info("SpeakingMonitor cancelled")
                break
            except Exception:
                self._logger.exception("Error in SpeakingMonitor loop")
                await asyncio.sleep(0.1)

    async def _buffer_audio(self) -> None:
        """Continuously buffer audio chunks for the rolling window."""
        while True:
            chunk = await self._audio_queue.get()
            if self._is_active:
                self._audio_buffer.append(chunk)

    async def _on_ai_speech(self, event) -> None:
        """Track what the AI is currently saying for classification context."""
        text = event.data.get("text", "")
        if text:
            # Keep a rolling buffer of the last N words
            words = self._last_ai_speech.split() + text.split()
            self._last_ai_speech = " ".join(words[-self._context_words:])

    async def _on_state_changed(self, event) -> None:
        """Activate/deactivate based on state transitions."""
        new_state = event.data.get("new_state", "")
        old_state = event.data.get("old_state", "")

        if new_state == TurnState.SPEAKING.value:
            # Don't clear buffers when returning from SOFT_INJECT → SPEAKING
            # (the AI is still mid-sentence, context should be preserved)
            if old_state == TurnState.SOFT_INJECT.value:
                self._logger.debug("SpeakingMonitor: SOFT_INJECT → SPEAKING (preserved buffers)")
            else:
                self._is_active = True
                self._audio_buffer.clear()
                self._last_ai_speech = ""
                self._logger.debug("SpeakingMonitor activated")

        elif new_state == TurnState.SOFT_INJECT.value:
            # Stay active during SOFT_INJECT — AI is still speaking
            self._logger.debug("SpeakingMonitor: SOFT_INJECT (staying active)")

        elif new_state in (
            TurnState.LISTENING.value,
            TurnState.THINKING.value,
            TurnState.INTERRUPTED.value,
        ):
            self._is_active = False
            self._logger.debug("SpeakingMonitor deactivated (%s)", new_state)

    # ─────────────────────────────────────────────────────────────────────
    # Classification pipeline (Gate 2 + Gate 3)
    # ─────────────────────────────────────────────────────────────────────

    async def _classify_interrupt(self, gate1_data: dict) -> None:
        """Run Gate 2 (filler check) and Gate 3 (LLM classification).

        CRITICAL: If any step fails, we fall back to firing INTERRUPT_DETECTED
        directly. Gate 1 already confirmed sustained user speech — silently
        swallowing the interrupt would make the system unresponsive.
        """
        try:
            # Bail if state changed since we started
            if self.session.state != TurnState.SPEAKING:
                return

            # ── Debounce: set IMMEDIATELY to prevent concurrent tasks ────
            self._last_classify_time = time.time()

            # ── Step 1: Get audio window for transcription ───────────────
            audio_window = self._get_audio_window()
            if audio_window is None or len(audio_window) < 100:
                self._logger.warning(
                    "Not enough audio for classification — fallback INTERRUPT"
                )
                await self._fallback_interrupt(gate1_data)
                return

            # ── Step 2: Transcribe via HF /transcribe/monitor ────────────
            transcript = await self._transcribe_short_clip(audio_window)
            if not transcript or not transcript.strip():
                # Empty transcript means STT couldn't recognize anything.
                # This is almost always TTS echo picked up by the mic —
                # real user speech would produce a transcript.
                # Treat as IGNORE, not INTERRUPT.
                self._logger.debug(
                    "Empty transcript from monitor STT — likely echo, IGNORE"
                )
                return

            self._logger.info("Monitor transcript: '%s'", transcript)

            # ── Gate 2: Filler word check ────────────────────────────────
            is_filler, matched_word = self.filler_detector.is_filler(transcript)
            if is_filler:
                self._logger.info(
                    "Gate 2: FILLER detected ('%s') — IGNORE",
                    matched_word or transcript,
                )
                # Note as implicit backchannel / positive reaction
                await self.event_bus.publish(
                    EventType.POSITIVE_REACTION,
                    {"text": transcript, "matched_word": matched_word},
                    source="SpeakingMonitor",
                )
                await self.event_bus.publish(
                    EventType.MONITOR_CLASSIFY,
                    {"classification": "IGNORE", "reason": "filler",
                     "transcript": transcript},
                    source="SpeakingMonitor",
                )
                return

            # ── Gate 3: allam-2-7b classification ────────────────────────
            classification = await self._classify_with_llm(transcript)

            self._logger.info(
                "Gate 3: %s (transcript='%s', ai_context='%s')",
                classification, transcript, self._last_ai_speech[-60:],
            )

            await self.event_bus.publish(
                EventType.MONITOR_CLASSIFY,
                {"classification": classification, "transcript": transcript,
                 "ai_context": self._last_ai_speech[-100:]},
                source="SpeakingMonitor",
            )

            # ── Stale check: state may have changed during async LLM call ─
            if self.session.state not in (
                TurnState.SPEAKING, TurnState.SOFT_INJECT,
            ):
                self._logger.info(
                    "Classification arrived but state is %s — discarding",
                    self.session.state.value,
                )
                return

            # ── Act on classification ────────────────────────────────────
            if classification == "IGNORE":
                await self.event_bus.publish(
                    EventType.POSITIVE_REACTION,
                    {"text": transcript},
                    source="SpeakingMonitor",
                )

            elif classification == "INJECT":
                # Soft inject: add to context silently, AI continues
                await self.session.inject_text(
                    content=f"[USER ADDED MID-SPEECH]: {transcript}",
                    priority="high",
                    source="user_mid_speech",
                )
                # Briefly transition to SOFT_INJECT to record the event
                await self.session.set_state(TurnState.SOFT_INJECT)
                # Return to SPEAKING immediately
                await self.session.set_state(TurnState.SPEAKING)
                self._logger.info("INJECT: '%s' added to context", transcript)

            elif classification == "INTERRUPT":
                # Full interrupt
                self._logger.info("INTERRUPT: killing TTS for '%s'", transcript)
                await self.event_bus.publish(
                    EventType.INTERRUPT_DETECTED,
                    {
                        "speech_prob": gate1_data.get("speech_prob", 0.0),
                        "during_sentence": -1,
                        "transcript": transcript,
                        "source": "speaking_monitor",
                    },
                    source="SpeakingMonitor",
                )

        except Exception:
            self._logger.exception(
                "Classification pipeline error — fallback INTERRUPT"
            )
            try:
                await self._fallback_interrupt(gate1_data)
            except Exception:
                self._logger.exception("Fallback interrupt also failed")

    async def _fallback_interrupt(self, gate1_data: dict) -> None:
        """Fire INTERRUPT_DETECTED directly when the classification pipeline
        fails. Gate 1 already confirmed the user was speaking — we must not
        silently swallow the interrupt."""
        self._logger.warning("Firing fallback INTERRUPT_DETECTED")
        await self.event_bus.publish(
            EventType.INTERRUPT_DETECTED,
            {
                "speech_prob": gate1_data.get("speech_prob", 0.0),
                "during_sentence": -1,
                "transcript": "",
                "source": "speaking_monitor_fallback",
            },
            source="SpeakingMonitor",
        )

    # ─────────────────────────────────────────────────────────────────────
    # Audio window extraction
    # ─────────────────────────────────────────────────────────────────────

    def _get_audio_window(self) -> Optional[np.ndarray]:
        """Extract the last audio_window_ms of buffered audio."""
        if not self._audio_buffer:
            return None

        # Calculate how many chunks we need
        chunk_ms = 30  # mic_stream default
        chunks_needed = max(1, self._audio_window_ms // chunk_ms)

        # Get the last N chunks
        available = list(self._audio_buffer)
        window_chunks = available[-chunks_needed:]

        if not window_chunks:
            return None

        return np.concatenate(window_chunks)

    # ─────────────────────────────────────────────────────────────────────
    # STT: HF /transcribe/monitor endpoint
    # ─────────────────────────────────────────────────────────────────────

    async def _transcribe_short_clip(self, audio: np.ndarray) -> str:
        """Transcribe a short audio clip via HF /transcribe/monitor endpoint.

        Falls back to Groq whisper if HF endpoint is not configured.
        """
        # Convert float32 audio to int16 PCM bytes
        audio_int16 = (audio * 32768.0).clip(-32768, 32767).astype(np.int16)
        pcm_bytes = audio_int16.tobytes()

        # Try HF endpoint first
        if self._hf_audio_base_url:
            try:
                return await self._transcribe_hf_monitor(pcm_bytes)
            except Exception:
                self._logger.warning(
                    "HF /transcribe/monitor failed, falling back to Groq"
                )

        # Fallback: Groq whisper
        return await self._transcribe_groq_short(pcm_bytes)

    async def _transcribe_hf_monitor(self, pcm_bytes: bytes) -> str:
        """Send PCM audio to the HF /transcribe/monitor endpoint."""
        url = f"{self._hf_audio_base_url.rstrip('/')}/transcribe/monitor"

        headers = {}
        if self._hf_audio_api_key:
            headers["Authorization"] = f"Bearer {self._hf_audio_api_key}"

        data = aiohttp.FormData()
        data.add_field(
            "audio",
            pcm_bytes,
            filename="monitor.pcm",
            content_type="application/octet-stream",
        )
        data.add_field("audio_format", "pcm")
        data.add_field("sample_rate", str(self._sample_rate))

        async with aiohttp.ClientSession() as http_session:
            async with http_session.post(
                url, data=data, headers=headers, timeout=aiohttp.ClientTimeout(total=5)
            ) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    self._logger.warning(
                        "HF monitor endpoint returned %d: %s", resp.status, text[:200]
                    )
                    return ""
                result = await resp.json()
                transcript = result.get("text", "").strip()
                ms = result.get("ms", 0)
                self._logger.debug(
                    "HF monitor STT: '%s' (%dms)", transcript, ms
                )
                return transcript

    async def _transcribe_groq_short(self, pcm_bytes: bytes) -> str:
        """Fallback: transcribe short clip via Groq whisper."""
        # Convert PCM bytes to WAV in memory
        audio_int16 = np.frombuffer(pcm_bytes, dtype=np.int16)
        buf = io.BytesIO()
        wavfile.write(buf, self._sample_rate, audio_int16)
        wav_bytes = buf.getvalue()

        try:
            file_tuple = ("monitor.wav", wav_bytes, "audio/wav")
            transcription = await self._groq_client.audio.transcriptions.create(
                file=file_tuple,
                model="whisper-large-v3-turbo",
                language="en",
                response_format="text",
            )
            if isinstance(transcription, str):
                return transcription.strip()
            return transcription.text.strip()
        except Exception:
            self._logger.exception("Groq short clip STT failed")
            return ""

    # ─────────────────────────────────────────────────────────────────────
    # LLM classification: allam-2-7b
    # ─────────────────────────────────────────────────────────────────────

    async def _classify_with_llm(self, transcript: str) -> str:
        """Classify user intent using allam-2-7b via Groq.

        Returns one of: 'IGNORE', 'INJECT', 'INTERRUPT'
        """
        # Build the user message with context
        ai_context = self._last_ai_speech[-100:] if self._last_ai_speech else "(just started speaking)"
        user_msg = f'Context:\nAI was saying: "{ai_context}"\nUser said: "{transcript}"\n\nReply with ONLY one word: IGNORE, INJECT, or INTERRUPT'

        try:
            response = await self._groq_client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": CLASSIFIER_SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.0,
                max_tokens=5,
                stream=False,
            )

            result = response.choices[0].message.content.strip().upper()

            # Validate — must be one of the three
            if result in ("IGNORE", "INJECT", "INTERRUPT"):
                return result

            # If the model returned something unexpected, try to parse
            for label in ("INTERRUPT", "INJECT", "IGNORE"):
                if label in result:
                    self._logger.warning(
                        "allam returned '%s', parsed as %s", result, label
                    )
                    return label

            # Default to IGNORE if we can't parse
            self._logger.warning(
                "allam returned unparseable '%s', defaulting to IGNORE", result
            )
            return "IGNORE"

        except Exception:
            self._logger.exception(
                "%s classification failed — defaulting to INTERRUPT "
                "(Gate 1 already confirmed sustained speech)",
                self._model,
            )
            # On error, default to INTERRUPT — Gate 1 confirmed the user was
            # speaking. Silently swallowing this as IGNORE would make the
            # system completely unresponsive to interrupts when the LLM is down.
            return "INTERRUPT"
