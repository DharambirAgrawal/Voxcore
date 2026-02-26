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
The AI assistant was speaking and paused because the user started talking.
Classify the user's intent as exactly one of three words:

IGNORE — The user is reacting passively without wanting to take over.
  Examples: "yeah", "right", "uh-huh", "wow", "haha", "oh interesting",
            laughter, brief affirmations mid-story (mm-hmm, cool).

INJECT — The user added something meaningful but doesn't need the AI to stop.
  Examples: "oh and also...", "by the way...", "actually I forgot to mention",
            adding context, minor corrections that don't change the topic.

INTERRUPT — The user wants the AI to stop and respond to something new.
  Examples: asking a question, changing the topic, contradicting,
            saying something that requires a new response from the AI.

RULE: If the user said something short and ambiguous, choose IGNORE.
RULE: If the user is clearly asking a question or making a statement that
      needs a response, choose INTERRUPT.

Reply with ONLY one word: IGNORE, INJECT, or INTERRUPT"""

# ═══════════════════════════════════════════════════════════════════════════════
# FAST STOP WORDS — these bypass the LLM entirely for instant response
# ═══════════════════════════════════════════════════════════════════════════════

# Words/phrases that ALWAYS mean "stop talking"
STOP_KEYWORDS: set[str] = {
    "stop", "wait", "hold on", "hold", "pause", "shut up", "be quiet",
    "enough", "quiet", "hush", "silence", "stop it", "stop stop",
    "stop stop stop", "okay stop", "ok stop", "please stop", "can you stop",
    "stop please", "that's enough", "ok enough", "okay enough",
    "stop talking", "hey stop", "no stop", "no no", "no no no",
}

# Words/phrases that are attention-getters = user wants to speak
ATTENTION_KEYWORDS: set[str] = {
    "hey", "hello", "excuse me", "listen", "wait wait",
    "actually", "but", "question", "i have a question",
    "can i", "let me", "what about", "how about",
    "thank you", "thanks", "okay thanks", "ok thanks",
    "that's great thanks", "alright thanks",
}


class SpeakingMonitor:
    """PersonaPlex-inspired interrupt classifier.

    When the user speaks during AI playback:
    1. PAUSE playback instantly (silence within 20ms)
    2. Collect audio for 600ms (clean mic, no echo)
    3. Transcribe with Whisper
    4. Fast keyword check → instant INTERRUPT for "stop/wait/hey"
    5. Echo + filler check → instant RESUME for false alarms
    6. LLM classify → INTERRUPT / INJECT / IGNORE(RESUME)

    This creates the natural "conversation pause" feeling — the AI stops
    talking when you start, listens, then either resumes or responds.
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
        self._model: str = monitor_cfg.get("model", "llama-3.1-8b-instant")
        self._debounce_s: float = monitor_cfg.get("gate3_debounce_s", 3.0)
        self._audio_window_ms: int = monitor_cfg.get("gate3_audio_window_ms", 2000)
        self._context_words: int = monitor_cfg.get("context_words", 30)
        self._collection_delay_ms: int = monitor_cfg.get("collection_delay_ms", 600)
        # HF Audio endpoint for /transcribe/monitor
        self._hf_audio_base_url: str = os.environ.get("HF_AUDIO_BASE_URL", "")
        self._hf_audio_api_key: str = os.environ.get("HF_AUDIO_API_KEY", "")

        # Groq client for classification
        self._groq_client = AsyncGroq(api_key=os.environ.get("GROQ_API_KEY"))

        # Audio buffering for rolling window
        self._sample_rate: int = config.get("audio", {}).get("sample_rate", 16000)
        self._audio_buffer: deque = deque(maxlen=500)  # ~15s of 30ms chunks

        # State tracking
        self._is_active: bool = False
        self._last_classify_time: float = 0.0
        self._last_ai_speech: str = ""  # Full AI response text for echo detection
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
            # Keep the entire AI response for the current turn to ensure
            # echo detection works even if TTS is lagging far behind LLM generation.
            self._last_ai_speech += text

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
    # PersonaPlex Classification Pipeline
    # ─────────────────────────────────────────────────────────────────────

    async def _classify_interrupt(self, gate1_data: dict) -> None:
        """PersonaPlex-inspired interrupt classification.

        Flow:
          1. PAUSE playback instantly (silence within 20ms)
          2. Wait 600ms for user to finish phrase (mic is clean — no echo)
          3. Transcribe with Whisper
          4. Fast keyword check → instant INTERRUPT for "stop/wait/hey"
          5. Echo check → instant RESUME for AI's own voice
          6. Filler check → instant RESUME for "uh-huh/yeah"
          7. LLM classify → INTERRUPT / INJECT / IGNORE(RESUME)

        The key insight: by PAUSING instead of DUCKING, we get:
        - Instant silence feedback (user feels heard in 20ms)
        - Clean mic signal (no AI audio contaminating the buffer)
        - Perfect STT transcription (no echo interference)
        - Natural conversation feel (like a human pausing to listen)
        """
        paused = False
        try:
            # Bail if state changed since we started
            if self.session.state != TurnState.SPEAKING:
                return

            # ── Debounce: set IMMEDIATELY to prevent concurrent tasks ────
            self._last_classify_time = time.time()

            # ── Step 1: PAUSE playback INSTANTLY ─────────────────────────
            await self.event_bus.publish(
                EventType.PLAYBACK_PAUSE,
                {"reason": "user_speaking"},
                source="SpeakingMonitor",
            )
            paused = True
            self._logger.debug("Playback paused — collecting user speech")

            # ── Step 2: Collection delay ─────────────────────────────────
            # Wait for user to finish their phrase. AI is SILENT so mic
            # gets a perfectly clean signal with zero echo contamination.
            delay_s = self._collection_delay_ms / 1000.0
            await asyncio.sleep(delay_s)

            # Bail if state changed during the wait
            if self.session.state != TurnState.SPEAKING:
                paused = False  # state handler took over
                return

            # ── Step 3: Transcribe ───────────────────────────────────────
            # Grab recent audio. Since AI was paused, this is pure user voice.
            audio_window = self._get_audio_window(ms_override=1200)
            if audio_window is None or len(audio_window) < 100:
                self._logger.warning(
                    "Not enough audio — fallback INTERRUPT"
                )
                await self._do_interrupt(gate1_data, "")
                paused = False
                return

            transcript = await self._transcribe_short_clip(audio_window)
            if not transcript or not transcript.strip():
                # Empty transcript = probably was echo from before the pause
                self._logger.debug(
                    "Empty transcript — likely pre-pause echo, RESUME"
                )
                await self._do_resume("empty_transcript")
                paused = False
                return

            self._logger.info("Monitor transcript: '%s'", transcript)

            # ── Step 4: Fast keyword check ───────────────────────────────
            # No LLM call needed — instant decision for common stop words.
            # This is the "say stop once and it stops" experience.
            normalised = transcript.strip().lower().rstrip(".!?,")
            if normalised in STOP_KEYWORDS:
                self._logger.info(
                    "STOP keyword detected ('%s') → instant INTERRUPT",
                    normalised,
                )
                await self._do_interrupt(gate1_data, transcript)
                paused = False
                return

            if normalised in ATTENTION_KEYWORDS:
                self._logger.info(
                    "ATTENTION keyword detected ('%s') → instant INTERRUPT",
                    normalised,
                )
                await self._do_interrupt(gate1_data, transcript)
                paused = False
                return

            # ── Step 5: Echo transcript check ────────────────────────────
            if self._is_echo_transcript(transcript):
                self._logger.info(
                    "Echo transcript detected ('%s') → RESUME", transcript,
                )
                await self._do_resume("echo_transcript")
                paused = False
                return

            # ── Step 6: Filler word check ────────────────────────────────
            is_filler, matched_word = self.filler_detector.is_filler(transcript)
            if is_filler:
                self._logger.info(
                    "FILLER detected ('%s') → RESUME",
                    matched_word or transcript,
                )
                await self.event_bus.publish(
                    EventType.POSITIVE_REACTION,
                    {"text": transcript, "matched_word": matched_word},
                    source="SpeakingMonitor",
                )
                await self._do_resume("filler")
                paused = False
                return

            # ── Step 7: LLM classification ───────────────────────────────
            # Only reaches here for ambiguous phrases that aren't keywords,
            # echo, or fillers. LLM decides the intent.
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

            # Stale check
            if self.session.state not in (
                TurnState.SPEAKING, TurnState.SOFT_INJECT,
            ):
                self._logger.info(
                    "Classification arrived but state is %s — discarding",
                    self.session.state.value,
                )
                paused = False
                return

            # ── Act on classification ────────────────────────────────────
            if classification == "IGNORE":
                await self._do_resume("ignore")
                paused = False

            elif classification == "INJECT":
                # Resume playback, but inject the user's words into context
                await self.session.inject_text(
                    content=f"[USER ADDED MID-SPEECH]: {transcript}",
                    priority="high",
                    source="user_mid_speech",
                )
                await self.session.set_state(TurnState.SOFT_INJECT)
                await self.session.set_state(TurnState.SPEAKING)
                self._logger.info("INJECT: '%s' added to context", transcript)
                await self._do_resume("inject")
                paused = False

            elif classification == "INTERRUPT":
                await self._do_interrupt(gate1_data, transcript)
                paused = False

        except Exception:
            self._logger.exception(
                "Classification pipeline error — fallback INTERRUPT"
            )
            try:
                await self._do_interrupt(gate1_data, "")
                paused = False
            except Exception:
                self._logger.exception("Fallback interrupt also failed")
        finally:
            # Safety net: always resume if we paused and didn't interrupt/resume
            if paused:
                try:
                    await self._do_resume("cleanup")
                except Exception:
                    pass

    # ─────────────────────────────────────────────────────────────────────
    # Action helpers
    # ─────────────────────────────────────────────────────────────────────

    async def _do_resume(self, reason: str) -> None:
        """Resume playback — AI continues from where it paused."""
        # Reset debounce from RESUME time, not from classify start.
        # Prevents echo from re-triggering Gate 3 immediately after resume.
        self._last_classify_time = time.time()
        await self.event_bus.publish(
            EventType.PLAYBACK_RESUME,
            {"reason": reason},
            source="SpeakingMonitor",
        )

    async def _do_interrupt(self, gate1_data: dict, transcript: str) -> None:
        """Full interrupt — kill TTS, go to INTERRUPTED state."""
        self._logger.info("INTERRUPT: killing TTS for '%s'", transcript or "(no transcript)")
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

    # ─────────────────────────────────────────────────────────────────────
    # Echo transcript detection
    # ─────────────────────────────────────────────────────────────────────

    def _is_echo_transcript(self, transcript: str) -> bool:
        """Check if the transcript is likely echo from the AI's speaker output.

        Compares words in the transcript against what the AI was recently
        saying. If >60% of transcript words appear in the AI's speech,
        it's almost certainly mic picking up speaker output — not the user.

        Examples that would be caught:
          AI says: "That's great, Dharambeer. How's your day going so far?"
          Transcript: "That's great." → 100% overlap → echo

          AI says: "I'd love to! Here's a little one."
          Transcript: "I'd love to." → 100% overlap → echo

          AI says: "Finley loved to collect shiny shells"
          Transcript: "to collect." → 100% overlap → echo
        """
        if not self._last_ai_speech:
            return False

        # Normalize: lowercase, strip punctuation
        import re as _re
        def _normalize(text: str) -> set[str]:
            words = _re.findall(r"[a-z']+", text.lower())
            # Filter out very short words that match anything (a, I, to)
            return {w for w in words if len(w) > 1}

        t_words = _normalize(transcript)
        ai_words = _normalize(self._last_ai_speech)

        if not t_words:
            return False

        overlap = t_words & ai_words
        ratio = len(overlap) / len(t_words)

        self._logger.debug(
            "Echo check: transcript='%s' overlap=%.0f%% (%s / %d words)",
            transcript, ratio * 100, overlap, len(t_words),
        )

        return ratio > 0.6

    # ─────────────────────────────────────────────────────────────────────
    # Audio window extraction
    # ─────────────────────────────────────────────────────────────────────

    def _get_audio_window(self, ms_override: Optional[int] = None) -> Optional[np.ndarray]:
        """Extract the last N ms of buffered audio.

        Args:
            ms_override: If set, use this many ms instead of audio_window_ms.
                         Used post-collection-delay to get a focused window.
        """
        if not self._audio_buffer:
            return None

        window_ms = ms_override if ms_override is not None else self._audio_window_ms
        # Calculate how many chunks we need
        chunk_ms = 30  # mic_stream default
        chunks_needed = max(1, window_ms // chunk_ms)

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
        """Classify user intent using LLM via Groq.

        Returns one of: 'IGNORE', 'INJECT', 'INTERRUPT'

        Handles thinking models (qwen3, etc.) that output <think> tags
        by stripping them before parsing the classification.
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
                max_tokens=50,
                stream=False,
            )

            raw_result = response.choices[0].message.content.strip()

            # Strip <think>...</think> tags from thinking models (qwen3, etc.)
            import re
            cleaned = re.sub(
                r"<think>.*?</think>",
                "",
                raw_result,
                flags=re.DOTALL | re.IGNORECASE,
            ).strip()
            # Also handle unclosed <think> tags (truncated thinking)
            cleaned = re.sub(
                r"<think>.*",
                "",
                cleaned,
                flags=re.DOTALL | re.IGNORECASE,
            ).strip()
            result = cleaned.upper() if cleaned else raw_result.upper()

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
