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
You are a real-time speech intent classifier for a voice AI.
The AI was speaking. It paused because the user spoke.
Classify the user's intent based on the transcript.

DECISION (choose ONE):
  IGNORE    — ONLY for reflexive 1-2 word sounds with no information:
              "haha", "wow", "oh", "hmm", "right", "yeah"
              These are involuntary reactions, not directed speech.
  INJECT    — user is adding something but does NOT want to stop the AI:
              Reactions: "I really liked that", "that's a good point"
              Deferred requests: "after this can you...", "when you're done tell me...",
              "after the story...", "also later can you...", "once you finish..."
              KEY: if the user says "after this" or "when you're done" before a request,
              they want the AI to CONTINUE what it's doing and handle it LATER.
              This is INJECT, NOT INTERRUPT.
  INTERRUPT — user wants the AI to STOP RIGHT NOW and respond:
              Immediate questions without deferral ("what is X?", "how do I...")
              Corrections, topic changes, or commands.
              No "after this"/"when done"/"later" qualifier present.

CRITICAL RULES:
- "after this", "after the story", "when you're done", "once you finish",
  "also can you later", "before you forget" → ALWAYS INJECT, never INTERRUPT.
  The user explicitly wants the AI to continue and handle the request afterward.
- If the transcript is 3+ words, it is almost NEVER IGNORE.
- Questions WITHOUT deferral ("can you tell me X" with no "after this") → INTERRUPT.
- Questions WITH deferral ("after this can you tell me X") → INJECT.
- Positive reactions with content ("I liked that", "good story") → INJECT.
- Only bare sounds/words with zero content → IGNORE.

If INTERRUPT, also classify TYPE:
  STOP              — wants silence, no response
  PAUSE             — brief hold, AI content may resume
  CORRECTION        — fixing something AI said
  NEW_QUESTION      — new unrelated topic
  SAME_TOPIC_REDIRECT — follow-up on current topic
  URGENCY           — safety, distress, critical issue
  DONE_LISTENING    — user has heard enough

Reply ONLY in this format (two lines):
DECISION: IGNORE|INJECT|INTERRUPT
TYPE: STOP|PAUSE|CORRECTION|NEW_QUESTION|SAME_TOPIC_REDIRECT|URGENCY|DONE_LISTENING|null"""

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
# V3 FIX: Removed multi-word phrases that overlap with deferred requests.
# Only truly unambiguous single-word attention-getters remain.
# Multi-word transcripts go through the LLM classifier for proper
# deferral detection ("after this can you..." = INJECT not INTERRUPT).
ATTENTION_KEYWORDS: set[str] = {
    "hey", "hello", "excuse me", "listen", "wait wait",
    "question", "i have a question",
}

# Prefixes: if transcript STARTS WITH any of these, it's attention/question.
# V3 FIX: Only kept prefixes that are NEVER used in deferred requests.
# Removed "can you", "could you", "tell me", "what about", etc. because
# those commonly appear WITH deferral ("after this can you...").
# The LLM classifier handles the nuance of immediate vs deferred.
ATTENTION_PREFIXES: tuple[str, ...] = (
    "excuse me",
)

# Deferral phrases — if ANY of these appear in the transcript, the user
# wants the AI to continue and handle the request LATER.  Forces INJECT.
DEFERRAL_PHRASES: tuple[str, ...] = (
    "after this", "after the story", "after that", "after you finish",
    "after you're done", "when you're done", "when you finish",
    "once you're done", "once you finish", "once this is done",
    "later can you", "also later", "before you forget",
    "remind me to", "remind me after", "don't forget to",
    "but first finish", "keep going but", "continue but",
)


class SpeakingMonitor:
    """PersonaPlex-inspired interrupt classifier (v3).

    When the user speaks during AI playback:
    1. Enter PENDING state (TTS fully paused, filters OFF)
    2. Play pre-pause clip ("mm" from clips_dir)
    3. Collect audio (clean mic, no echo — AI is silent)
    4. Fast keyword check → instant INTERRUPT for "stop/wait/hey"
    5. Echo + filler check → instant RESUME for false alarms
    6. LLM classify (Gate 3) → CLASSIFIED event → InterruptRouter
    7. VAD-end action: wait for user silence before routing

    V3 changes from v2:
    - PENDING state: full TTS pause (not ducked)
    - CLASSIFIED event: published to EventBus for InterruptRouter routing
    - Two-path Gate 1: Path A (burst) skips filler check, Path B goes full pipeline
    - VAD-end action: 300ms silence before acting on classification
    - Gate 3 timeout: max 3s wait before default PAUSE
    """

    def __init__(
        self,
        session: Session,
        event_bus: EventBus,
        mic_stream: MicStream,
        filler_detector: FillerDetector,
        config: dict,
        audio_player=None,
        gate3_done_event: Optional[asyncio.Event] = None,
    ) -> None:
        self.session = session
        self.event_bus = event_bus
        self.mic_stream = mic_stream
        self.filler_detector = filler_detector
        self._audio_player = audio_player       # direct ref for pause/resume
        self._gate3_done_event = gate3_done_event  # stops filler clip

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

        # Gate 3 timeout
        self._gate3_timeout_s: float = monitor_cfg.get("gate3_timeout_s", 3.0)
        # VAD-end silence wait
        self._vad_silence_ms: int = monitor_cfg.get("vad_interrupt_silence_ms", 300)
        # Re-classify interval (samples) — every 1.5s of audio
        self._reclassify_samples: int = monitor_cfg.get("gate3_reclassify_every_samples", 24000)
        # Pre-pause clip from config
        bc_cfg = config.get("backchannel", {})
        self._clips_dir: str = bc_cfg.get("clips_dir", "backchannel/clips/heart/")
        self._pre_pause_clip: str = monitor_cfg.get("pre_pause_clip", "mm_hmm.wav")

        # HF endpoint failure counter — skip HF after consecutive failures
        self._hf_consecutive_failures: int = 0
        self._hf_max_failures: int = 1  # after 1 failure, go straight to Groq

        # Pre-pause chunk count — used to prefer clean post-pause audio
        self._pre_pause_chunk_count: int = 0

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

                # Only process during SPEAKING (or PENDING for re-triggers)
                if self.session.state not in (TurnState.SPEAKING, TurnState.PENDING):
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

                # Get Gate 1 path info
                gate1_path = event.data.get("path", "B")

                # Run classification in background task
                asyncio.create_task(
                    self._classify_interrupt(event.data, gate1_path=gate1_path)
                )

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
            if old_state == TurnState.SOFT_INJECT.value:
                self._logger.debug("SpeakingMonitor: SOFT_INJECT → SPEAKING (preserved buffers)")
            elif old_state == TurnState.PAUSED.value:
                # Resuming from PAUSED — keep buffers, stay active
                self._logger.debug("SpeakingMonitor: PAUSED → SPEAKING (resumed)")
            else:
                self._is_active = True
                self._audio_buffer.clear()
                self._last_ai_speech = ""
                self._logger.debug("SpeakingMonitor activated")

        elif new_state == TurnState.SOFT_INJECT.value:
            self._logger.debug("SpeakingMonitor: SOFT_INJECT (staying active)")

        elif new_state == TurnState.PAUSED.value:
            # PAUSED — stay active (we may need to re-classify)
            self._logger.debug("SpeakingMonitor: PAUSED (staying active)")

        elif new_state == TurnState.PENDING.value:
            # PENDING — classification in progress
            self._logger.debug("SpeakingMonitor: PENDING (classifying)")

        elif new_state in (
            TurnState.LISTENING.value,
            TurnState.THINKING.value,
            TurnState.INTERRUPTED.value,
        ):
            self._is_active = False
            # If leaving PENDING/PAUSED, ensure filler clip stops
            if old_state in (TurnState.PENDING.value, TurnState.PAUSED.value):
                if self._gate3_done_event is not None:
                    self._gate3_done_event.set()
                    self._logger.debug("gate3_done_event set (state left PENDING/PAUSED)")
            self._logger.debug("SpeakingMonitor deactivated (%s)", new_state)

    # ─────────────────────────────────────────────────────────────────────
    # PersonaPlex Classification Pipeline
    # ─────────────────────────────────────────────────────────────────────

    async def _classify_interrupt(
        self, gate1_data: dict, gate1_path: str = "B",
    ) -> None:
        """V3 PersonaPlex-inspired interrupt classification.

        Flow:
          1. Enter PENDING state (TTS fully paused, filters OFF)
          2. Play pre-pause filler clip (breath/mm) instantly
          3. Run Gate 3 classification in background
          4. Fast keyword check → instant INTERRUPT for "stop/wait/hey"
          5. Echo check → instant RESUME
          6. Path A: skip filler check (burst = intentional)
             Path B: filler check → instant RESUME
          7. LLM classify (Gate 3) with timeout
          8. VAD-end action: wait for user silence with re-classification
          9. Publish CLASSIFIED event → InterruptRouter
        """
        paused = False
        try:
            if self.session.state not in (TurnState.SPEAKING, TurnState.PENDING):
                return

            self._last_classify_time = time.time()

            # ── Step 1: PAUSE playback → enter PENDING ───────────────────
            # Call audio_player.pause() DIRECTLY for ~3ms response
            if self._audio_player is not None:
                await self._audio_player.pause()
            else:
                # Fallback: event-based pause
                await self.event_bus.publish(
                    EventType.PLAYBACK_PAUSE,
                    {"reason": "user_speaking"},
                    source="SpeakingMonitor",
                )

            # V3 FIX: Keep the LAST ~1s of audio (the speech that triggered
            # Gate 1) but discard everything older.  The old approach cleared
            # the entire buffer, which threw away the user's actual utterance
            # because Gate 1 fires 600-700ms AFTER speech starts — the user
            # may already be finishing "can you stop" by the time we get here.
            # Keeping ~1s preserves that speech; the echo-transcript check
            # handles any residual AI audio that leaks through.
            _keep_ms = 1200  # keep last 1.2 s (covers Gate 1 window + margin)
            _keep_chunks = max(1, _keep_ms // 30)  # ~40 chunks at 30 ms each
            _kept = list(self._audio_buffer)[-_keep_chunks:]
            self._audio_buffer.clear()
            self._audio_buffer.extend(_kept)
            self._pre_pause_chunk_count = len(_kept)  # mark boundary
            self._logger.debug(
                "Buffer trimmed: kept %d chunks (~%dms) of pre-Gate1 speech",
                len(_kept), len(_kept) * 30,
            )

            # Set session state to PENDING
            await self.session.set_state(TurnState.PENDING)
            paused = True

            # Play pre-pause filler clip IMMEDIATELY (~3ms from Gate 2 pass)
            # This gives the user instant audio feedback while Gate 3 runs
            if self._gate3_done_event is not None:
                self._gate3_done_event.clear()
            clip_path = os.path.join(self._clips_dir, self._pre_pause_clip)
            if self._audio_player is not None and os.path.isfile(clip_path):
                # Play filler clip exactly once
                asyncio.create_task(self._audio_player.play_clip(clip_path))

            # Publish PENDING event for WebSocket clients
            await self.event_bus.publish(
                EventType.PENDING,
                {"gate1_path": gate1_path, "urgency_hint": gate1_path == "A"},
                source="SpeakingMonitor",
            )

            self._logger.debug(
                "PENDING — playback paused, filler clip playing, collecting user speech (path=%s)",
                gate1_path,
            )

            # ── Step 2: Collection delay ─────────────────────────────────
            # V3 FIX: Reduced from config value (600ms) to 350ms.
            # The longer speech-end wait (Step 2b) compensates — this gets
            # us to transcription faster for short commands while still
            # capturing full utterances via the adaptive silence detector.
            delay_s = min(self._collection_delay_ms / 1000.0, 0.35)
            await asyncio.sleep(delay_s)

            if self.session.state not in (
                TurnState.SPEAKING, TurnState.PENDING, TurnState.PAUSED,
            ):
                paused = False
                return

            # ── Step 2b: Wait for user to finish speaking ────────────────
            # V3 FIX: No hard max_wait — the user can speak as long as they
            # want.  We just wait for 1000ms of contiguous silence so users
            # can breathe or pause naturally between phrases ("after this...
            # [breath] ...can you tell me...") without getting cut off.
            # Previous 700ms was clipping at natural inter-phrase pauses.
            # The 10s safety cap only triggers in pathological cases.
            await self._wait_for_speech_end(max_wait_s=10.0, silence_ms=1000)

            if self.session.state not in (
                TurnState.SPEAKING, TurnState.PENDING, TurnState.PAUSED,
            ):
                paused = False
                return

            # ── Step 3: Transcribe ───────────────────────────────────────
            # V3 FIX: Use full available buffer (up to 10s) since the user
            # can now speak as long as they want.  The smart windowing in
            # _get_audio_window() already prefers clean post-pause audio.
            audio_window = self._get_audio_window(ms_override=10000)
            if audio_window is None or len(audio_window) < 100:
                self._logger.warning("Not enough audio — fallback INTERRUPT")
                await self._publish_classified("INTERRUPT", "STOP", "", gate1_data)
                paused = False
                return

            transcript = await self._transcribe_short_clip(audio_window)
            
            decision = "IGNORE"
            interrupt_type = "null"
            urgency_hint = (gate1_path == "A")

            if not transcript or not transcript.strip():
                # V3 FIX: Don't resume immediately on empty transcript if user is still making noise.
                # Treat as default IGNORE and let VAD (Step 8) wait for them to finish, where it might re-trigger STT.
                self._logger.warning("Empty transcript from STT — treating as IGNORE (waiting for VAD)")
                transcript = ""
            else:
                self._logger.info("Monitor transcript: '%s' (path=%s)", transcript, gate1_path)

                # ── Step 4: Fast keyword check ───────────────────────────────
                normalised = transcript.strip().lower().rstrip(".!?,")

                # Check for deferral phrases FIRST — these always mean INJECT
                has_deferral = any(d in normalised for d in DEFERRAL_PHRASES)

                # Exact match OR any multi-word stop phrase is a substring
                # (catches "hey can you stop" matching "can you stop" from the set).
                # Single-word keywords are only checked exact — avoids matching
                # "stopped", "waiting", etc.
                # Guard: if has_deferral, treat as deferred INJECT not immediate stop.
                _is_stop = (not has_deferral) and (
                    normalised in STOP_KEYWORDS or
                    any(kw in normalised for kw in STOP_KEYWORDS if " " in kw)
                )
                if _is_stop:
                    self._logger.info("STOP keyword → instant INTERRUPT")
                    await self._publish_classified(
                        "INTERRUPT", "STOP", transcript, gate1_data,
                    )
                    paused = False
                    return

                # If deferral phrase detected, skip keyword bypass → let LLM classify
                # "after this can you tell me bitcoin price" should be INJECT, not INTERRUPT
                if has_deferral:
                    self._logger.info(
                        "Deferral phrase detected → skipping keyword check, using LLM"
                    )
                elif normalised in ATTENTION_KEYWORDS or normalised.startswith(ATTENTION_PREFIXES):
                    self._logger.info("ATTENTION keyword/prefix → instant INTERRUPT")
                    await self._publish_classified(
                        "INTERRUPT", "NEW_QUESTION", transcript, gate1_data,
                    )
                    paused = False
                    return

                # ── Step 5: Echo transcript check ────────────────────────────
                if self._is_echo_transcript(transcript):
                    self._logger.info("Echo transcript → RESUME")
                    await self._do_resume("echo_transcript")
                    paused = False
                    return

                # ── Step 6: Filler word check ────────────────────────────────
                # Path A (HIGH_ENERGY_BURST) skips filler check
                if gate1_path != "A":
                    is_filler, matched_word = self.filler_detector.is_filler(transcript)
                    if is_filler:
                        self._logger.info("FILLER → RESUME")
                        await self.event_bus.publish(
                            EventType.POSITIVE_REACTION,
                            {"text": transcript, "matched_word": matched_word},
                            source="SpeakingMonitor",
                        )
                        await self._do_resume("filler")
                        paused = False
                        return

                # ── Step 7: LLM classification (Gate 3) with timeout ─────────
                try:
                    decision, interrupt_type = await asyncio.wait_for(
                        self._classify_with_llm(
                            transcript,
                            speech_position="mid_sentence",
                            urgency_hint=urgency_hint,
                        ),
                        timeout=self._gate3_timeout_s,
                    )
                except asyncio.TimeoutError:
                    self._logger.warning(
                        "Gate 3 timeout (%.1fs) — default INTERRUPT/PAUSE",
                        self._gate3_timeout_s,
                    )
                    decision, interrupt_type = "INTERRUPT", "PAUSE"

                # Safety net: if LLM returned IGNORE but we already confirmed a
                # deferral phrase ("after this / when you're done / ..."), the
                # model either returned empty or misclassified.  Force INJECT so
                # the deferred request is not silently dropped.
                if decision == "IGNORE" and has_deferral:
                    self._logger.warning(
                        "LLM returned IGNORE but deferral phrase confirmed — overriding to INJECT"
                    )
                    decision, interrupt_type = "INJECT", "null"

                self._logger.info(
                    "Gate 3: decision=%s type=%s (transcript='%s')",
                    decision, interrupt_type, transcript,
                )

            # ── Step 8: VAD-end action — wait for user silence ───────────
            # Re-classify every 1.5s if user keeps speaking
            decision, interrupt_type = await self._wait_for_user_to_finish(
                transcript, gate1_data, decision, interrupt_type,
                urgency_hint=urgency_hint,
            )

            # ── Step 9: Publish CLASSIFIED event ─────────────────────────
            if self.session.state not in (
                TurnState.SPEAKING, TurnState.SOFT_INJECT,
                TurnState.PENDING, TurnState.PAUSED,
            ):
                self._logger.info(
                    "State changed to %s — discarding classification",
                    self.session.state.value,
                )
                paused = False
                return

            # Route based on decision
            if decision == "IGNORE":
                # Play a soft acknowledgment so user feels heard
                if self._audio_player is not None:
                    ack_clip = os.path.join(self._clips_dir, "got_it.wav")
                    if os.path.exists(ack_clip):
                        try:
                            await self._audio_player.play_clip(ack_clip)
                        except Exception:
                            pass  # Don't block resume on clip failure
                await self._do_resume("ignore")
                paused = False
            elif decision == "INJECT":
                await self._publish_classified(
                    "INJECT", "SAME_TOPIC_REDIRECT", transcript, gate1_data,
                    deferred=has_deferral,
                )
                paused = False
            elif decision == "INTERRUPT":
                await self._publish_classified(
                    "INTERRUPT", interrupt_type, transcript, gate1_data,
                )
                paused = False

        except Exception:
            self._logger.exception("Classification pipeline error — fallback INTERRUPT")
            try:
                await self._publish_classified(
                    "INTERRUPT", "STOP", "", gate1_data,
                )
                paused = False
            except Exception:
                self._logger.exception("Fallback also failed")
        finally:
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
        self._last_classify_time = time.time()
        # Stop the filler clip first
        if self._gate3_done_event is not None:
            self._gate3_done_event.set()
        # Resume via direct call (faster) or event fallback
        if self._audio_player is not None:
            await self._audio_player.resume()
        else:
            await self.event_bus.publish(
                EventType.PLAYBACK_RESUME,
                {"reason": reason},
                source="SpeakingMonitor",
            )
        # Return to SPEAKING state
        await self.session.set_state(TurnState.SPEAKING)

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

    # ── CLASSIFIED event publishing ──────────────────────────────────

    async def _publish_classified(
        self,
        decision: str,
        interrupt_type: str,
        transcript: str,
        gate1_data: dict,
        deferred: bool = False,
    ) -> None:
        """V3: Publish a CLASSIFIED event for InterruptRouter to handle."""
        await self.event_bus.publish(
            EventType.CLASSIFIED,
            {
                "decision": decision,
                "interrupt_type": interrupt_type,
                "transcript": transcript,
                "speech_prob": gate1_data.get("speech_prob", 0.0),
                "rms": gate1_data.get("rms", 0.0),
                "path": gate1_data.get("path", "B"),
                "ai_context": self._last_ai_speech[-100:],
                "deferred": deferred,
            },
            source="SpeakingMonitor",
        )
        self._logger.info(
            "CLASSIFIED: decision=%s type=%s transcript='%s'",
            decision, interrupt_type, transcript,
        )

    async def _wait_for_speech_end(
        self, max_wait_s: float = 1.8, silence_ms: int = 200
    ) -> None:
        """Block until the mic has been silent for `silence_ms` ms or `max_wait_s` elapses.

        Called after the initial collection delay so we capture the user's
        full utterance before running filler / LLM classification.
        Polls every 50ms; exits as soon as a contiguous silence window is seen.
        """
        silence_threshold_s = silence_ms / 1000.0
        check_interval = 0.05
        elapsed = 0.0
        silence_duration = 0.0

        while elapsed < max_wait_s:
            recent = self.mic_stream.last_200ms()
            if len(recent) > 0:
                rms = float(np.sqrt(np.mean(recent ** 2)))
                noise_floor = self.mic_stream.noise_floor
                is_silent = rms < noise_floor * 2.5
            else:
                is_silent = True

            if is_silent:
                silence_duration += check_interval
                if silence_duration >= silence_threshold_s:
                    self._logger.debug(
                        "Speech-end VAD: %.0fms silence after %.0fms total",
                        silence_duration * 1000, elapsed * 1000,
                    )
                    return
            else:
                silence_duration = 0.0

            await asyncio.sleep(check_interval)
            elapsed += check_interval

        self._logger.debug(
            "Speech-end VAD: max_wait %.1fs reached without silence — proceeding",
            max_wait_s,
        )

    async def _wait_for_user_to_finish(
        self,
        transcript: str,
        gate1_data: dict,
        decision: str,
        interrupt_type: str,
        urgency_hint: bool = False,
    ) -> tuple[str, str]:
        """V3: Wait for user silence before acting on classification.

        While waiting, accumulates audio. Every 1.5s of continuous speech,
        re-transcribes and re-classifies (user may have changed intent).

        Returns the (potentially updated) (decision, interrupt_type) tuple.
        """
        silence_threshold_s = self._vad_silence_ms / 1000.0  # 300ms
        check_interval = 0.05  # 50ms
        max_wait = 3.0  # max total wait for silence
        reclassify_interval_s = self._reclassify_samples / self._sample_rate  # 1.5s

        elapsed = 0.0
        time_since_reclassify = 0.0
        silence_duration = 0.0

        while elapsed < max_wait:
            # Check recent audio energy
            recent = self.mic_stream.last_200ms()
            if len(recent) > 0:
                rms = float(np.sqrt(np.mean(recent ** 2)))
                noise_floor = self.mic_stream.noise_floor
                is_silent = rms < noise_floor * 2.0
            else:
                is_silent = True

            if is_silent:
                silence_duration += check_interval
                if silence_duration >= silence_threshold_s:
                    # User has been silent long enough — act on current classification
                    self._logger.debug(
                        "VAD-end: %.0fms silence — proceeding with %s/%s",
                        silence_duration * 1000, decision, interrupt_type,
                    )
                    return decision, interrupt_type
            else:
                silence_duration = 0.0
                time_since_reclassify += check_interval

            # Re-classify every 1.5s if user keeps speaking
            if time_since_reclassify >= reclassify_interval_s:
                time_since_reclassify = 0.0
                self._logger.debug("Re-classifying after %.1fs of continued speech", reclassify_interval_s)

                # Get fresh audio and re-transcribe
                audio_window = self._get_audio_window(ms_override=2000)
                if audio_window is not None and len(audio_window) > 100:
                    new_transcript = await self._transcribe_short_clip(audio_window)
                    if new_transcript and new_transcript.strip():
                        transcript = new_transcript
                        try:
                            new_decision, new_type = await asyncio.wait_for(
                                self._classify_with_llm(
                                    transcript,
                                    speech_position="mid_sentence",
                                    urgency_hint=urgency_hint,
                                ),
                                timeout=2.0,
                            )
                            decision, interrupt_type = new_decision, new_type
                            self._logger.info(
                                "Re-classified: %s/%s (transcript='%s')",
                                decision, interrupt_type, transcript,
                            )
                        except asyncio.TimeoutError:
                            self._logger.debug("Re-classify timed out — keeping previous result")

            await asyncio.sleep(check_interval)
            elapsed += check_interval

        self._logger.debug("VAD-end: max wait %.1fs reached — proceeding", max_wait)
        return decision, interrupt_type

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
        """Extract audio from the rolling buffer, preferring post-pause audio.

        The buffer contains two regions after Gate 1:
          [0 .. _pre_pause_chunk_count)  — recorded while TTS was playing (echo-contaminated)
          [_pre_pause_chunk_count .. end) — recorded after TTS paused (clean)

        Strategy:
          • If we have >= 500ms of post-pause audio AND it contains actual speech
            (RMS > noise_floor × 1.5), use ONLY post-pause chunks (up to ms_override).
          • Otherwise fall back to the full buffer. The critical case is when the
            user finished speaking BEFORE Gate 1 fired — the post-pause chunks are
            then just silence, and using them gives STT nothing to work with.
            The echo-transcript check downstream catches pure AI echo.
        """
        if not self._audio_buffer:
            return None

        window_ms = ms_override if ms_override is not None else self._audio_window_ms
        chunk_ms = 30  # mic_stream default
        chunks_needed = max(1, window_ms // chunk_ms)

        available = list(self._audio_buffer)
        pre_count = min(self._pre_pause_chunk_count, len(available))
        post_chunks = available[pre_count:]
        post_ms = len(post_chunks) * chunk_ms

        if post_ms >= 500:
            # Check if post-pause chunks actually contain speech or are just
            # silence accumulated during the _wait_for_speech_end wait period.
            noise_floor = self.mic_stream.noise_floor
            speech_threshold = noise_floor * 1.5 if noise_floor > 0 else 0.005
            post_audio = np.concatenate(post_chunks) if post_chunks else np.array([])
            post_rms = float(np.sqrt(np.mean(post_audio ** 2))) if len(post_audio) > 0 else 0.0
            has_speech = post_rms > speech_threshold

            if has_speech:
                # Enough clean post-pause audio — use only that
                window_chunks = post_chunks[-chunks_needed:]
                self._logger.debug(
                    "Audio window: using %d post-pause chunks (%dms), skipping %d pre-pause (rms=%.4f)",
                    len(window_chunks), len(window_chunks) * chunk_ms, pre_count, post_rms,
                )
            else:
                # Post-pause is silence (user finished speaking before Gate 1 fired).
                # Fall back to full buffer to capture the pre-pause speech utterance.
                window_chunks = available[-chunks_needed:]
                self._logger.debug(
                    "Audio window: post-pause silent (rms=%.4f < %.4f) — using full buffer %d chunks (%dms)",
                    post_rms, speech_threshold, len(window_chunks), len(window_chunks) * chunk_ms,
                )
        else:
            # Short command case — include pre-pause buffer
            window_chunks = available[-chunks_needed:]
            self._logger.debug(
                "Audio window: using full buffer %d chunks (%dms) (post-pause only %dms)",
                len(window_chunks), len(window_chunks) * chunk_ms, post_ms,
            )

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

        # Try HF endpoint first — but skip if it has failed too many times
        if self._hf_audio_base_url and self._hf_consecutive_failures < self._hf_max_failures:
            try:
                result = await self._transcribe_hf_monitor(pcm_bytes)
                self._hf_consecutive_failures = 0  # reset on success
                return result
            except Exception:
                self._hf_consecutive_failures += 1
                self._logger.warning(
                    "HF /transcribe/monitor failed (%d/%d), falling back to Groq",
                    self._hf_consecutive_failures, self._hf_max_failures,
                )
        elif self._hf_consecutive_failures >= self._hf_max_failures:
            self._logger.debug(
                "HF skipped (%d consecutive failures) — using Groq directly",
                self._hf_consecutive_failures,
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
            # timeout slashed to 1.5s to prevent massive 5-second hangs on HF failures
            async with http_session.post(
                url, data=data, headers=headers, timeout=aiohttp.ClientTimeout(total=1.5)
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

    async def _classify_with_llm(
        self, transcript: str, speech_position: str = "mid_sentence",
        urgency_hint: bool = False,
    ) -> tuple[str, str]:
        """V3: Classify user intent using LLM via Groq.

        Returns (decision, interrupt_type) tuple:
          decision: 'IGNORE', 'INJECT', or 'INTERRUPT'
          interrupt_type: 'STOP', 'PAUSE', 'CORRECTION', 'NEW_QUESTION',
                         'SAME_TOPIC_REDIRECT', 'URGENCY', 'DONE_LISTENING', or 'null'

        Sends context including AI speech, speech_position, and urgency_hint.
        Parses two-line "DECISION: X\nTYPE: Y" response format.
        """
        import re

        # Build rich context for the classifier
        ai_context = self._last_ai_speech[-200:] if self._last_ai_speech else "(just started speaking)"
        urgency_str = "yes — Path A burst detected" if urgency_hint else "no"

        user_msg = (
            f'Context:\n'
            f'AI was saying: "{ai_context}"\n'
            f'User said: "{transcript}"\n'
            f'Speech position: {speech_position}\n'
            f'Urgency hint: {urgency_str}\n\n'
            f'Reply ONLY in the two-line format:\n'
            f'DECISION: IGNORE|INJECT|INTERRUPT\n'
            f'TYPE: STOP|PAUSE|CORRECTION|NEW_QUESTION|SAME_TOPIC_REDIRECT|URGENCY|DONE_LISTENING|null'
        )

        try:
            raw_result = ""
            # Retry once on empty response — cold endpoints (GPT-4o-mini, etc.)
            # occasionally return blank on the first call.
            for attempt in range(2):
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
                if raw_result:
                    break
                if attempt == 0:
                    self._logger.warning(
                        "Classifier returned empty (attempt 1/2) — retrying after 300ms"
                    )
                    await asyncio.sleep(0.3)

            # Strip <think>...</think> tags from thinking models (qwen3, etc.)
            cleaned = re.sub(
                r"<think>.*?</think>", "", raw_result,
                flags=re.DOTALL | re.IGNORECASE,
            ).strip()
            cleaned = re.sub(
                r"<think>.*", "", cleaned,
                flags=re.DOTALL | re.IGNORECASE,
            ).strip()
            if not cleaned:
                cleaned = raw_result

            # Parse two-line DECISION+TYPE format
            decision, itype = self._parse_decision_type(cleaned)

            self._logger.info(
                "LLM raw='%s' → decision=%s type=%s",
                cleaned.replace('\n', ' | '), decision, itype,
            )
            return decision, itype

        except Exception:
            self._logger.exception(
                "%s classification failed — defaulting to INTERRUPT/STOP",
                self._model,
            )
            return "INTERRUPT", "STOP"

    @staticmethod
    def _parse_decision_type(text: str) -> tuple[str, str]:
        """Parse 'DECISION: X\\nTYPE: Y' format from LLM response.

        Handles various formatting quirks (extra whitespace, single-line, etc.)
        """
        import re

        upper = text.upper()

        # Try standard two-line parse
        decision_match = re.search(r"DECISION\s*:\s*(IGNORE|INJECT|INTERRUPT)", upper)
        type_match = re.search(
            r"TYPE\s*:\s*(STOP|PAUSE|CORRECTION|NEW_QUESTION|SAME_TOPIC_REDIRECT|URGENCY|DONE_LISTENING|NULL)",
            upper,
        )

        decision = decision_match.group(1) if decision_match else None
        itype = type_match.group(1) if type_match else "null"
        if itype == "NULL":
            itype = "null"

        # If we got a valid decision, return it
        if decision:
            # For IGNORE/INJECT, type should be null
            if decision in ("IGNORE", "INJECT"):
                itype = "null"
            elif decision == "INTERRUPT" and itype == "null":
                # INTERRUPT without a type — default to STOP
                itype = "STOP"
            return decision, itype

        # Fallback: try to find just a decision word
        for label in ("INTERRUPT", "INJECT", "IGNORE"):
            if label in upper:
                fallback_type = "null"
                if label == "INTERRUPT":
                    fallback_type = "STOP"
                return label, fallback_type

        # Can't parse at all — default to IGNORE
        return "IGNORE", "null"
