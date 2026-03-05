"""
╔══════════════════════════════════════════════════════════════════════════════════╗
║                  VOXCORE — input/interruption_detector.py                        ║
║        MONITORS VAD WHILE SPEAKING — FIRES INTERRUPT IF USER TALKS OVER        ║
╚══════════════════════════════════════════════════════════════════════════════════╝

PURPOSE:
    Runs a parallel VAD check specifically during the SPEAKING state. When the
    system is speaking (TTS audio playing) and the user starts talking, this
    module detects it and fires INTERRUPT_DETECTED.
    
    It does NOT handle STT — just kills the audio. The main STT loop picks up
    the user's interrupting speech normally when SPEECH_END fires.

    Critical for full-duplex feel: the user can interrupt the AI at any time.

═══════════════════════════════════════════════════════════════════════════════════
IMPORTS REQUIRED:
═══════════════════════════════════════════════════════════════════════════════════

import asyncio                          # Async operations
import logging                          # Module logger
import time                             # Duration tracking

from core.session import Session, TurnState
from core.event_bus import EventBus, EventType
from input.mic_stream import MicStream
from input.vad import VADProcessor      # Re-uses the same VAD model for probability

═══════════════════════════════════════════════════════════════════════════════════
CLASSES:
═══════════════════════════════════════════════════════════════════════════════════

──────────────────────────────────────────────────────────────────────────────────
CLASS: InterruptionDetector
──────────────────────────────────────────────────────────────────────────────────
    Monitors audio input during SPEAKING state to detect user interruptions.

    CONSTRUCTOR: __init__(self, session: Session, event_bus: EventBus,
                          mic_stream: MicStream, vad: VADProcessor, config: dict)
    ─────────────────────────────────────────────────────────────
        INPUTS:
            - session: Session — For checking current TurnState
            - event_bus: EventBus — For publishing INTERRUPT_DETECTED
            - mic_stream: MicStream — For getting audio chunks
            - vad: VADProcessor — For using the loaded Silero VAD model
            - config: dict — The "audio" section from config.yaml
        
        INITIALIZES:
            self.session: Session              = session
            self.event_bus: EventBus           = event_bus
            self.mic_stream: MicStream         = mic_stream
            self.vad: VADProcessor             = vad
            
            self._interrupt_threshold: float   = 0.6    # Higher than normal VAD threshold
                                                         # (need more confidence during speaker output
                                                         #  because TTS audio may bleed into mic)
            self._interrupt_duration_ms: int   = 300     # Must detect speech for 300ms before interrupting
                                                         # (avoids false triggers from TTS audio bleed)
            self._speech_detected_at: float    = 0.0     # When user speech first detected during SPEAKING
            self._audio_queue: asyncio.Queue   = None    # Consumer queue from mic_stream
            
            self._logger: logging.Logger       = logging.getLogger("InterruptDetector")

    METHODS:
    ─────────────────────────────────────────────────────────────

    async def run(self) -> None
        INPUTS: None
        OUTPUT: None (runs forever)
        WHAT IT DOES:
            1. Gets audio consumer queue: self._audio_queue = mic_stream.add_consumer()
            2. Enters infinite loop:
               a. chunk = await self._audio_queue.get()
               b. If session.state != TurnState.SPEAKING:
                  Reset self._speech_detected_at = 0.0
                  Continue (skip processing — only active during SPEAKING)
               c. speech_prob = vad._get_speech_probability(chunk)
               d. await self._check_interrupt(speech_prob)

    async def _check_interrupt(self, speech_prob: float) -> None
        INPUTS:
            - speech_prob: float — Current speech probability from VAD
        OUTPUT: None (fires event if interrupt detected)
        WHAT IT DOES:
            If speech_prob > self._interrupt_threshold:
                If self._speech_detected_at == 0.0:
                    # First detection — start timing
                    self._speech_detected_at = time.time()
                Else:
                    # Already detecting — check duration
                    duration_ms = (time.time() - self._speech_detected_at) * 1000
                    If duration_ms >= self._interrupt_duration_ms:
                        # Confirmed interrupt!
                        1. Publish INTERRUPT_DETECTED event with data:
                           {"speech_prob": speech_prob, "during_sentence": -1}
                        2. Log: "INTERRUPT detected ({duration_ms:.0f}ms of speech during SPEAKING)"
                        3. Reset self._speech_detected_at = 0.0
                        4. Wait briefly (100ms) to avoid re-triggering:
                           await asyncio.sleep(0.1)
            Else:
                # No speech — reset detection
                self._speech_detected_at = 0.0
        
        WHY 300ms THRESHOLD:
            When TTS audio is playing through speakers, some of it can bleed
            into the microphone (especially without echo cancellation). A 300ms
            threshold ensures we only interrupt for genuine user speech, not
            momentary audio bleed. Real interruptions are sustained speech
            that easily exceeds 300ms.

    async def _handle_state_change(self, event) -> None
        INPUTS:
            - event: Event — STATE_CHANGED event
        OUTPUT: None
        WHAT IT DOES:
            1. When state changes TO SPEAKING: enable interrupt detection
            2. When state changes FROM SPEAKING: disable and reset
            3. This is registered as a callback on STATE_CHANGED event
        
        NOTE: This is an alternative to polling session.state in the run loop.
              Can use either approach.

═══════════════════════════════════════════════════════════════════════════════════
EXPORTS:
    - InterruptionDetector   (class)
═══════════════════════════════════════════════════════════════════════════════════

ECHO CANCELLATION NOTE:
    VoxCore does NOT implement acoustic echo cancellation (AEC). This means
    TTS audio playing through speakers can be picked up by the microphone.
    The 300ms duration threshold and higher speech probability threshold (0.6)
    help mitigate false interrupts from audio bleed.
    
    For production, consider:
    - Using headphones (eliminates the problem)
    - Muting the mic during TTS playback (simpler but loses true full-duplex)
    - Implementing software AEC using WebRTC's AEC module
    - Using a hardware AEC solution
"""

"""
╔══════════════════════════════════════════════════════════════════════════════════╗
║                  VOXCORE — input/interruption_detector.py                        ║
║        MONITORS VAD WHILE SPEAKING — FIRES INTERRUPT IF USER TALKS OVER        ║
╚══════════════════════════════════════════════════════════════════════════════════╝
"""

import asyncio
import logging
import time

import numpy as np
import torch

from core.session import Session, TurnState
from core.event_bus import EventBus, EventType
from input.mic_stream import MicStream
from input.vad import VADProcessor


class InterruptionDetector:
    """Gate 1 — Monitors audio during SPEAKING for sustained user speech.

    V3 two-path system:
      Path A: HIGH_ENERGY_BURST — 150ms, 8× noise floor. Catches sharp commands
              ("stop", "hold on"). Skips Gate 2, goes directly to Gate 3.
      Path B: NORMAL_SPEECH — 600ms, 3× noise floor. For regular interruptions.
              Proceeds to Gate 2 (filler check) then Gate 3 (LLM classification).

    Discrimination strategy:
      1. Hard cooldown (500ms) after entering SPEAKING
      2. Continuously adapting echo RMS baseline (EMA)
      3. Energy gate: rms must exceed max(ambient, echo_ema × 3, absolute_min)
      4. VAD probability gate (0.60)
      5. Two-path sustained speech window

    When speaking_monitor is ENABLED (v2/v3):
      - Fires GATE1_PASSED with path='A' or path='B'
      - SpeakingMonitor handles Gate 2 (filler) and Gate 3 (LLM classification)

    When speaking_monitor is DISABLED:
      - Falls back to v1 behavior: fires INTERRUPT_DETECTED directly

    Uses its OWN Silero VAD model (main VADProcessor resets during SPEAKING).
    """

    def __init__(
        self,
        session: Session,
        event_bus: EventBus,
        mic_stream: MicStream,
        vad: VADProcessor,
        config: dict,
    ) -> None:
        self.session = session
        self.event_bus = event_bus
        self.mic_stream = mic_stream
        self.vad = vad

        self._own_model = None
        # V2: config is now the full config dict — read audio settings from nested section
        audio_cfg = config.get("audio", config) if isinstance(config, dict) else config
        self._sample_rate: int = audio_cfg.get("sample_rate", 16000)

        # ── Speaking monitor integration ─────────────────────────────────
        monitor_cfg = config.get("speaking_monitor", {}) if isinstance(config, dict) else {}
        self._monitor_enabled: bool = monitor_cfg.get("enabled", True)

        # ── Tuning knobs ─────────────────────────────────────────────────
        self._interrupt_threshold: float = 0.60

        # V3: Two-path Gate 1 configuration
        self._path_a_duration_ms: int = monitor_cfg.get("gate1_path_a_min_duration_ms", 150)
        self._path_a_energy_ratio: float = monitor_cfg.get("gate1_path_a_energy_ratio", 13.0)
        self._path_b_duration_ms: int = monitor_cfg.get("gate1_path_b_min_duration_ms", 750)
        self._path_b_energy_ratio: float = monitor_cfg.get("gate1_path_b_energy_ratio", 6.0)

        # Legacy single-path fallback (when monitor disabled)
        self._interrupt_duration_ms: int = (
            monitor_cfg.get("gate1_min_duration_ms", 300)
            if self._monitor_enabled
            else 300
        )
        # Entry gate: noise_floor × this must be exceeded before any
        # accumulation window opens. Configurable so room conditions
        # can be tuned without code changes.
        self._energy_multiplier: float = monitor_cfg.get("gate1_entry_multiplier", 4.0)

        # ── Cooldown ─────────────────────────────────────────────────────
        self._state_cooldown_ms: float = 500.0

        self._speech_detected_at: float = 0.0
        self._last_speech_at: float = 0.0       # last frame with prob > threshold
        self._speech_hold_ms: float = 1200.0    # hold window: natural pauses between words
                                                 # (raised 400→1200ms: V5 echo correlation can
                                                 #  block user+echo mixed frames for 400-800ms
                                                 #  causing premature accumulation reset.
                                                 #  1200ms gives real speech time to punch through
                                                 #  the echo without the timer expiring.
                                                 #  Pure echo still expires after 1200ms of only
                                                 #  echo-blocked frames — no false fire since a
                                                 #  non-echo frame at ≥7× is needed to actually
                                                 #  trigger Path A.)
        self._speaking_since: float = 0.0
        self._audio_queue: asyncio.Queue = None

        # ── FIX: Self-tracked state transitions ──────────────────────────
        self._was_speaking_state: bool = False

        # ── Playback pause/resume tracking (PersonaPlex) ─────────────────
        # When SpeakingMonitor pauses playback, Gate 1 must be suppressed
        # to avoid re-triggering on stale audio. On resume, a cooldown
        # lets echo_ema recalibrate to the real playback level.
        self._playback_paused: bool = False
        self._resume_cooldown_until: float = 0.0
        self._resume_cooldown_ms: float = 1500.0

        self._logger = logging.getLogger("InterruptionDetector")

    # ─────────────────────────────────────────────────────────────────────
    # Model
    # ─────────────────────────────────────────────────────────────────────

    def _load_own_model(self) -> None:
        """Load a private Silero VAD model instance."""
        try:
            model, _ = torch.hub.load(
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                force_reload=False,
            )
        except Exception:
            from silero_vad import load_silero_vad  # type: ignore[import-untyped]
            model = load_silero_vad()
        self._own_model = model
        self._logger.info("InterruptionDetector: private Silero VAD model loaded")

    def _get_speech_probability(self, chunk: np.ndarray) -> float:
        tensor = torch.from_numpy(chunk).float()
        return float(self._own_model(tensor, self._sample_rate).item())

    # ─────────────────────────────────────────────────────────────────────
    # Echo baseline
    # ─────────────────────────────────────────────────────────────────────



    # ─────────────────────────────────────────────────────────────────────
    # PersonaPlex pause/resume callbacks
    # ─────────────────────────────────────────────────────────────────────

    async def _on_playback_pause(self, event) -> None:
        """SpeakingMonitor paused playback — suppress Gate 1 detection."""
        self._playback_paused = True
        self._speech_detected_at = 0.0
        self._last_speech_at = 0.0

    async def _on_playback_resume(self, event) -> None:
        """Playback resumed — recalibrate echo baseline before detecting."""
        self._playback_paused = False
        self._resume_cooldown_until = (
            time.monotonic() + self._resume_cooldown_ms / 1000.0
        )
        self._speech_detected_at = 0.0
        self._last_speech_at = 0.0
        self._logger.debug(
            "Resume → echo recalibration cooldown %.0fms",
            self._resume_cooldown_ms,
        )

    # ─────────────────────────────────────────────────────────────────────
    # Main loop — self-tracks state transitions
    # ─────────────────────────────────────────────────────────────────────

    async def run(self) -> None:
        self._load_own_model()
        # V3 FIX: Use raw consumer — Gate0 must NOT filter InterruptionDetector's
        # audio stream. Gate0 blocking fragments duration accumulation, preventing
        # the two-path system from ever reaching its thresholds. InterruptionDetector
        # has its own energy gating + VAD and doesn't need upstream echo filtering.
        self._audio_queue = self.mic_stream.add_raw_consumer()
        self._logger.info(
            "InterruptionDetector started (vad=%.2f, "
            "pathA=%dms/%.0f×, pathB=%dms/%.1f×, "
            "cooldown=%dms, monitor=%s)",
            self._interrupt_threshold,
            self._path_a_duration_ms, self._path_a_energy_ratio,
            self._path_b_duration_ms, self._path_b_energy_ratio,
            int(self._state_cooldown_ms),
            "ENABLED" if self._monitor_enabled else "disabled",
        )

        # Subscribe to pause/resume events from PersonaPlex pipeline
        self.event_bus.subscribe(EventType.PLAYBACK_PAUSE, self._on_playback_pause)
        self.event_bus.subscribe(EventType.PLAYBACK_RESUME, self._on_playback_resume)

        while True:
            chunk = await self._audio_queue.get()

            is_speaking_now = self.session.state == TurnState.SPEAKING

            # ── NOT speaking → reset and skip ────────────────────────────
            if not is_speaking_now:
                if self._was_speaking_state:
                    # Just LEFT speaking
                    self._logger.debug("SPEAKING ended — interrupt detection paused")
                self._was_speaking_state = False
                self._speech_detected_at = 0.0
                self._last_speech_at = 0.0
                continue

            # ══════════════════════════════════════════════════════════════
            # FIX: Detect SPEAKING entry HERE, not in event handler
            # This guarantees _speaking_since is always set correctly
            # ══════════════════════════════════════════════════════════════
            if not self._was_speaking_state:
                # Just ENTERED speaking state
                self._was_speaking_state = True
                self._speaking_since = time.monotonic()
                self._speech_detected_at = 0.0
                self._last_speech_at = 0.0
                # Reset pause/resume state from any previous cycle
                self._playback_paused = False
                self._resume_cooldown_until = 0.0
                if self._own_model is not None:
                    self._own_model.reset_states()
                self._logger.debug(
                    "SPEAKING entered — cooldown started (%.0fms)",
                    self._state_cooldown_ms,
                )

            now = time.monotonic()
            rms = float(np.sqrt(np.mean(chunk ** 2)))
            speech_prob = self._get_speech_probability(chunk)

            # ── Playback paused → suppress detection entirely ────────
            if self._playback_paused:
                continue

            # ── Post-resume cooldown → recalibrate ───────────────────
            if now < self._resume_cooldown_until:
                continue

            # ── Cooldown: ignore first window after entering SPEAKING ────
            elapsed_ms = (now - self._speaking_since) * 1000.0
            if elapsed_ms < self._state_cooldown_ms:
                continue

            # ══════════════════════════════════════════════════════════
            # Energy gate — V3: simple threshold for all paths
            # ══════════════════════════════════════════════════════════
            # AriaVoiceFilter + Gate0 handle echo upstream in mic_stream.
            # We only need a noise-floor gate here, no echo_ema tracking.
            noise_floor = self.mic_stream.noise_floor

            ambient_gate = (
                noise_floor * self._energy_multiplier
                if noise_floor > 0
                else 0
            )
            energy_threshold = ambient_gate

            if rms < energy_threshold:
                # Don't immediately reset — brief energy dips between phonemes
                # (e.g. /p/ in "stop") should not break a speech window.
                # Only reset if no recent high-prob speech frame within hold time.
                if self._speech_detected_at > 0 and self._last_speech_at > 0:
                    gap_ms = (now - self._last_speech_at) * 1000.0
                    if gap_ms > self._speech_hold_ms:
                        self._speech_detected_at = 0.0
                        self._last_speech_at = 0.0
                elif self._speech_detected_at == 0.0:
                    pass  # not timing, nothing to reset
                continue

            # ── Echo guard: skip AI's own voice (raw consumer + query filters) ──
            # InterruptionDetector uses add_raw_consumer() so upstream echo
            # filtering does not apply here. Call mic_stream.check_echo() to
            # query the same AriaVoiceFilter + Gate0 + V5 playback-ref directly.
            #
            # IMPORTANT: During a real interrupt the mic captures a MIX of user
            # voice + TTS echo.  Some 30ms frames will have high enough echo
            # correlation to trip V5 / Gate0 even though the user IS speaking.
            # Hard-resetting the accumulation timer on every echo frame prevents
            # the two-path system from ever reaching 150ms / 750ms.
            #
            # Fix: during an active accumulation window, don't hard-reset.
            # Just skip the frame (don't advance timers).  The existing 400ms
            # speech-hold window will naturally expire and reset if only echo
            # is present (no real speech frames refreshing _last_speech_at).
            if self.mic_stream.check_echo(chunk):
                if self._speech_detected_at > 0 and self._last_speech_at > 0:
                    # Active accumulation — check hold window instead of hard reset
                    gap_ms = (now - self._last_speech_at) * 1000.0
                    if gap_ms > self._speech_hold_ms:
                        # No real speech for 400ms+ during echo → probably just echo
                        self._logger.debug(
                            "Echo during accumulation: hold window expired (%.0fms) — reset",
                            gap_ms,
                        )
                        self._speech_detected_at = 0.0
                        self._last_speech_at = 0.0
                    else:
                        self._logger.debug(
                            "Echo frame skipped during active accumulation (hold %.0fms)",
                            gap_ms,
                        )
                continue

            # ── Passed energy gate — check VAD + duration ────────────────
            if self._speech_detected_at == 0.0:
                # First frame passing energy gate — log for visibility
                self._logger.info(
                    "Interrupt candidate: prob=%.3f rms=%.4f gate=%.4f",
                    speech_prob,
                    rms,
                    energy_threshold,
                )
            await self._check_interrupt(speech_prob, rms)

    # ─────────────────────────────────────────────────────────────────────
    # Interrupt check
    # ─────────────────────────────────────────────────────────────────────

    async def _check_interrupt(self, speech_prob: float, rms: float) -> None:
        now = time.monotonic()

        if speech_prob > self._interrupt_threshold:
            self._last_speech_at = now
            if self._speech_detected_at == 0.0:
                self._speech_detected_at = now
                self._logger.debug(
                    "Interrupt timer accumulating (prob=%.3f, rms=%.4f)",
                    speech_prob, rms,
                )
            else:
                duration_ms = (now - self._speech_detected_at) * 1000
                noise_floor = self.mic_stream.noise_floor

                if self._monitor_enabled:
                    # ── V3: Two-path Gate 1 ─────────────────────────
                    path = None
                    energy_ratio = rms / max(noise_floor, 1e-8)

                    # Path A: HIGH_ENERGY_BURST — short + very loud
                    if (duration_ms >= self._path_a_duration_ms
                            and energy_ratio >= self._path_a_energy_ratio):
                        path = "A"
                    # Path B: NORMAL_SPEECH — sustained + moderate
                    elif (duration_ms >= self._path_b_duration_ms
                            and energy_ratio >= self._path_b_energy_ratio):
                        path = "B"

                    if path is not None:
                        # ── FIRE: publish GATE1_PASSED for SpeakingMonitor ──
                        self._logger.info(
                            "Gate 1 PASSED (path=%s, %.0fms, prob=%.2f, "
                            "ratio=%.1f×, rms=%.4f)",
                            path, duration_ms, speech_prob,
                            energy_ratio, rms,
                        )
                        await self.event_bus.publish(
                            EventType.GATE1_PASSED,
                            {
                                "path": path,
                                "duration_ms": duration_ms,
                                "speech_prob": speech_prob,
                                "energy_ratio": energy_ratio,
                                "rms": rms,
                            },
                            source="InterruptionDetector",
                        )
                        # Reset timers + brief cooldown to prevent re-fire
                        self._speech_detected_at = 0.0
                        self._last_speech_at = 0.0
                        await asyncio.sleep(0.5)
                    else:
                        self._logger.debug(
                            "Gate 1: %.0fms (prob=%.2f, ratio=%.1f×) — waiting",
                            duration_ms, speech_prob, energy_ratio,
                        )
        else:
            # Non-speech frame: only reset timer if hold window expired
            if self._speech_detected_at > 0 and self._last_speech_at > 0:
                gap_ms = (now - self._last_speech_at) * 1000.0
                if gap_ms > self._speech_hold_ms:
                    self._speech_detected_at = 0.0
                    self._last_speech_at = 0.0
            elif self._speech_detected_at > 0 and self._last_speech_at == 0.0:
                self._speech_detected_at = 0.0