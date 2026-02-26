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

    Without hardware AEC the mic picks up TTS echo.  Silero VAD reports
    prob ≈ 1.0 for echo because it IS speech — just not the user's.

    Discrimination strategy:
      1. Hard cooldown (500ms) after entering SPEAKING
      2. Continuously adapting echo RMS baseline (EMA)
      3. Energy gate: rms must exceed max(ambient, echo_ema × 3, absolute_min)
      4. VAD probability gate (0.60)
      5. Sustained speech window — configurable via speaking_monitor.gate1_min_duration_ms
         (defaults to 600ms when monitor enabled, 300ms when disabled)

    When speaking_monitor is ENABLED (v2):
      - Fires GATE1_PASSED instead of INTERRUPT_DETECTED
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
        self._interrupt_duration_ms: int = (
            monitor_cfg.get("gate1_min_duration_ms", 300)
            if self._monitor_enabled
            else 300
        )
        self._energy_multiplier: float = 2.5

        # ── AEC mode vs legacy echo EMA ───────────────────────────────
        self._aec_enabled: bool = audio_cfg.get("echo_cancellation", True)
        # Legacy echo baseline (only used when AEC is OFF)
        self._echo_ema: float = 0.0
        self._echo_ema_initialized: bool = False
        self._echo_gate_multiplier: float = 3.0
        self._min_absolute_rms: float = 0.005

        # ── Cooldown ─────────────────────────────────────────────────────
        self._state_cooldown_ms: float = 500.0

        self._speech_detected_at: float = 0.0
        self._last_speech_at: float = 0.0       # last frame with prob > threshold
        self._speech_hold_ms: float = 400.0     # hold window: natural pauses between words
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

    def _update_echo_ema(self, rms: float) -> None:
        """Asymmetric EMA: fast rise (catch loud echo), slow decay."""
        if not self._echo_ema_initialized:
            self._echo_ema = rms
            self._echo_ema_initialized = True
        else:
            alpha = 0.25 if rms > self._echo_ema else 0.03
            self._echo_ema = alpha * rms + (1.0 - alpha) * self._echo_ema

    def _should_update_echo_ema(self, rms: float) -> bool:
        """Update baseline only for frames that look like echo, not user voice."""
        if not self._echo_ema_initialized:
            return True
        return rms < self._echo_ema * self._echo_gate_multiplier

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
        # V3: With AEC, use regular consumer (echo already removed).
        # Legacy: use raw consumer + echo EMA discrimination.
        if self._aec_enabled:
            self._audio_queue = self.mic_stream.add_consumer()
        else:
            self._audio_queue = self.mic_stream.add_raw_consumer()
        self._logger.info(
            "InterruptionDetector started (vad=%.2f, dur=%dms, "
            "cooldown=%dms, min_rms=%.4f, monitor=%s, aec=%s)",
            self._interrupt_threshold,
            self._interrupt_duration_ms,
            int(self._state_cooldown_ms),
            self._min_absolute_rms,
            "ENABLED" if self._monitor_enabled else "disabled",
            "ON" if self._aec_enabled else "OFF",
        )

        non_speech_gate = 0.35

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
                self._echo_ema = 0.0
                self._echo_ema_initialized = False
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
                if not self._aec_enabled and rms > 0.0005:
                    self._update_echo_ema(rms)
                continue

            # ── Cooldown: ignore first window after entering SPEAKING ────
            elapsed_ms = (now - self._speaking_since) * 1000.0
            if elapsed_ms < self._state_cooldown_ms:
                if not self._aec_enabled and rms > 0.0005:
                    self._update_echo_ema(rms)
                continue

            # ══════════════════════════════════════════════════════════
            # Energy gate — AEC vs legacy path
            # ══════════════════════════════════════════════════════════
            noise_floor = self.mic_stream.noise_floor

            if self._aec_enabled:
                # ── AEC mode: echo already removed ──
                # Simple threshold: rms > noise_floor × multiplier
                # No echo EMA needed — the signal is clean.
                ambient_gate = (
                    noise_floor * self._energy_multiplier
                    if noise_floor > 0
                    else 0
                )
                energy_threshold = max(ambient_gate, self._min_absolute_rms)
            else:
                # ── Legacy mode: echo EMA tracking ──
                if speech_prob < non_speech_gate:
                    if self._should_update_echo_ema(rms):
                        self._update_echo_ema(rms)
                elif self._echo_ema < 0.003:
                    self._update_echo_ema(rms)
                elif self._echo_ema > 0 and rms < self._echo_ema * 2.0:
                    self._update_echo_ema(rms)

                ambient_gate = (
                    noise_floor * self._energy_multiplier
                    if noise_floor > 0
                    else 0
                )
                echo_gate = (
                    self._echo_ema * self._echo_gate_multiplier
                    if self._echo_ema > 0
                    else 0
                )
                energy_threshold = max(
                    ambient_gate, echo_gate, self._min_absolute_rms
                )

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

            # ── Passed energy gate — check VAD + duration ────────────────
            if self._speech_detected_at == 0.0:
                # First frame passing energy gate — log for visibility
                self._logger.info(
                    "Interrupt candidate: prob=%.3f rms=%.4f gate=%.4f "
                    "(aec=%s, echo_ema=%.4f)",
                    speech_prob,
                    rms,
                    energy_threshold,
                    "ON" if self._aec_enabled else "OFF",
                    self._echo_ema,
                )
            await self._check_interrupt(speech_prob, rms)

    # ─────────────────────────────────────────────────────────────────────
    # Interrupt check
    # ─────────────────────────────────────────────────────────────────────

    async def _check_interrupt(self, speech_prob: float, rms: float) -> None:
        now = time.monotonic()

        if speech_prob > self._interrupt_threshold:
            self._last_speech_at = now  # track latest high-prob speech frame
            if self._speech_detected_at == 0.0:
                self._speech_detected_at = now
                self._logger.debug(
                    "Interrupt timer accumulating (prob=%.3f, rms=%.4f)",
                    speech_prob,
                    rms,
                )
            else:
                duration_ms = (now - self._speech_detected_at) * 1000
                self._logger.debug(
                    "Gate 1: %.0f/%.0fms (prob=%.2f)",
                    duration_ms, self._interrupt_duration_ms, speech_prob,
                )
                if duration_ms >= self._interrupt_duration_ms:
                    if self._monitor_enabled:
                        # ── V2: Defer to SpeakingMonitor (Gate 2 + 3) ────
                        await self.event_bus.publish(
                            EventType.GATE1_PASSED,
                            data={
                                "speech_prob": speech_prob,
                                "rms": rms,
                                "duration_ms": duration_ms,
                                "echo_ema": self._echo_ema,
                            },
                            source="InterruptionDetector",
                        )
                        self._logger.info(
                            "Gate 1 PASSED (%.0fms, prob=%.2f, rms=%.4f) → SpeakingMonitor",
                            duration_ms, speech_prob, rms,
                        )
                    else:
                        # ── V1 fallback: fire INTERRUPT_DETECTED directly ─
                        await self.event_bus.publish(
                            EventType.INTERRUPT_DETECTED,
                            data={
                                "speech_prob": speech_prob,
                                "during_sentence": -1,
                            },
                        )
                        self._logger.info(
                            "INTERRUPT detected (%.0fms, prob=%.2f, "
                            "rms=%.4f, echo_ema=%.4f, ratio=%.1f×)",
                            duration_ms,
                            speech_prob,
                            rms,
                            self._echo_ema,
                            rms / self._echo_ema if self._echo_ema > 0 else 0,
                        )
                    self._speech_detected_at = 0.0
                    self._last_speech_at = 0.0
                    await asyncio.sleep(0.5)
        else:
            # Non-speech frame: only reset timer if hold window expired
            if self._speech_detected_at > 0 and self._last_speech_at > 0:
                gap_ms = (now - self._last_speech_at) * 1000.0
                if gap_ms > self._speech_hold_ms:
                    self._speech_detected_at = 0.0
                    self._last_speech_at = 0.0
            elif self._speech_detected_at > 0 and self._last_speech_at == 0.0:
                # Timer started but never saw high-prob frame — reset
                self._speech_detected_at = 0.0