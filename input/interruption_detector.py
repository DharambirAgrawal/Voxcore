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

# import asyncio
# import logging
# import time

# import numpy as np
# import torch

# from core.session import Session, TurnState
# from core.event_bus import EventBus, EventType
# from input.mic_stream import MicStream
# from input.vad import VADProcessor


# class InterruptionDetector:
#     """Monitors audio input during SPEAKING state to detect user interruptions.

#     Without hardware AEC, the mic picks up TTS audio from speakers.  Silero VAD
#     reports prob ≈ 1.0 for this echo because it IS speech — just not the user's.
#     VAD probability alone is therefore useless for distinguishing user speech
#     from echo during playback.

#     The ONLY reliable signal is **RMS energy**: the user's voice is physically
#     closer to the microphone than the speakers, so it's dramatically louder
#     (typically 5-20× higher RMS).

#     Architecture:
#       1. **Hard cooldown** (1 000 ms after entering SPEAKING) — ignore
#          everything while the pipeline ramps up.
#       2. **Continuously adapting echo baseline** — an exponential moving
#          average (EMA) tracks the typical RMS during SPEAKING.  This adapts
#          to actual speaker output level regardless of when TTS starts.
#       3. **Energy gate** — chunk RMS must exceed
#          ``max(ambient_gate, echo_ema × echo_multiplier)``
#          to even be considered.  With a 3× multiplier, echo at RMS 0.008
#          sets the gate at 0.024; a real user voice at 0.05+ passes easily.
#       4. **VAD threshold** (0.65) — kept as a secondary sanity check.
#       5. **Sustained-speech window** (500 ms) — prevents single loud
#          transients from triggering.

#     Uses its OWN Silero VAD model instance (the main VADProcessor resets
#     its model state during SPEAKING for echo suppression).
#     """

#     def __init__(
#         self,
#         session: Session,
#         event_bus: EventBus,
#         mic_stream: MicStream,
#         vad: VADProcessor,
#         config: dict,
#     ) -> None:
#         self.session = session
#         self.event_bus = event_bus
#         self.mic_stream = mic_stream
#         self.vad = vad

#         self._own_model = None  # Loaded lazily in run()
#         self._sample_rate: int = config.get("sample_rate", 16000)

#         # ── Tuning knobs ─────────────────────────────────────────────────
#         self._interrupt_threshold: float = 0.65       # VAD probability gate
#         self._interrupt_duration_ms: int = 400        # sustained speech required
#         self._energy_multiplier: float = 2.5          # ambient gate = noise_floor × this

#         # ── Continuously adapting echo baseline ──────────────────────────
#         self._echo_ema: float = 0.0                   # exponential moving average of RMS
#         self._echo_ema_alpha: float = 0.08            # EMA smoothing (lower = slower adapt)
#         self._echo_gate_multiplier: float = 3.0       # must exceed echo_ema × this
#         self._echo_ema_initialized: bool = False

#         # ── Cooldown ─────────────────────────────────────────────────────
#         self._state_cooldown_ms: float = 1000.0       # hard ignore window

#         self._speech_detected_at: float = 0.0
#         self._speaking_since: float = 0.0
#         self._audio_queue: asyncio.Queue = None

#         self._logger = logging.getLogger("InterruptionDetector")

#     def _load_own_model(self) -> None:
#         """Load a private Silero VAD model so we don't share state with the
#         main VADProcessor (which resets its model during SPEAKING)."""
#         try:
#             model, _ = torch.hub.load(
#                 repo_or_dir="snakers4/silero-vad",
#                 model="silero_vad",
#                 force_reload=False,
#             )
#         except Exception:
#             from silero_vad import load_silero_vad  # type: ignore[import-untyped]
#             model = load_silero_vad()
#         self._own_model = model
#         self._logger.info("InterruptionDetector: private Silero VAD model loaded")

#     def _get_speech_probability(self, chunk: np.ndarray) -> float:
#         """Run Silero VAD on a chunk using our private model instance."""
#         tensor = torch.from_numpy(chunk).float()
#         return float(self._own_model(tensor, self._sample_rate).item())

#     async def run(self) -> None:
#         """Main loop: consume audio, check state, run VAD, check for interrupt."""
#         self._load_own_model()
#         self._audio_queue = self.mic_stream.add_consumer()
#         self.event_bus.subscribe(EventType.STATE_CHANGED, self._handle_state_change)
#         self._logger.info(
#             "InterruptionDetector started (vad_thr=%.2f, duration=%dms, "
#             "echo_gate=%.1f×, cooldown=%dms)",
#             self._interrupt_threshold, self._interrupt_duration_ms,
#             self._echo_gate_multiplier, int(self._state_cooldown_ms),
#         )

#         while True:
#             chunk = await self._audio_queue.get()

#             if self.session.state != TurnState.SPEAKING:
#                 self._speech_detected_at = 0.0
#                 continue

#             now = time.monotonic()
#             rms = float(np.sqrt(np.mean(chunk ** 2)))

#             # ── Hard cooldown: ignore everything for first 1s ────────────
#             elapsed_ms = (now - self._speaking_since) * 1000 if self._speaking_since > 0 else 0
#             if elapsed_ms < self._state_cooldown_ms:
#                 # Still update echo EMA even during cooldown once audio is
#                 # flowing, so the baseline is warm by the time cooldown ends.
#                 if rms > 0.0005:
#                     self._update_echo_ema(rms)
#                 continue

#             # ── VAD probability FIRST — classify chunk before touching EMA ──
#             speech_prob = self._get_speech_probability(chunk)

#             # ── Update echo EMA only on non-speech frames (prob < 0.3) ────
#             # CRITICAL: if we update on every frame, the EMA chases the user's
#             # voice upward as they speak, making the gate too high to trigger.
#             # By only updating when Silero says "not speech", the EMA stays
#             # anchored to the actual echo/ambient level.
#             if speech_prob < 0.3:
#                 self._update_echo_ema(rms)

#             # ── RMS energy gate ──────────────────────────────────────────
#             noise_floor = self.mic_stream.noise_floor
#             ambient_gate = noise_floor * self._energy_multiplier if noise_floor > 0 else 0
#             echo_gate = self._echo_ema * self._echo_gate_multiplier if self._echo_ema > 0 else 0
#             energy_threshold = max(ambient_gate, echo_gate)

#             if rms < energy_threshold:
#                 if self._speech_detected_at > 0.0:
#                     self._logger.debug(
#                         "Energy gate reset (rms=%.4f < gate %.4f "
#                         "[ambient=%.4f, echo_ema=%.4f × %.1f = %.4f])",
#                         rms, energy_threshold, ambient_gate,
#                         self._echo_ema, self._echo_gate_multiplier, echo_gate,
#                     )
#                 self._speech_detected_at = 0.0
#                 continue

#             self._logger.debug(
#                 "Interrupt check: prob=%.3f rms=%.4f echo_ema=%.4f "
#                 "gate=%.4f (%.1f× above)",
#                 speech_prob, rms, self._echo_ema,
#                 energy_threshold, rms / energy_threshold if energy_threshold > 0 else 0,
#             )
#             await self._check_interrupt(speech_prob, rms)

#     def _update_echo_ema(self, rms: float) -> None:
#         """Update the exponential moving average of RMS energy."""
#         if not self._echo_ema_initialized:
#             self._echo_ema = rms
#             self._echo_ema_initialized = True
#         else:
#             self._echo_ema = (self._echo_ema_alpha * rms
#                               + (1 - self._echo_ema_alpha) * self._echo_ema)

#     async def _check_interrupt(self, speech_prob: float, rms: float) -> None:
#         """Analyze speech probability to detect sustained interruption."""
#         now = time.monotonic()

#         if speech_prob > self._interrupt_threshold:
#             if self._speech_detected_at == 0.0:
#                 self._speech_detected_at = now
#                 self._logger.debug(
#                     "Interrupt candidate started (prob=%.3f, rms=%.4f, echo_ema=%.4f)",
#                     speech_prob, rms, self._echo_ema,
#                 )
#             else:
#                 duration_ms = (now - self._speech_detected_at) * 1000
#                 if duration_ms >= self._interrupt_duration_ms:
#                     await self.event_bus.publish(
#                         EventType.INTERRUPT_DETECTED,
#                         data={"speech_prob": speech_prob, "during_sentence": -1},
#                     )
#                     self._logger.info(
#                         "INTERRUPT detected (%.0fms speech, prob=%.2f, "
#                         "rms=%.4f, echo_ema=%.4f, ratio=%.1f×)",
#                         duration_ms, speech_prob, rms, self._echo_ema,
#                         rms / self._echo_ema if self._echo_ema > 0 else 0,
#                     )
#                     self._speech_detected_at = 0.0
#                     await asyncio.sleep(0.5)
#                 else:
#                     self._logger.debug(
#                         "Interrupt building: %.0fms / %dms (rms=%.4f)",
#                         duration_ms, self._interrupt_duration_ms, rms,
#                     )
#         else:
#             if self._speech_detected_at > 0.0:
#                 elapsed = (now - self._speech_detected_at) * 1000
#                 self._logger.debug(
#                     "Interrupt candidate reset after %.0fms (prob=%.3f < %.2f)",
#                     elapsed, speech_prob, self._interrupt_threshold,
#                 )
#             self._speech_detected_at = 0.0

#     async def _handle_state_change(self, event) -> None:
#         """Handle STATE_CHANGED events to reset detection on state transitions."""
#         new_state = event.data.get("new_state")
#         old_state = event.data.get("old_state")

#         if new_state == TurnState.SPEAKING or new_state == "speaking":
#             self._speech_detected_at = 0.0
#             self._speaking_since = time.monotonic()
#             # Reset echo EMA for this new SPEAKING turn
#             self._echo_ema = 0.0
#             self._echo_ema_initialized = False
#             # Reset private model state
#             if self._own_model is not None:
#                 self._own_model.reset_states()
#         elif old_state == TurnState.SPEAKING or old_state == "speaking":
#             self._speech_detected_at = 0.0
#             self._speaking_since = 0.0


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
    """Monitors audio during SPEAKING to detect user interruptions.

    Without hardware AEC the mic picks up TTS echo.  Silero VAD reports
    prob ≈ 1.0 for echo because it IS speech — just not the user's.

    Discrimination strategy:
            1. Hard cooldown (500ms) after entering SPEAKING
      2. Continuously adapting echo RMS baseline (EMA)
      3. Energy gate: rms must exceed max(ambient, echo_ema × 3, absolute_min)
      4. VAD probability gate (0.60)
      5. Sustained speech window (300ms)

    State transitions are tracked INSIDE the main loop — not dependent
    on the event bus callback (which may fire late or not at all).

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
        self._sample_rate: int = config.get("sample_rate", 16000)

        # ── Tuning knobs ─────────────────────────────────────────────────
        self._interrupt_threshold: float = 0.60
        self._interrupt_duration_ms: int = 300
        self._energy_multiplier: float = 2.5

        # ── Echo baseline tracking ───────────────────────────────────────
        self._echo_ema: float = 0.0
        self._echo_ema_initialized: bool = False
        self._echo_gate_multiplier: float = 3.0
        self._min_absolute_rms: float = 0.005

        # ── Cooldown ─────────────────────────────────────────────────────
        self._state_cooldown_ms: float = 500.0

        self._speech_detected_at: float = 0.0
        self._speaking_since: float = 0.0
        self._audio_queue: asyncio.Queue = None

        # ── FIX: Self-tracked state transitions ──────────────────────────
        self._was_speaking_state: bool = False

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
    # Main loop — self-tracks state transitions
    # ─────────────────────────────────────────────────────────────────────

    async def run(self) -> None:
        self._load_own_model()
        self._audio_queue = self.mic_stream.add_consumer()
        self._logger.info(
            "InterruptionDetector started (vad=%.2f, dur=%dms, "
            "echo_gate=%.1f×, cooldown=%dms, min_rms=%.4f)",
            self._interrupt_threshold,
            self._interrupt_duration_ms,
            self._echo_gate_multiplier,
            int(self._state_cooldown_ms),
            self._min_absolute_rms,
        )

        non_speech_gate = 0.35

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
                self._echo_ema = 0.0
                self._echo_ema_initialized = False
                if self._own_model is not None:
                    self._own_model.reset_states()
                self._logger.debug(
                    "SPEAKING entered — cooldown started (%.0fms)",
                    self._state_cooldown_ms,
                )

            now = time.monotonic()
            rms = float(np.sqrt(np.mean(chunk ** 2)))
            speech_prob = self._get_speech_probability(chunk)

            # ── Cooldown: ignore first window, but warm echo EMA on non-speech ──
            elapsed_ms = (now - self._speaking_since) * 1000.0
            if elapsed_ms < self._state_cooldown_ms:
                if speech_prob < non_speech_gate and rms > 0.0005:
                    self._update_echo_ema(rms)
                continue

            # ── Update echo EMA only on non-speech frames ────────────────
            if speech_prob < non_speech_gate and self._should_update_echo_ema(rms):
                self._update_echo_ema(rms)

            # ── Energy gate ──────────────────────────────────────────────
            noise_floor = self.mic_stream.noise_floor
            ambient_gate = (
                noise_floor * self._energy_multiplier if noise_floor > 0 else 0
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
                self._speech_detected_at = 0.0
                continue

            # ── Passed energy gate — check VAD + duration ────────────────
            self._logger.debug(
                "Interrupt candidate: prob=%.3f rms=%.4f gate=%.4f "
                "(echo_ema=%.4f, ratio=%.1f×)",
                speech_prob,
                rms,
                energy_threshold,
                self._echo_ema,
                rms / energy_threshold if energy_threshold > 0 else 0,
            )
            await self._check_interrupt(speech_prob, rms)

    # ─────────────────────────────────────────────────────────────────────
    # Interrupt check
    # ─────────────────────────────────────────────────────────────────────

    async def _check_interrupt(self, speech_prob: float, rms: float) -> None:
        now = time.monotonic()

        if speech_prob > self._interrupt_threshold:
            if self._speech_detected_at == 0.0:
                self._speech_detected_at = now
                self._logger.debug(
                    "Interrupt timer started (prob=%.3f, rms=%.4f)",
                    speech_prob,
                    rms,
                )
            else:
                duration_ms = (now - self._speech_detected_at) * 1000
                if duration_ms >= self._interrupt_duration_ms:
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
                    await asyncio.sleep(0.5)
        else:
            self._speech_detected_at = 0.0