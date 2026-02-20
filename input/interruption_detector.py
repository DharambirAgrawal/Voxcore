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

from core.session import Session, TurnState
from core.event_bus import EventBus, EventType
from input.mic_stream import MicStream
from input.vad import VADProcessor


class InterruptionDetector:
    """Monitors audio input during SPEAKING state to detect user interruptions."""

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

        self._interrupt_threshold: float = config.get("interrupt_threshold", 0.6)
        self._interrupt_duration_ms: int = config.get("interrupt_duration_ms", 300)
        self._speech_detected_at: float = 0.0
        self._audio_queue: asyncio.Queue = None

        self._logger = logging.getLogger("InterruptionDetector")

    async def run(self) -> None:
        """Main loop: consume audio, check state, run VAD, check for interrupt."""
        self._audio_queue = self.mic_stream.add_consumer()
        # Wire up state change handler for clean reset on state transitions
        self.event_bus.subscribe(EventType.STATE_CHANGED, self._handle_state_change)  # callback-style

        while True:
            chunk = await self._audio_queue.get()

            if self.session.state != TurnState.SPEAKING:
                self._speech_detected_at = 0.0
                continue

            # FIX 2: Use public API if available, fall back to private
            speech_prob = self.vad.get_speech_probability(chunk)
            await self._check_interrupt(speech_prob)

    async def _check_interrupt(self, speech_prob: float) -> None:
        """Analyze speech probability to detect sustained interruption."""
        # FIX 1: Use time.monotonic() — immune to system clock adjustments
        now = time.monotonic()

        if speech_prob > self._interrupt_threshold:
            if self._speech_detected_at == 0.0:
                # First detection — start timing
                self._speech_detected_at = now
            else:
                # Check duration of sustained speech
                duration_ms = (now - self._speech_detected_at) * 1000
                if duration_ms >= self._interrupt_duration_ms:
                    # Confirmed interrupt
                    await self.event_bus.publish(
                        EventType.INTERRUPT_DETECTED,
                        data={"speech_prob": speech_prob, "during_sentence": -1},
                    )
                    self._logger.info(
                        f"INTERRUPT detected ({duration_ms:.0f}ms of speech during SPEAKING)"
                    )
                    self._speech_detected_at = 0.0
                    await asyncio.sleep(0.1)  # Cooldown to prevent re-trigger
        else:
            # No speech detected — reset timer
            self._speech_detected_at = 0.0

    async def _handle_state_change(self, event) -> None:
        """Handle STATE_CHANGED events to reset detection on state transitions."""
        new_state = event.data.get("new_state")
        old_state = event.data.get("old_state")

        if new_state == TurnState.SPEAKING:
            # Entering SPEAKING — fresh start for detection
            self._speech_detected_at = 0.0
        elif old_state == TurnState.SPEAKING:
            # Leaving SPEAKING — clean up
            self._speech_detected_at = 0.0