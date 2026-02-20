"""
╔══════════════════════════════════════════════════════════════════════════════════╗
║                    VOXCORE — backchannel/cue_detector.py                         ║
║        PHRASE BOUNDARY DETECTION — FINDS NATURAL BACKCHANNEL OPPORTUNITIES      ║
╚══════════════════════════════════════════════════════════════════════════════════╝

PURPOSE:
    Runs a secondary energy analysis on the raw mic stream (NOT on STT transcripts —
    this runs before STT for zero latency). Uses librosa to detect:
    - RMS energy drops below threshold mid-utterance (phrase boundary)
    - Pauses of 300-600ms that are NOT end-of-turn (shorter than VAD's 400ms silence)
    
    When these patterns appear during LISTENING state and enough time has passed
    since the last backchannel, fires a BACKCHANNEL_OPPORTUNITY event.

    This module runs on a completely independent async track. It consumes the
    same audio stream as VAD but does its own analysis. It never blocks any
    other module.

═══════════════════════════════════════════════════════════════════════════════════
IMPORTS REQUIRED:
═══════════════════════════════════════════════════════════════════════════════════

import asyncio                          # Async operations
import logging                          # Module logger
import time                             # Timing for backchannel cooldown
import numpy as np                      # Audio array operations
import librosa                          # Audio analysis (RMS energy, spectral features)

from core.session import Session, TurnState
from core.event_bus import EventBus, EventType
from input.mic_stream import MicStream

═══════════════════════════════════════════════════════════════════════════════════
CLASSES:
═══════════════════════════════════════════════════════════════════════════════════

──────────────────────────────────────────────────────────────────────────────────
CLASS: CueDetector
──────────────────────────────────────────────────────────────────────────────────
    Detects natural phrase boundaries in user speech for backchannel timing.

    CONSTRUCTOR: __init__(self, session: Session, event_bus: EventBus,
                          mic_stream: MicStream, config: dict)
    ─────────────────────────────────────────────────────────────
        INPUTS:
            - session: Session — For checking TurnState and speech duration
            - event_bus: EventBus — For publishing BACKCHANNEL_OPPORTUNITY events
            - mic_stream: MicStream — Source of raw audio chunks
            - config: dict — The "backchannel" section from config.yaml:
                - min_pause_ms: int (350)
                - min_gap_between_s: int (8)
                - min_user_speech_s: int (2)
                - energy_threshold: float (0.02)
              AND "audio" section:
                - sample_rate: int (16000)
                - chunk_ms: int (30)
        
        INITIALIZES:
            self.session: Session                = session
            self.event_bus: EventBus             = event_bus
            self.mic_stream: MicStream           = mic_stream
            
            # Config
            self.min_pause_ms: int               = config["backchannel"].get("min_pause_ms", 350)
            self.min_gap_between_s: int          = config["backchannel"].get("min_gap_between_s", 8)
            self.min_user_speech_s: int          = config["backchannel"].get("min_user_speech_s", 2)
            self.energy_threshold: float         = config["backchannel"].get("energy_threshold", 0.02)
            self.sample_rate: int                = config["audio"].get("sample_rate", 16000)
            self.chunk_ms: int                   = config["audio"].get("chunk_ms", 30)
            
            # State
            self._last_backchannel_time: float   = 0.0     # When last backchannel fired
            self._user_speech_start: float       = 0.0     # When user started current speech segment
            self._low_energy_start: float        = 0.0     # When energy first dropped below threshold
            self._energy_buffer: list[float]     = []      # Rolling window of RMS values (last 500ms)
            self._audio_queue: asyncio.Queue     = None    # Consumer queue from mic_stream
            
            self._logger: logging.Logger         = logging.getLogger("CueDetector")

    METHODS:
    ─────────────────────────────────────────────────────────────

    async def run(self) -> None
        INPUTS: None
        OUTPUT: None (runs forever)
        WHAT IT DOES:
            1. Gets audio consumer queue: self._audio_queue = mic_stream.add_consumer()
            2. Registers callback for SPEECH_START to track user speech onset
            3. Enters infinite loop:
               a. chunk = await self._audio_queue.get()
               b. If session.state != TurnState.LISTENING: continue
               c. rms = self._compute_rms(chunk)
               d. self._energy_buffer.append(rms)
                  Keep only last ~500ms worth (500/30 ≈ 17 entries)
               e. await self._detect_phrase_boundary(rms)

    def _compute_rms(self, chunk: np.ndarray) -> float
        INPUTS:
            - chunk: np.ndarray — Audio chunk, float32, normalized [-1, 1]
        OUTPUT:
            - float — Root Mean Square energy of the chunk
        WHAT IT DOES:
            1. Computes: np.sqrt(np.mean(chunk ** 2))
            2. Returns the RMS value
        
        RMS gives a measure of audio loudness. Low RMS = silence/pause.
        High RMS = speech or noise.

    async def _detect_phrase_boundary(self, rms: float) -> None
        INPUTS:
            - rms: float — Current chunk's RMS energy
        OUTPUT: None (may fire BACKCHANNEL_OPPORTUNITY event)
        WHAT IT DOES:
            GUARD CONDITIONS (all must be true to fire):
                1. Session state is LISTENING (user is talking)
                2. User has been speaking > self.min_user_speech_s seconds
                3. Time since last backchannel > self.min_gap_between_s seconds
            
            DETECTION LOGIC:
                If rms < self.energy_threshold:
                    # Energy dropped — possible phrase boundary
                    If self._low_energy_start == 0:
                        self._low_energy_start = time.time()
                    Else:
                        pause_ms = (time.time() - self._low_energy_start) * 1000
                        If self.min_pause_ms <= pause_ms <= 600:
                            # This is a phrase boundary pause (not end-of-turn)
                            # End-of-turn would be caught by VAD at 400ms silence
                            1. Compute context:
                               is_question = self._detect_question_intonation()
                               speech_duration = time.time() - self._user_speech_start
                            2. Publish BACKCHANNEL_OPPORTUNITY event with data:
                               {"is_question": is_question, "energy_level": avg_energy, "speech_duration_s": speech_duration}
                            3. Set self._last_backchannel_time = time.time()
                            4. Reset self._low_energy_start = 0
                            5. Log: "Backchannel opportunity detected"
                Else:
                    # Energy is above threshold — speech resumed
                    self._low_energy_start = 0

    def _detect_question_intonation(self) -> bool
        INPUTS: None (uses self._energy_buffer)
        OUTPUT: bool — Whether the recent speech pattern suggests a question
        WHAT IT DOES:
            1. Looks at the last 10 entries in self._energy_buffer
            2. If energy was rising before the pause (upward intonation), returns True
            3. Simple heuristic: if energy_buffer[-3] < energy_buffer[-6], likely question
            4. Returns False by default
        
        NOTE: This is a rough heuristic. True pitch analysis would require
              F0 extraction which is more expensive. This simple energy-based
              approach works "well enough" for backchannel selection.

    def _on_speech_start(self) -> None
        INPUTS: None
        OUTPUT: None
        WHAT IT DOES:
            1. Sets self._user_speech_start = time.time()
            2. Clears self._energy_buffer
            3. Resets self._low_energy_start = 0
        
        Called when SPEECH_START event fires (registered in run()).

    def reset(self) -> None
        INPUTS: None
        OUTPUT: None
        WHAT IT DOES:
            Resets all state: _low_energy_start=0, _energy_buffer=[], _user_speech_start=0

═══════════════════════════════════════════════════════════════════════════════════
EXPORTS:
    - CueDetector   (class)
═══════════════════════════════════════════════════════════════════════════════════

HOW THIS DIFFERS FROM VAD:
    VAD answers: "Is the user speaking or silent?"
    CueDetector answers: "Is this a natural pause within speech where I should react?"
    
    VAD triggers SPEECH_END (end of turn). CueDetector triggers BACKCHANNEL_OPPORTUNITY
    (natural reaction point within a turn). They look at the same audio but serve
    completely different purposes.
"""



"""
VOXCORE — backchannel/cue_detector.py
Phrase boundary detection — finds natural backchannel opportunities.
"""

import asyncio
import logging
import time
import numpy as np

from core.session import Session, TurnState
from core.event_bus import EventBus, EventType
from input.mic_stream import MicStream

# Number of energy buffer entries for ~500ms at 30ms chunks
_ENERGY_BUFFER_SIZE = 17
# Upper bound for phrase boundary pause (ms) — beyond this, VAD handles it
_MAX_PAUSE_MS = 600


class CueDetector:
    """Detects natural phrase boundaries in user speech for backchannel timing."""

    def __init__(
        self,
        session: Session,
        event_bus: EventBus,
        mic_stream: MicStream,
        config: dict,
    ) -> None:
        self.session: Session = session
        self.event_bus: EventBus = event_bus
        self.mic_stream: MicStream = mic_stream

        # Config — backchannel
        bc_cfg = config.get("backchannel", {})
        self.min_pause_ms: int = bc_cfg.get("min_pause_ms", 350)
        self.min_gap_between_s: int = bc_cfg.get("min_gap_between_s", 8)
        self.min_user_speech_s: int = bc_cfg.get("min_user_speech_s", 2)
        self.energy_threshold: float = bc_cfg.get("energy_threshold", 0.02)

        # Config — audio
        audio_cfg = config.get("audio", {})
        self.sample_rate: int = audio_cfg.get("sample_rate", 16000)
        self.chunk_ms: int = audio_cfg.get("chunk_ms", 30)

        # State
        self._last_backchannel_time: float = 0.0
        self._user_speech_start: float = 0.0
        self._low_energy_start: float = 0.0
        self._energy_buffer: list[float] = []
        self._audio_queue: asyncio.Queue | None = None

        self._logger: logging.Logger = logging.getLogger("CueDetector")

    async def run(self) -> None:
        """Main loop — consume mic audio and detect phrase boundaries."""
        self._audio_queue = self.mic_stream.add_consumer()

        # Register for speech start events
        self.event_bus.subscribe(EventType.SPEECH_START, self._on_speech_start_event)

        self._logger.info("CueDetector running")

        while True:
            try:
                chunk = await self._audio_queue.get()

                # Only analyze during LISTENING state
                if self.session.state != TurnState.LISTENING:
                    continue

                rms = self._compute_rms(chunk)

                # Maintain rolling energy buffer (~500ms window)
                self._energy_buffer.append(rms)
                if len(self._energy_buffer) > _ENERGY_BUFFER_SIZE:
                    self._energy_buffer = self._energy_buffer[-_ENERGY_BUFFER_SIZE:]

                await self._detect_phrase_boundary(rms)

            except asyncio.CancelledError:
                self._logger.info("CueDetector cancelled")
                break
            except Exception:
                self._logger.exception("Error in CueDetector loop")
                await asyncio.sleep(0.1)

    @staticmethod
    def _compute_rms(chunk: np.ndarray) -> float:
        """Compute Root Mean Square energy of an audio chunk."""
        return float(np.sqrt(np.mean(chunk ** 2)))

    async def _detect_phrase_boundary(self, rms: float) -> None:
        """Check if current energy pattern indicates a phrase boundary."""
        now = time.time()

        # Guard: must be in LISTENING state
        if self.session.state != TurnState.LISTENING:
            return

        # Guard: user must have been speaking long enough
        if self._user_speech_start == 0:
            return
        if now - self._user_speech_start < self.min_user_speech_s:
            return

        # Guard: cooldown since last backchannel
        if now - self._last_backchannel_time < self.min_gap_between_s:
            return

        # Detection logic
        if rms < self.energy_threshold:
            # Energy dropped — possible phrase boundary
            if self._low_energy_start == 0:
                self._low_energy_start = now
            else:
                pause_ms = (now - self._low_energy_start) * 1000

                if self.min_pause_ms <= pause_ms <= _MAX_PAUSE_MS:
                    # Phrase boundary detected
                    is_question = self._detect_question_intonation()
                    speech_duration = now - self._user_speech_start
                    avg_energy = (
                        float(np.mean(self._energy_buffer))
                        if self._energy_buffer
                        else 0.0
                    )

                    await self.event_bus.publish(
                        EventType.BACKCHANNEL_OPPORTUNITY,
                        {
                            "is_question": is_question,
                            "energy_level": avg_energy,
                            "speech_duration_s": speech_duration,
                        },
                    )

                    self._last_backchannel_time = now
                    self._low_energy_start = 0
                    self._logger.debug("Backchannel opportunity detected")
        else:
            # Energy above threshold — speech resumed
            self._low_energy_start = 0

    def _detect_question_intonation(self) -> bool:
        """Heuristic: check if energy was rising before pause (upward intonation)."""
        buf = self._energy_buffer
        if len(buf) < 10:
            return False

        # Compare energy at two points before the pause
        # Rising energy before silence suggests question intonation
        recent = buf[-3] if len(buf) >= 3 else 0.0
        earlier = buf[-6] if len(buf) >= 6 else 0.0

        return recent > earlier

    async def _on_speech_start_event(self, data: dict) -> None:
        """Callback for SPEECH_START event."""
        self._on_speech_start()

    def _on_speech_start(self) -> None:
        """Reset tracking state when user starts a new speech segment."""
        self._user_speech_start = time.time()
        self._energy_buffer.clear()
        self._low_energy_start = 0

    def reset(self) -> None:
        """Reset all internal state."""
        self._low_energy_start = 0
        self._energy_buffer.clear()
        self._user_speech_start = 0